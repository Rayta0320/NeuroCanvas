"""
リアルタイム推論用に窓単位のモデルの学習・保存

neurosense.pyで4秒窓ごとの特徴(61次元)を作り、各窓にトライアルの V/A(0〜1) を付けて学習する。
→ .joblibを作る
osc_server.pyでLoadしてTouchDesigner にV/Aを返す。

学習は被験者内で正規化。
新しい装着者には本人統計が無いので、
    - 学習データ全体の平均/分散(global)を .joblib に保存しておき、装着直後のキャリブレーションで本人の平均/分散に置き換える。

実行:
    python3 train_window.py  #全30被験者で学習・保存
"""

import os
import sys
import time
import json
import argparse

import numpy as np

import joblib
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb
from neurosense import (load_neurosense_windows, feature_names,
                        BIDS_DIR, OUT_DIR, LABEL_NAMES, LABEL_MIN, LABEL_MAX,
                        get_features)

MODEL_PATH = os.path.join(OUT_DIR, "va_window_model.joblib")
SVR_MAX = 15000          

WIN_SEC = 16.0
WIN_STEP = 2.0
LGBM_PARAMS = dict(n_estimators=600, learning_rate=0.02, num_leaves=31,
                   min_child_samples=80, subsample=0.8, colsample_bytree=0.8, verbose=-1)


def fit_model(X, y):
    #LightGBM単体 -> V/A
    m = MultiOutputRegressor(lgb.LGBMRegressor(**LGBM_PARAMS)); m.fit(X, y); return m

def predict_model(m, X):
    return m.predict(X)

#SVR + LightGBM
def build_model():
    svr = MultiOutputRegressor(SVR(C=1.0, kernel="rbf", gamma="scale"))
    gbm = MultiOutputRegressor(lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.03, num_leaves=15,
        subsample=0.8, colsample_bytree=0.8, verbose=-1))
    return svr, gbm

def per_subject_zscore(X, subjects):
    #被験者内で平均0・分散1に正規化
    Xz = np.empty_like(X)
    for s in np.unique(subjects):
        m = subjects == s
        mu, sd = X[m].mean(0), X[m].std(0) + 1e-8
        Xz[m] = (X[m] - mu) / sd
    return Xz

#Concordance Correlation Coefficient: precision and accuracy
def ccc(y_true, y_pred):
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    mt, mp = yt.mean(), yp.mean()
    vt, vp = yt.var(), yp.var()
    cov = ((yt - mt) * (yp - mp)).mean()
    return 2 * cov / (vt + vp + (mt - mp) ** 2 + 1e-12)

def window_feature_names():
    """window_features() の並び順（61次元）。neurosense.feature_names() の base と一致。"""
    full = feature_names() #122 = mean + std
    half = len(full) // 2
    #mean と stdにそれぞれ
    return [n.split("|", 1)[1] for n in full[:half]]

#SVR(RBF)+LightGBM
def fit_ensemble(Xz, y, seed=0):
    svr, gbm = build_model()
    rng = np.random.default_rng(seed)
    if len(Xz) > SVR_MAX:
        idx = rng.choice(len(Xz), SVR_MAX, replace=False)
        svr.fit(Xz[idx], y[idx])
    else:
        svr.fit(Xz, y)
    gbm.fit(Xz, y)
    return svr, gbm

def predict_ensemble(svr, gbm, Xz):
    return 0.5 * (svr.predict(Xz) + gbm.predict(Xz))

def quick_eval(Xw, yw, subjects, n_test=6, seed=0):
    subs = np.unique(subjects)
    rng = np.random.default_rng(seed)
    n_test = min(n_test, max(1, len(subs) // 5))      #全体の２割か最低１人はTest
    test_subs = set(rng.choice(subs, n_test, replace=False))
    te = np.array([s in test_subs for s in subjects])
    tr = ~te
    # z化は各人内で正規化
    Xz = per_subject_zscore(Xw, subjects)
    print(f"学習{tr.sum()}窓 / テスト{te.sum()}窓 "
          f"（テスト被験者 {sorted(test_subs)}）")
    t0 = time.time()
    m = fit_model(Xz[tr], yw[tr]) #LightGBM単体
    pred = predict_model(m, Xz[te])
    print(f"学習{time.time()-t0:.1f}s")
    for j, n in enumerate(LABEL_NAMES):
        err = pred[:, j] - yw[te, j]
        mse = float(np.mean(err ** 2)); base = float(np.var(yw[te, j]))
        verdict = "OK" if mse < base else "No"
        print(f"   {n:8s} | MSE={mse:.4f} (base {base:.4f}) {verdict} "
              f"CCC={ccc(yw[te,j],pred[:,j]):.3f} "
              f"PCC={np.corrcoef(yw[te,j],pred[:,j])[0,1]:.3f}") #relationship btw true and pred
    return pred, yw[te] 


def deshrink_stats(oos_pred, oos_y, yw):
    """
    MSE -> モデルの予測は中心によりやすい。
    →p = label_mean + (p - pred_mean) * (label_std / pred_std)
    で予測を外に引き延ばす
    """
    label_mean = yw.mean(0).astype(np.float32)
    label_std = yw.std(0).astype(np.float32)
    pred_mean = oos_pred.mean(0).astype(np.float32)
    pred_std = (oos_pred.std(0) + 1e-6).astype(np.float32)
    return {
        "deshrink_label_mean": label_mean,
        "deshrink_label_std": label_std,
        "deshrink_pred_mean": pred_mean,
        "deshrink_pred_std": pred_std,
    }

def train_and_save(bids_dir=BIDS_DIR, limit=None, do_eval=True,
                   model_path=MODEL_PATH):
    print(f"窓単位モデルの学習（リアルタイム推論用, 窓={WIN_SEC}s / LightGBM単体）")
    # 窓特徴をキャッシュから取る
    Xw, yw, subjects = get_features(WIN_SEC, limit=limit)
    yw = yw.astype(np.float32)

    quick_eval(Xw, yw, subjects)

    print("\n全窓を per-subject z-score して LightGBM単体で学習")
    Xz = per_subject_zscore(Xw, subjects)
    t0 = time.time()
    model = fit_model(Xz, yw)
    print(f"学習完了 ({time.time()-t0:.1f}s)")

    global_mu = Xw.mean(0).astype(np.float32)
    global_sd = (Xw.std(0) + 1e-8).astype(np.float32)

    bundle = {
        "svr": None,  #結局LightGBM単体で学習（SVRは長窓で精度低下）
        "gbm": model,
        "global_mu": global_mu, # window_features の平均
        "global_sd": global_sd, # 標準偏差
        "feature_names": window_feature_names(),
        "label_names": LABEL_NAMES,
        "label_raw_scale": [LABEL_MIN, LABEL_MAX],
        "fs": 128, "win_sec": WIN_SEC, "win_step": WIN_STEP,
        "channels": ["TP9", "AF7", "AF8", "TP10"],
        "n_train_windows": int(len(Xw)),
        "n_subjects": int(len(np.unique(subjects))),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": f"predict = gbm.predict; 入力は window_features(61, {WIN_SEC}s窓)を z化",
    }
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(bundle, model_path)
    print(f"\n保存: {model_path}")
    print(f"  窓数={bundle['n_train_windows']}  被験者={bundle['n_subjects']}  "
          f"特徴={len(global_mu)}次元")
    return model_path


def main():
    ap = argparse.ArgumentParser(description="窓単位 V/A モデルの学習・保存")
    ap.add_argument("--bids-dir", default=BIDS_DIR)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--model-path", default=MODEL_PATH)
    args = ap.parse_args()
    train_and_save(bids_dir=args.bids_dir, limit=args.limit,
                   do_eval=not args.no_eval, model_path=args.model_path)


if __name__ == "__main__":
    main()
