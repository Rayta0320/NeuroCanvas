"""
train_personal.py — 本人モデルの学習・保存
実行:
  python3 personal/train_personal.py --blend 0.7 --temp 1.5
出力:
  preprocessed_data_neurosense/va_personal_model.joblib
"""

import os
import sys
import glob
import time
import argparse

import numpy as np
import joblib
import lightgbm as lgb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))       # va_prediction_model/
sys.path.insert(0, os.path.join(ROOT, "touchdesigner"))
import va_features as vf                                                  # numpy/scipyのみ
from scipy import signal

REC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings")
GLOBAL_MODEL = os.path.join(ROOT, "preprocessed_data_neurosense", "va_window_model.joblib")
OUT_MODEL = os.path.join(ROOT, "preprocessed_data_neurosense", "va_personal_model.joblib")

ORIG_FS, FS, DECIM = 256, 128, 2
WIN_SEC, WIN_STEP = 16.0, 2.0
LABEL_NAMES = ["valence", "arousal"]


def eeg_to_windows(eeg):
    #録音済みの生EEG全体(4, N*256) -> (n_win, 61) を、モデル学習・評価用の特徴量窓に変換する
    if eeg.shape[1] < ORIG_FS * WIN_SEC:
        return np.empty((0, 61), np.float32)
    dec = signal.decimate(eeg.astype(np.float64), DECIM, ftype="fir", axis=1)  # (4, N/2 @128)
    w, s = int(WIN_SEC * FS), int(WIN_STEP * FS)
    feats = [vf.window_features(dec[:, i:i + w])
             for i in range(0, dec.shape[1] - w + 1, s)]
    return np.stack(feats).astype(np.float32) if feats else np.empty((0, 61), np.float32)

#特徴量の名前
def feature_base_names():
    bands = list(vf.BANDS.keys())
    base = []
    for c in vf.CH:
        for bn in bands:
            base.append(f"DE_{c}_{bn}")
    base.append("FAA_alpha_AF8-AF7")
    for l, r in vf.LEFT_RIGHT_PAIRS:
        for bn in bands:
            base.append(f"DASM_{l}-{r}_{bn}")
            base.append(f"RASM_{l}-{r}_{bn}")
    for c in vf.CH:
        base.append(f"ratio_beta-alpha_{c}")
        base.append(f"ratio_theta-beta_{c}")
    for c in vf.CH:
        base.append(f"Hjorth_activity_{c}")
        base.append(f"Hjorth_mobility_{c}")
        base.append(f"Hjorth_complexity_{c}")
    return base


def main():
    ap = argparse.ArgumentParser(description="本人モデル(soft分類→V/A)の学習")
    ap.add_argument("--rec", default=None, help="録音npz（未指定なら最新）")
    ap.add_argument("--blend", type=float, default=0.7, help="本人モデルの重み(残りは全体モデル)")
    ap.add_argument("--temp", type=float, default=1.5, help="確率のシャープ化(>1で派手)")
    ap.add_argument("--out", default=OUT_MODEL)
    args = ap.parse_args()

    rec = args.rec or (sorted(glob.glob(os.path.join(REC_DIR, "calib_*.npz")))[-1]
                       if glob.glob(os.path.join(REC_DIR, "calib_*.npz")) else None)
    if not rec or not os.path.exists(rec):
        print(f"[ERROR] 録音が無い: {rec}\n  先に python3 personal/record_calibration.py")
        sys.exit(1)
    d = np.load(rec, allow_pickle=True)
    emotions = [str(e) for e in d["emotions"]]
    anchors_all = d["anchors"].astype(np.float32)  #(n_emotion, 2) V,A
    print(f"録音: {os.path.basename(rec)}  感情={emotions}")

    X, ycls, used_idx = [], [], []
    for ei, name in enumerate(emotions):
        key = f"eeg_{name}"
        if key not in d:
            continue
        F = eeg_to_windows(d[key])
        if len(F) == 0:
            print(f"   [skip] {name}: 窓が作れない（録音が短い）"); continue
        X.append(F); ycls += [ei] * len(F); used_idx.append(ei)
        print(f"   {name:11s}: {len(F):4d} 窓")
    if len(set(ycls)) < 2:
        print("[ERROR] 2感情以上の有効データが必要"); sys.exit(1)
    X = np.concatenate(X).astype(np.float32)
    ycls = np.asarray(ycls)
    anchors = anchors_all                               

    #本人スケール(z化)
    mu = X.mean(0).astype(np.float32)
    sd = (X.std(0) + 1e-8).astype(np.float32)
    Xz = (X - mu) / sd

    # --- ソフト分類器（4感情）---
    print("\n[学習] LightGBM multiclass（class_weight=balanced）")
    clf = lgb.LGBMClassifier(objective="multiclass", num_class=len(emotions),
                             n_estimators=300, learning_rate=0.03, num_leaves=15,
                             subsample=0.8, colsample_bytree=0.8,
                             class_weight="balanced", verbose=-1)
    clf.fit(Xz, ycls)

    #全体データで学習した既存モデルを読み込んで、個人モデルに一緒に入れる
    gbm_global = None
    if os.path.exists(GLOBAL_MODEL):
        gb = joblib.load(GLOBAL_MODEL)
        gbm_global = gb.get("gbm")

    #valence: ソフト分類 -> 感情のV座標---
    v_anchors = anchors[:, 0].astype(np.float32)         # 感情ごとの valence
    P = clf.predict_proba(Xz)
    Ps = P ** args.temp; Ps /= Ps.sum(1, keepdims=True)
    v_pred = Ps @ v_anchors

    #arousal: β/α 比（生理指標）を本人レンジ[5%,95%]で 0〜1 に張り直す
    base_names = feature_base_names()
    ba_idx = [i for i, n in enumerate(base_names) if n.startswith("ratio_beta-alpha")]
    ba = X[:, ba_idx].mean(1)                            
    ba_low, ba_high = np.percentile(ba, [5, 95])
    for ei, name in enumerate(emotions):
        m = ycls == ei

    bundle = {
        "model_kind": "personal_hybrid",
        "clf": clf,
        "emotion_names": emotions,
        "v_anchors": v_anchors,
        "temperature": float(args.temp),
        "blend_weight": float(args.blend),  #　valence 本人 : 全体(BIDS) = blend : (1-blend)
        "gbm_global": gbm_global,           # BIDS全体モデル。valenceブレンド用
        # arousal: β/α 比を本人レンジで0〜1に
        "ba_idx": np.asarray(ba_idx, dtype=int),
        "ba_low": float(ba_low), "ba_high": float(ba_high),
        # 共通
        "global_mu": mu, "global_sd": sd,   # 本人スケール（z化既定）
        "label_names": LABEL_NAMES,
        "fs": FS, "win_sec": WIN_SEC, "win_step": WIN_STEP,
        "channels": ["TP9", "AF7", "AF8", "TP10"],
        "n_train_windows": int(len(X)),
        "source_recording": os.path.basename(rec),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "valence=softmax(P^temp)@v_anchors blend gbm_global[:,0]; arousal=clip((mean(βα)-low)/(high-low))",
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump(bundle, args.out)
    print(f"\n保存: {args.out}")
    print("次: TDの Modelpath をこの va_personal_model.joblib に向けて Re-Init")

if __name__ == "__main__":
    main()
