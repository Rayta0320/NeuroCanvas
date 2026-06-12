"""
NeuroSense (BIDS, Muse 4ch/256Hz) から特徴量データを作る前処理

データ構造:
    BIDS/sub-IDxxx/ses-Sxxx/eeg/*.edf      4ch, 256Hz, 64秒/セッション(=1曲)
                            .../*_eeg.json  TaskDescription 内に AVG_Valence/AVG_Arousal

実行:
    python3 neurosense.py             
"""

import os
import re
import sys
import json
import time
import glob
import argparse

import numpy as np
from scipy import signal

import pyedflib

FS = 128                   
WIN_SEC, WIN_STEP = 4.0, 2.0  

# Muse2 のチャネル順
CH = ["TP9", "AF7", "AF8", "TP10"]
LABEL_NAMES = ["valence", "arousal"]
LABEL_MIN, LABEL_MAX = 1.0, 9.0   # DEAP/NeuroSense の自己評価は 1〜9（0〜1正規化に使う）

MIN_SAMPLES = int(WIN_SEC * FS)   # 4秒窓に満たないトライアルは特徴を作れない

def normalize_labels(y):
    """生の DEAP ラベル(1〜9) を 0〜1 に線形正規化"""
    return (np.asarray(y, dtype=np.float32) - LABEL_MIN) / (LABEL_MAX - LABEL_MIN)

#All features name -> mean & std of features
def feature_names():
    bands = list(BANDS.keys())                    # delta, theta, alpha, beta, gamma
    base = []
    # DE : 4ch × 5帯域
    for c in CH:
        for bn in bands:
            base.append(f"DE_{c}_{bn}")
    # FAA : alpha の左右おでこ差
    base.append("FAA_alpha_AF8-AF7")
    # DASM / RASM : 対称ペア(AF7-AF8 / TP9-TP10) × 5帯域
    for l, r in LEFT_RIGHT_PAIRS:
        for bn in bands:
            base.append(f"DASM_{l}-{r}_{bn}")
            base.append(f"RASM_{l}-{r}_{bn}")
    # 帯域比 : beta/alpha, theta/beta
    for c in CH:
        base.append(f"ratio_beta-alpha_{c}")
        base.append(f"ratio_theta-beta_{c}")
    # Hjorth : activity, mobility, complexity
    for c in CH:
        base.append(f"Hjorth_activity_{c}")
        base.append(f"Hjorth_mobility_{c}")
        base.append(f"Hjorth_complexity_{c}")
    # extract_trial_features は [窓平均(61) ∥ 窓標準偏差(61)] を連結 -> 122次元
    return [f"mean|{b}" for b in base] + [f"std|{b}" for b in base]

ROOT = os.path.dirname(os.path.abspath(__file__))
BIDS_DIR = os.path.join(ROOT, "BIDS")
OUT_DIR = os.path.join(ROOT, "preprocessed_data_neurosense")

NS_FS = 256                       # NeuroSense の元サンプリング
DECIM = NS_FS // FS               # 256 -> 128 は 2 で間引き

#Loading EDF file
def parse_va(json_path):
    """eeg.json の TaskDescription から AVG_Valence / AVG_Arousal を取り出す"""
    d = json.load(open(json_path))
    t = d.get("TaskDescription", "")
    v = re.search(r"AVG_Valence:\s*([\d.]+)", t)
    a = re.search(r"AVG_Arousal:\s*([\d.]+)", t)
    if not (v and a):
        return None
    return float(v.group(1)), float(a.group(1))

def read_edf_4ch(edf_path):
    """EDF を読み、va_prediction の並び(TP9,AF7,AF8,TP10)・128Hz に整える -> (4, samples)"""
    r = pyedflib.EdfReader(edf_path)
    names = r.getSignalLabels()
    sigs = {nm: r.readSignal(i) for i, nm in enumerate(names)}
    r.close()
    # sort in CH order（AF7/TP9/TP10/AF8）
    arr = np.stack([sigs[c] for c in CH]).astype(np.float64)   # (4, 256Hz*64)
    # Using Anti-Aliasing Filter(FIR) to reduce Hz: 256 -> 128 
    arr = signal.decimate(arr, DECIM, ftype="fir", axis=1) 
    return arr

#Extracting features
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13),
         "beta": (13, 30), "gamma": (30, 45)}
LEFT_RIGHT_PAIRS = [("AF7", "AF8"), ("TP9", "TP10")]  # 対称ペア(左,右)

def _bandpass(x, lo, hi, fs=FS):
    b, a = signal.butter(4, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return signal.filtfilt(b, a, x)

def _band_power(x, band, fs=FS):
    xf = _bandpass(x, band[0], band[1], fs)
    return np.var(xf) + 1e-12          # 線形パワー(常に正)

def dif_e(power):
    return 0.5 * np.log(2 * np.pi * np.e * power)   # 微分エントロピー

def _hjorth(x):
    dx, ddx = np.diff(x), np.diff(np.diff(x))
    v0, v1, v2 = np.var(x) + 1e-12, np.var(dx) + 1e-12, np.var(ddx) + 1e-12
    mob = np.sqrt(v1 / v0)
    return v0, mob, np.sqrt(v2 / v1) / mob          # activity, mobility, complexity

def window_features(win): #(4, 4*128)
    """win: shape (4ch, samples) -> 1次元の特徴ベクトル
    DE : 20
    FAA : 1
    DASM/RASM: 20
    帯域比 : 8
    Hjorth : 12
    => 61
    """
    ch = {name: win[i] for i, name in enumerate(CH)}
    powers = {(c, bn): _band_power(ch[c], b) for c in CH for bn, b in BANDS.items()}
    de = {k: dif_e(v) for k, v in powers.items()}

    feats = []
    # DE: 4ch x 5帯域 = 20
    feats += [de[(c, bn)] for c in CH for bn in BANDS]
    # FAA: alpha の左右おでこ差 (valence の主力)
    feats += [de[("AF8", "alpha")] - de[("AF7", "alpha")]]
    # DASM / RASM: 対称ペア x 5帯域
    for l, r in LEFT_RIGHT_PAIRS:
        for bn in BANDS:
            feats.append(de[(l, bn)] - de[(r, bn)])                 # DASM(差)
            feats.append(powers[(l, bn)] / powers[(r, bn)])         # RASM(比)
    # 帯域比: beta/alpha, theta/beta (arousal 側)
    for c in CH:
        feats.append(powers[(c, "beta")] / powers[(c, "alpha")])
        feats.append(powers[(c, "theta")] / powers[(c, "beta")])
    # Hjorth: 4ch x 3
    for c in CH:
        feats += list(_hjorth(ch[c]))
    return np.asarray(feats, dtype=np.float32)

#長さ４以上のWindowであればCacheから特徴を取ってくる
def cache_path(win_sec):
    if abs(win_sec - 4.0) < 1e-6:
        return os.path.join(OUT_DIR, "window_features_cache.npz")     
    return os.path.join(OUT_DIR, f"window_features_cache_w{int(win_sec)}.npz")

#16秒窓を2秒ずつずらしながら特徴抽出する -> 61dim feature -> V/A Label + subID
def extract_windows(win_sec, win_step=WIN_STEP, bids_dir=BIDS_DIR, limit=None):
    w, s = int(win_sec * FS), int(win_step * FS)
    subs_dirs = sorted(glob.glob(os.path.join(bids_dir, "sub-*")))
    if limit:
        subs_dirs = subs_dirs[:limit]
    Xs, ys, subj = [], [], []
    t0 = time.time()
    print(f"[extract] win={win_sec}s step={win_step}s  ({len(subs_dirs)}被験者)")
    for si, sd in enumerate(subs_dirs, 1):
        sid = os.path.basename(sd)
        for edf in sorted(glob.glob(os.path.join(sd, "ses-*", "eeg", "*_eeg.edf"))):
            jf = edf.replace("_eeg.edf", "_eeg.json")
            va = parse_va(jf) if os.path.exists(jf) else None
            if va is None:
                continue
            try:
                trial = read_edf_4ch(edf)
            except Exception:
                continue
            if trial.shape[1] < w:
                continue
            for i in range(0, trial.shape[1] - w + 1, s):
                Xs.append(window_features(trial[:, i:i + w]))
                ys.append(va); subj.append(sid)
        if si % 10 == 0:
            print(f"   {si}/{len(subs_dirs)}  ({time.time()-t0:.0f}s)")
    return (np.stack(Xs).astype(np.float32),
            (np.asarray(ys, np.float32) - 1.0) / 8.0, #1〜9 -> 0〜1
            np.asarray(subj))

#cache or make to get feature 
def get_features(win_sec, limit=None):
    cp = cache_path(win_sec)
    if os.path.exists(cp) and limit is None:
        d = np.load(cp, allow_pickle=True)
        print(f"[cache] {os.path.basename(cp)}  Xw{d['Xw'].shape}")
        return d["Xw"], d["yw"], d["subjects"]
    Xw, yw, subs = extract_windows(win_sec, limit=limit)
    if limit is None:
        np.savez_compressed(cp, Xw=Xw, yw=yw, subjects=subs)
        print(f"[cache] 保存: {os.path.basename(cp)}  Xw{Xw.shape}")
    return Xw, yw, subs


def extract_trial_features(trial, fs=FS): #trial(4, ~8192)
    """trial: (4ch, samples) -> 窓ごとに特徴を出し、平均と標準偏差で1本に集約"""
    w, s = int(WIN_SEC * fs), int(WIN_STEP * fs) #窓サイズとステップサイズを設定 (4*128, 2*128)
    wins = [window_features(trial[:, i:i + w])
            for i in range(0, trial.shape[1] - w + 1, s)]
    wins = np.stack(wins)
    return np.concatenate([wins.mean(0), wins.std(0)])   # mean & std

def load_neurosense(bids_dir=BIDS_DIR, limit=None): #limit -> limit # of subjects
    subs_dirs = sorted(glob.glob(os.path.join(bids_dir, "sub-*")))
    if limit:
        subs_dirs = subs_dirs[:limit]
    if not subs_dirs:
        raise FileNotFoundError(f"sub-* が見つかりません: {bids_dir}")

    print(f"対象: {len(subs_dirs)} 被験者  ({bids_dir})")
    Xs, ys, subs = [], [], [] #Xs: Feature vec of EEG, ys: V/A label, subs: subject ID
    t0 = time.time()
    for si, sd in enumerate(subs_dirs, 1):
        sid = os.path.basename(sd)
        edfs = sorted(glob.glob(os.path.join(sd, "ses-*", "eeg", "*_eeg.edf")))
        n_ok = 0
        for edf in edfs:
            jf = edf.replace("_eeg.edf", "_eeg.json")
            va = parse_va(jf) if os.path.exists(jf) else None
            if va is None: #ignore non-label sessions
                continue                                
            try:
                trial = read_edf_4ch(edf)  #(4, ~8192) 128Hz
            except Exception as e:
                print(f"   [skip] {os.path.basename(edf)}: {e}")
                continue
            if trial.shape[1] < MIN_SAMPLES:             # 4秒窓に満たない短いセッション
                print(f"   [skip] {os.path.basename(edf)}: too short "
                      f"({trial.shape[1]} < {MIN_SAMPLES} samples)")
                continue
            Xs.append(extract_trial_features(trial))
            ys.append(va)
            subs.append(sid)
            n_ok += 1
        print(f"[{si:2d}/{len(subs_dirs)}] {sid}: {n_ok} sessions  "
              f"({time.time()-t0:5.1f}s)")
        if si == 1:
            print(f"[1] EDF読込(4ch*{NS_FS}Hz)  [2] ch並べ替え->{CH}  "
                  f"[3] {NS_FS}->{FS}Hz間引き  [4] V/A(1〜9->0〜1)  [5] 特徴122次元")

    X = np.stack(Xs).astype(np.float32) #EEG Feature
    y = normalize_labels(ys) #V/A Label (to 0-1)
    subjects = np.asarray(subs) #Subject label
    print(f"\n抽出完了: X{X.shape}  y{y.shape} "
          f"(0〜1, range[{y.min():.2f},{y.max():.2f}])  "
          f"被験者{len(np.unique(subjects))}  ({time.time()-t0:.1f}s)")
    return X, y, subjects


def load_neurosense_windows(bids_dir=BIDS_DIR, limit=None):
    """
    リアルタイム推論用の窓単位のデータセット。
    4秒窓ごとに window_features(61次元) を出し、窓単位にラベルを割り当てる。
    return: Xw(n_windows,61), yw(n_windows,2) 0〜1, subjects(n_windows)
    """
    subs_dirs = sorted(glob.glob(os.path.join(bids_dir, "sub-*")))
    if limit:
        subs_dirs = subs_dirs[:limit]
    w, s = int(WIN_SEC * FS), int(WIN_STEP * FS)
    Xs, ys, subj = [], [], []
    t0 = time.time()
    print(f"対象: {len(subs_dirs)} 被験者（窓単位）  ({bids_dir})")
    for si, sd in enumerate(subs_dirs, 1):
        sid = os.path.basename(sd)
        n_win = 0
        for edf in sorted(glob.glob(os.path.join(sd, "ses-*", "eeg", "*_eeg.edf"))):
            jf = edf.replace("_eeg.edf", "_eeg.json")
            va = parse_va(jf) if os.path.exists(jf) else None
            if va is None:
                continue
            try:
                trial = read_edf_4ch(edf)
            except Exception:
                continue
            if trial.shape[1] < MIN_SAMPLES:
                continue
            for i in range(0, trial.shape[1] - w + 1, s):
                Xs.append(window_features(trial[:, i:i + w]))
                ys.append(va)
                subj.append(sid)
                n_win += 1
        print(f"[{si:2d}/{len(subs_dirs)}] {sid}: {n_win} windows  "
              f"({time.time()-t0:5.1f}s)")
    Xw = np.stack(Xs).astype(np.float32)
    yw = normalize_labels(ys)
    subjects = np.asarray(subj)
    print(f"\n窓抽出完了: Xw{Xw.shape}  yw{yw.shape}  "
          f"被験者{len(np.unique(subjects))}  ({time.time()-t0:.1f}s)")
    return Xw, yw, subjects

#Output: neurosense_va_features.npz, meta.json
def save(X, y, subjects, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    names = feature_names()
    npz_path = os.path.join(out_dir, "neurosense_va_features.npz")
    #save preprocessed data as .npz format
    np.savez_compressed(npz_path, X=X, y=y, subjects=subjects,
                        feature_names=np.array(names),
                        label_names=np.array(LABEL_NAMES))
    #data info
    meta = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": "NeuroSense BIDS (Muse 4ch/256Hz)",
        "n_trials": int(X.shape[0]),
        "n_subjects": int(len(np.unique(subjects))),
        "feature_dim": int(X.shape[1]),
        "label_names": LABEL_NAMES,
        "label_normalized": True,
        "label_raw_scale": [LABEL_MIN, LABEL_MAX],
        "label_range": [0.0, 1.0],
        "config": {"orig_fs": NS_FS, "fs": FS, "channels": CH,
                   "label_source": "eeg.json TaskDescription AVG_Valence/AVG_Arousal"},
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as fp:
        json.dump(meta, fp, indent=2, ensure_ascii=False)
    print(f"\n保存: {npz_path}")
    return npz_path

#if results (prediction.npz) exist -> show the results
def show_examples(out_dir=OUT_DIR, n_per_subject=5, n_subjects=3):
    pred_path = os.path.join(out_dir, "results", "predictions.npz")
    if not os.path.exists(pred_path):
        return
    d = np.load(pred_path, allow_pickle=True)
    y, preds, subs = d["y"], d["loso_preds"], d["subjects"]
    if preds.size == 0:
        return
    print("\n" + "=" * 70)
    print("テストデータ(Leave-One-Subject-Out)での 予測 vs 正解 の例")
    print("=" * 70)
    print(f'{"subj":>9} {"trial":>5} | {"V_true":>7} {"V_pred":>7} {"err":>6} '
          f'| {"A_true":>7} {"A_pred":>7} {"err":>6}')
    print("-" * 70)
    for s in np.unique(subs)[:n_subjects]:
        idx = np.where(subs == s)[0][:n_per_subject]
        for k, i in enumerate(idx):
            ve, ae = preds[i, 0] - y[i, 0], preds[i, 1] - y[i, 1]
            print(f'{str(s):>9} {k:>5} | {y[i,0]:7.3f} {preds[i,0]:7.3f} {ve:+6.3f} '
                  f'| {y[i,1]:7.3f} {preds[i,1]:7.3f} {ae:+6.3f}')
    print("-" * 70)
    for j, n in enumerate(LABEL_NAMES):
        print(f'{n:8s}: true mean={y[:,j].mean():.3f} std={y[:,j].std():.3f} | '
              f'pred mean={preds[:,j].mean():.3f} std={preds[:,j].std():.3f} '
              f'(予測がmean付近に固まる=underfit)')


def run(bids_dir=BIDS_DIR, out_dir=OUT_DIR, limit=None, force=False):
    """
    EDF -> 特徴(122次元)を抽出して保存する。
    リアルタイム推論用の窓単位モデル学習は train_window.py が担当。
    """
    npz_path = os.path.join(out_dir, "neurosense_va_features.npz")
    print("NeuroSense : 前処理（EDF -> 特徴ベクトル）")
    if os.path.exists(npz_path) and not force:
        print(f"[skip] 既存を使用: {npz_path}  (作り直すには --force)")
    else:
        X, y, subjects = load_neurosense(bids_dir, limit=limit)
        save(X, y, subjects, out_dir=out_dir)
    print("\n完了。窓単位モデルの学習・保存はpython3 train_window.pyを実行。")

'''
bids_dir: NeuroSense BIDSデータの場所
out_dir: 前処理結果の保存先
limit: 先頭N人だけ処理する、動作確認用
force: 既存ファイルがあっても作り直すか
'''
def main():
    ap = argparse.ArgumentParser(description="NeuroSense 前処理（EDF -> 特徴ベクトル保存）")
    ap.add_argument("--bids-dir", default=BIDS_DIR)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--limit", type=int, default=None, help="先頭N被験者だけ")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    run(bids_dir=args.bids_dir, out_dir=args.out_dir, limit=args.limit,
        force=args.force)

if __name__ == "__main__":
    main()