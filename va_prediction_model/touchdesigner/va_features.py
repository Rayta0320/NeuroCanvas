# 4channel EEG -> V/A prediction feature

"""
win: shape (4ch, samples) -> 61次元の特徴ベクトル
5つの特徴量が使われている。
1) Differential Entropy: powerをlogのスケールで表現。4ch * 5帯域 = 20次元
2) Frontal Alpha Asymmetry: 前頭部のalpha の左右差 = 1次元
3) Differential Asymmetry & Rational Asymmetry: 左右対称ペア * 5帯域 * 2種類 = 20次元
4) 帯域比: beta/alpha と theta/beta を各chで 4ch * 2 = 8次元
5) Hjorth: EEG波形の振幅の大きさ、早さやどのような早さで行われているかをみる 4ch * 3 = 12次元
"""

import numpy as np
from scipy import signal

FS = 128                       # 特徴抽出のサンプリング（256Hzから間引いた後）
WIN_SEC, WIN_STEP = 4.0, 2.0
CH = ["TP9", "AF7", "AF8", "TP10"]  #4chの並び
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13),
         "beta": (13, 30), "gamma": (30, 45)}
LEFT_RIGHT_PAIRS = [("AF7", "AF8"), ("TP9", "TP10")]   # 対称ペア(左,右)

def _bandpass(x, lo, hi, fs=FS):
    b, a = signal.butter(4, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return signal.filtfilt(b, a, x)

def _band_power(x, band, fs=FS):
    xf = _bandpass(x, band[0], band[1], fs)
    return np.var(xf) + 1e-12          

# 微分エントロピー
def dif_e(power):
    return 0.5 * np.log(2 * np.pi * np.e * power)   

# activity, mobility, complexity
def _hjorth(x):
    dx, ddx = np.diff(x), np.diff(np.diff(x))
    v0, v1, v2 = np.var(x) + 1e-12, np.var(dx) + 1e-12, np.var(ddx) + 1e-12
    mob = np.sqrt(v1 / v0)
    return v0, mob, np.sqrt(v2 / v1) / mob          


def window_features(win):
    ch = {name: win[i] for i, name in enumerate(CH)}
    powers = {(c, bn): _band_power(ch[c], b) for c in CH for bn, b in BANDS.items()}
    de = {k: dif_e(v) for k, v in powers.items()}

    feats = []
    # DE: 4ch * 5帯域 = 20
    feats += [de[(c, bn)] for c in CH for bn in BANDS]
    # FAA: alpha の左右おでこ差 (valence の主力)
    feats += [de[("AF8", "alpha")] - de[("AF7", "alpha")]]
    # DASM / RASM: 対称ペアの5帯域における差と比
    for l, r in LEFT_RIGHT_PAIRS:
        for bn in BANDS:
            feats.append(de[(l, bn)] - de[(r, bn)])              
            feats.append(powers[(l, bn)] / powers[(r, bn)])        
    # 帯域比: beta/alpha, theta/beta (arousal 側)
    for c in CH:
        feats.append(powers[(c, "beta")] / powers[(c, "alpha")])
        feats.append(powers[(c, "theta")] / powers[(c, "beta")])
    # Hjorth: 4ch x 3
    for c in CH:
        feats += list(_hjorth(ch[c]))
    return np.asarray(feats, dtype=np.float32)
