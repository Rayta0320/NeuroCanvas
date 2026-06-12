import os
import sys
import time
import warnings
from collections import deque

import numpy as np
from scipy import signal

# LightGBM/sklearn が予測ごとに出す無害な警告（特徴名なし）を抑制（TDのtextport対策）
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Muse2 / TD から来る生サンプリング
ORIG_FS = 256      
# 特徴抽出のサンプリング                
FS = 128                           
DECIM = ORIG_FS // FS             
DEFAULT_WIN_SEC = 16.0             
CH_ORDER = ["TP9", "AF7", "AF8", "TP10"]


def _this_dir(owner=None):
    if owner is not None:
        try:
            d = owner.par.Codedir.eval()
            if d:
                return d
        except Exception:
            pass
    return os.getcwd()

class ExtVAInfer:
    def __init__(self, ownerComp):
        self.ownerComp = ownerComp #me.op('ext_va_infer').module.ExtVAInfer(me)
        self.vf = self._import_va_features()          
        model_path = self._resolve_model_path()

        import joblib
        b = joblib.load(model_path)
        self.model_kind = b.get("model_kind", "regression")
        self.mu = np.asarray(b["global_mu"], dtype=np.float64)   
        self.sd = np.asarray(b["global_sd"], dtype=np.float64)
        self.global_mu, self.global_sd = self.mu.copy(), self.sd.copy()
        self.label_names = b.get("label_names", ["valence", "arousal"])

        #BIDSで学習したモデルと個人のモデルを組み合わせる。
        if self.model_kind == "personal_hybrid":
            # valence=感情ソフト分類, arousal=β/α比
            self.clf = b["clf"]
            self.v_anchors = np.asarray(b["v_anchors"], dtype=np.float64)
            self.emotion_names = b.get("emotion_names", [])
            self.temperature = float(b.get("temperature", 1.5))
            #個人モデルと全体モデルのBlend
            self.blend_weight = float(b.get("blend_weight", 0.7))
            self.gbm_global = b.get("gbm_global")
            self.ba_idx = np.asarray(b["ba_idx"], dtype=int)
            self.ba_low = float(b["ba_low"]); self.ba_high = float(b["ba_high"])
            # arousalはb/a比を用いる。中心だけEMA+較正由来の固定スケール
            log_lo = float(np.log(max(self.ba_low, 1e-6)))
            log_hi = float(np.log(max(self.ba_high, 1e-6)))
            self.ba_scale = max((log_hi - log_lo) / 3.3, 0.15)   # 固定の振れ幅(log空間)
            self.ba_mean = None
            self.ba_alpha = 0.01    
            self.ba_gain = 1.0      # sigmoidの広がり（大=派手）
            self.svr = self.gbm = None
        elif self.model_kind == "soft_emotion":
            # 本人モデル: ソフト分類(感情) 
            self.clf = b["clf"]
            self.anchors = np.asarray(b["anchors"], dtype=np.float64)   # (n_emotion,2)
            self.emotion_names = b.get("emotion_names", [])
            self.temperature = float(b.get("temperature", 1.5))
            self.blend_weight = float(b.get("blend_weight", 0.7))       # 本人:全体
            self.gbm_global = b.get("gbm_global")                       # 転移ブレンド用(16s回帰)
            self.svr = self.gbm = None
        else:
            self.svr, self.gbm = b.get("svr"), b["gbm"]   # svr=None なら LightGBM単体

        self.win_sec = float(b.get("win_sec", DEFAULT_WIN_SEC))
        self.win_samples = int(ORIG_FS * self.win_sec)   # 256Hz基準のサンプル数

        self.ema = float(self._par("Ema", 0.3))
        self.cal_sec = float(self._par("Calsec", 60.0))
        self.buf = deque(maxlen=self.win_samples + ORIG_FS)   #生サンプル(4ch)
        self.va = np.array([0.5, 0.5])                   # EMA平滑の状態
        self.calibrating = False
        self.cal_feats = []
        self.cal_start = 0.0
        if self.model_kind == "personal_hybrid":
            kind = f"personal_hybrid(V=soft{len(self.v_anchors)}+BIDS, A=β/α)"
        elif self.model_kind == "soft_emotion":
            kind = f"personal soft_emotion({len(self.anchors)})"
        else:
            kind = "LightGBM" if self.svr is None else "SVR+LightGBM"
        self._set_status(f"loaded {kind} ({len(self.mu)}dim, {self.win_sec:.0f}s窓) — not calibrated")

    def _import_va_features(self):
        #TD内から取ってくる
        try:
            dat = self.ownerComp.op("va_features")
            if dat is not None and hasattr(dat, "module"):
                return dat.module
        except Exception:
            pass
        import va_features as vf
        return vf

    def _resolve_model_path(self):
        code_dir = _this_dir(self.ownerComp)
        default_model = os.path.normpath(os.path.join(
            code_dir, "..", "preprocessed_data_neurosense", "va_window_model.joblib"))
        return self._par("Modelpath", default_model)
    
    #Read Modelpath parameter
    def _par(self, name, default):
        try:
            v = getattr(self.ownerComp.par, name).eval()
            return v if v not in (None, "") else default
        except Exception:
            return default

    def _set_status(self, msg):
        try:
            self.ownerComp.par.Status = msg
        except Exception:
            pass
    
    #Script CHOP -> Save
    def Push(self, arr):
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        # (4, n) を1サンプルずつ append
        for j in range(arr.shape[1]):
            self.buf.append(arr[:, j])

    
    def _current_feature(self):
        if len(self.buf) < self.win_samples:
            return None
        win = np.stack(list(self.buf)[-self.win_samples:], axis=1)   # (4, win_samples: 256)
        dec = signal.decimate(win, DECIM, ftype="fir", axis=1)       # (4, win_samples/2: 128)
        return self.vf.window_features(dec).astype(np.float64)

    def _predict_raw(self, feat):
        z = ((feat - self.mu) / self.sd).reshape(1, -1)
        if self.model_kind == "personal_hybrid":
            # valence: 感情ソフト分類 -> v_anchors（BIDS全体とブレンド）
            p = self.clf.predict_proba(z)[0]
            p = np.power(p, self.temperature); p = p / (p.sum() + 1e-12)
            v = float(p @ self.v_anchors)
            if self.gbm_global is not None and self.blend_weight < 1.0:
                vg = float(np.ravel(self.gbm_global.predict(z))[0])   # BIDSのValence
                #個人モデル＋BIDSで学習されたモデル
                v = self.blend_weight * v + (1.0 - self.blend_weight) * vg
            # arousal: log-median β/α。中心はEMAで滑らかにする
            x = float(np.log(np.mean(feat[self.ba_idx]) + 1e-6))
            if self.ba_mean is None:
                self.ba_mean = x
            else:
                self.ba_mean += self.ba_alpha * (x - self.ba_mean) 
            #正規化
            zz = (x - self.ba_mean) / self.ba_scale                  
            a = 1.0 / (1.0 + np.exp(-self.ba_gain * zz)) #0-1
            out = np.array([v, a])
        #V/Aどちらも回帰モデル
        elif self.model_kind == "soft_emotion":   # 本人モデル: 感情確率 -> V/A
            p = self.clf.predict_proba(z)[0]
            p = np.power(p, self.temperature)
            p = p / (p.sum() + 1e-12)
            out = p @ self.anchors                 # (2,) = Σ P(e)*anchor(e)
            if self.gbm_global is not None and self.blend_weight < 1.0:
                g = self.gbm_global.predict(z)[0]  # 全体モデル(16s回帰)とブレンド
                out = self.blend_weight * out + (1.0 - self.blend_weight) * g
        elif self.svr is None:                     # LightGBM単体
            out = self.gbm.predict(z)[0]
        else:                                       # SVR+LightGBM アンサンブル
            out = 0.5 * (self.svr.predict(z) + self.gbm.predict(z))[0]
        return np.clip(out, 0.0, 1.0)

    #推論
    def Predict(self):
        feat = self._current_feature()
        if feat is None:
            return self.va
        if self.calibrating:
            self._update_calibration(feat)
        raw = self._predict_raw(feat)
        self.va = self.ema * raw + (1 - self.ema) * self.va
        return self.va #[valence, arousal]

    # 装着者の平均/分散で z正規化
    def StartCalibration(self):
        self.cal_feats = []
        self.cal_start = time.time()
        self.calibrating = True
        self._set_status(f"calibrating {self.cal_sec:.0f}s — stay relaxed")

    #Update calibration: if greater than cal_sec or more than 10 stacks
    def _update_calibration(self, feat):
        self.cal_feats.append(feat)
        if time.time() - self.cal_start >= self.cal_sec and len(self.cal_feats) >= 10:
            F = np.stack(self.cal_feats)
            self.mu = F.mean(0)
            self.sd = F.std(0) + 1e-8
            self.calibrating = False
            self._set_status(f"calibrated on {len(self.cal_feats)} windows")

    #original mu and std
    def ResetCalibration(self):
        self.mu, self.sd = self.global_mu.copy(), self.global_sd.copy()
        self.calibrating = False
        self._set_status("calibration reset to global")
