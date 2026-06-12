"""
リアルタイム V/A 推論サーバ（TouchDesigner <-> Python, OSC）
============================================================

TouchDesigner で取り出した Muse2 の生EEG(4ch@256Hz)を OSC で受け取り、
4秒スライド窓ごとに学習済みモデル(va_window_model.joblib)で V/A を予測し、
0〜1 の連続値を OSC で TouchDesigner に返し続ける。

データの流れ:
    Muse2 ─> TouchDesigner ─[OSC /muse/eeg (TP9,AF7,AF8,TP10)]─> 本サーバ
    本サーバ ─[OSC /valence, /arousal (0〜1)]─> TouchDesigner ─> ビジュアル

正規化（キャリブレーション）:
    学習は被験者内z化。新しい装着者には本人統計が無いので、
    起動後 CAL_SEC 秒のあいだの特徴で本人の平均/分散を測り、それでz化する。
    キャリブ前は学習データ全体の平均/分散(global)で代用。OSC /calibrate で再実行。

実行:
    python3 osc_server.py                 # 受信5005 / 送信5006 で常駐
    python3 osc_server.py --selftest      # NeuroSenseの1曲を流し込み動作確認（OSC不要）
    python3 osc_server.py --in-port 5005 --out-port 5006 --out-ip 127.0.0.1

TouchDesigner 側:
    送信: OSC Out CHOP / DAT で  /muse/eeg  に4ch(TP9,AF7,AF8,TP10)を 256Hz で送る
    受信: OSC In CHOP で  /valence  /arousal  を受ける（各 0〜1）
"""

import os
import sys
import time
import argparse
import threading
from collections import deque

import numpy as np
from scipy import signal

try:
    import joblib
    from neurosense import window_features
except ImportError as e:
    print("[ERROR] import 失敗:", e)
    sys.exit(1)

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(ROOT, "preprocessed_data_neurosense", "va_window_model.joblib")

# ストリーム設定（学習と一致させる）
ORIG_FS = 256                  # Muse2 / TD から来る生サンプリング
FS = 128                       # 学習時の特徴サンプリング
DECIM = ORIG_FS // FS          # 256 -> 128
DEFAULT_WIN_SEC = 16.0         # フォールバック。実際の窓長は モデル(bundle)の win_sec を使う
CH_ORDER = ["TP9", "AF7", "AF8", "TP10"]   # 受信する4chの順番（Mind Monitor /muse/eeg と一致）


class VAPredictor:
    def __init__(self, model_path=DEFAULT_MODEL, pred_hz=4.0, ema=0.3,
                 cal_sec=60.0, clip=True):
        b = joblib.load(model_path)
        self.model_kind = b.get("model_kind", "regression")
        self.mu = np.asarray(b["global_mu"], dtype=np.float64)
        self.sd = np.asarray(b["global_sd"], dtype=np.float64)
        self.global_mu, self.global_sd = self.mu.copy(), self.sd.copy()
        self.label_names = b.get("label_names", ["valence", "arousal"])
        if self.model_kind == "personal_hybrid":
            self.clf = b["clf"]
            self.v_anchors = np.asarray(b["v_anchors"], dtype=np.float64)
            self.temperature = float(b.get("temperature", 1.5))
            self.blend_weight = float(b.get("blend_weight", 0.7))
            self.gbm_global = b.get("gbm_global")
            self.ba_idx = np.asarray(b["ba_idx"], dtype=int)
            self.ba_low = float(b["ba_low"]); self.ba_high = float(b["ba_high"])
            self.svr = self.gbm = None
            kind = f"personal_hybrid(V=soft+BIDS, A=β/α)"
        elif self.model_kind == "soft_emotion":
            self.clf = b["clf"]
            self.anchors = np.asarray(b["anchors"], dtype=np.float64)
            self.temperature = float(b.get("temperature", 1.5))
            self.blend_weight = float(b.get("blend_weight", 0.7))
            self.gbm_global = b.get("gbm_global")
            self.svr = self.gbm = None
            kind = f"personal soft_emotion({len(self.anchors)})"
        else:
            self.svr, self.gbm = b.get("svr"), b["gbm"]
            kind = "LightGBM" if self.svr is None else "SVR+LightGBM"
        # 窓長はモデル(bundle)から取得（学習と一致）
        self.win_sec = float(b.get("win_sec", DEFAULT_WIN_SEC))
        self.win_samples = int(ORIG_FS * self.win_sec)
        print(f"[model] {model_path}")
        print(f"        {kind}  窓数={b.get('n_train_windows')} 被験者={b.get('n_subjects')} "
              f"特徴={len(self.mu)}次元  窓={self.win_sec:.0f}s  作成={b.get('created')}")

        self.buf = deque(maxlen=self.win_samples + ORIG_FS)   #if over max_len => deleted
        self.lock = threading.Lock() #EEGを受け取るスレッドと、推論するスレッドを同時に行う
        self.pred_dt = 1.0 / pred_hz
        self.ema = ema
        self.va = np.array([0.5, 0.5])      # 出力の平滑化状態
        self.clip = clip

        self.cal_sec = cal_sec
        self.calibrating = False
        self.cal_feats = []
        self.cal_start = 0.0
        self.running = False
        self._on_va = None                 

    # input data from 4ch -> use lock to protect data from being used while processing.
    def push_sample(self, ch4):
        with self.lock:
            self.buf.append(np.asarray(ch4, dtype=np.float64))

    #リアルタイム推論用の61次元特徴量を作る
    def _current_feature(self):
        with self.lock:
            if len(self.buf) < self.win_samples:
                return None
            #if enough data -> use the recent win_sec data
            arr = np.stack(list(self.buf)[-self.win_samples:], axis=1)   # (4, win_samples@256)
        #downsample
        dec = signal.decimate(arr, DECIM, ftype="fir", axis=1)      # (4, win_samples/2 @128)
        return window_features(dec).astype(np.float64)

    #feature -> prediction (V/A)
    def _predict(self, feat):
        z = ((feat - self.mu) / self.sd).reshape(1, -1)
        if self.model_kind == "personal_hybrid":
            p = self.clf.predict_proba(z)[0]
            p = np.power(p, self.temperature); p = p / (p.sum() + 1e-12)
            v = float(p @ self.v_anchors)
            if self.gbm_global is not None and self.blend_weight < 1.0:
                vg = float(np.ravel(self.gbm_global.predict(z))[0])
                v = self.blend_weight * v + (1.0 - self.blend_weight) * vg
            ba = float(np.mean(feat[self.ba_idx]))
            a = (ba - self.ba_low) / (self.ba_high - self.ba_low + 1e-12)
            out = np.array([v, a])
        elif self.model_kind == "soft_emotion":
            p = self.clf.predict_proba(z)[0]
            p = np.power(p, self.temperature); p = p / (p.sum() + 1e-12)
            out = p @ self.anchors
            if self.gbm_global is not None and self.blend_weight < 1.0:
                out = self.blend_weight * out + (1.0 - self.blend_weight) * self.gbm_global.predict(z)[0]
        elif self.svr is None:                     # LightGBM単体
            out = self.gbm.predict(z)[0]
        else:                                       # SVR+LightGBM
            out = 0.5 * (self.svr.predict(z) + self.gbm.predict(z))[0]
        if self.clip:
            out = np.clip(out, 0.0, 1.0)
        return out

    #その人ようの特徴量スケールを作る
    def start_calibration(self):
        self.cal_feats = []
        self.cal_start = time.time()
        self.calibrating = True
        print(f"[calib] キャリブ開始（{self.cal_sec:.0f}秒, 安静で装着してください）")

    #calculate mean and std for calibration
    def _update_calibration(self, feat):
        self.cal_feats.append(feat)
        if time.time() - self.cal_start >= self.cal_sec and len(self.cal_feats) >= 10:
            F = np.stack(self.cal_feats)
            self.mu = F.mean(0)
            self.sd = F.std(0) + 1e-8
            self.calibrating = False
            print(f"[calib] 完了: {len(self.cal_feats)}窓で本人の平均/分散に更新")

    
    def loop(self):
        self.running = True
        while self.running:
            t0 = time.time()
            #build feature
            feat = self._current_feature()
            if feat is not None:
                if self.calibrating:
                    self._update_calibration(feat)
                raw = self._predict(feat)
                #EMA smoothing -> reduce sudden change from last prediction
                self.va = self.ema * raw + (1 - self.ema) * self.va
                #send prediction to ToughDesigner
                if self._on_va:
                    self._on_va(float(self.va[0]), float(self.va[1]))
            #wait for next prediction
            dt = self.pred_dt - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)

    def stop(self):
        self.running = False

"""
model_path : 読み込む学習済みモデルのパス
in_ip      : OSCを受信するIP
in_port    : OSCを受信するポート
out_ip     : 予測結果を送る先のIP
out_port   : 予測結果を送る先のポート
eeg_addr   : EEGデータを受け取るOSCアドレス
pred_hz    : 1秒あたりの予測回数
ema        : 予測値の平滑化係数
cal_sec    : キャリブレーション秒数
auto_calib : 起動時に自動キャリブレーションするか
"""
def run_server(model_path, in_ip, in_port, out_ip, out_port,
               eeg_addr, pred_hz, ema, cal_sec, auto_calib):
    from pythonosc import dispatcher as osc_dispatcher
    from pythonosc import osc_server as osc_srv
    from pythonosc.udp_client import SimpleUDPClient

    model = VAPredictor(model_path, pred_hz=pred_hz, ema=ema, cal_sec=cal_sec)
    client = SimpleUDPClient(out_ip, out_port)

    def send_va(v, a):
        client.send_message("/valence", v)
        client.send_message("/arousal", a)
        client.send_message("/va", [v, a])
    model._on_va = send_va

    def handle_eeg(address, *args):
        # 4ch（TP9,AF7,AF8,TP10）が1サンプルぶん届く想定
        if len(args) >= 4:
            model.push_sample(args[:4])

    def handle_calibrate(address, *args):
        model.start_calibration()
    
    #OSCアドレスが来たら、この関数を呼ぶ
    disp = osc_dispatcher.Dispatcher()
    disp.map(eeg_addr, handle_eeg)
    disp.map("/eeg", handle_eeg)           
    disp.map("/calibrate", handle_calibrate)

    server = osc_srv.ThreadingOSCUDPServer((in_ip, in_port), disp)
    th = threading.Thread(target=model.loop, daemon=True)
    th.start()
    if auto_calib:
        model.start_calibration()

    print(f"[osc] 受信 {in_ip}:{in_port}  EEGアドレス '{eeg_addr}' / '/eeg'")
    print(f"[osc] 送信 {out_ip}:{out_port}  /valence /arousal /va  ({pred_hz:.0f}Hz更新)")
    print("[osc] Ctrl-C で停止")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[osc] 停止します")
        model.stop()


def run_selftest(model_path, n_subjects=2, pred_hz=4.0, ema=0.3):
    import glob, json, re
    from neurosense import BIDS_DIR
    pred = VAPredictor(model_path, pred_hz=pred_hz, ema=ema)
    outputs = []
    pred._on_va = lambda v, a: outputs.append((v, a))
    th = threading.Thread(target=pred.loop, daemon=True)
    th.start()

    edfs = sorted(glob.glob(os.path.join(BIDS_DIR, "sub-*", "ses-S001", "eeg", "*_eeg.edf")))[:n_subjects]
    import pyedflib
    for edf in edfs:
        jf = edf.replace("_eeg.edf", "_eeg.json")
        t = json.load(open(jf))["TaskDescription"]
        v_true = (float(re.search(r"AVG_Valence:\s*([\d.]+)", t).group(1)) - 1) / 8
        a_true = (float(re.search(r"AVG_Arousal:\s*([\d.]+)", t).group(1)) - 1) / 8
        r = pyedflib.EdfReader(edf)
        names = r.getSignalLabels()
        sig = {nm: r.readSignal(i) for i, nm in enumerate(names)}
        r.close()
        stream = np.stack([sig[c] for c in CH_ORDER], axis=1)   # (N,4) 256Hz, TP9,AF7,AF8,TP10

        print(f"\n=== {os.path.basename(os.path.dirname(os.path.dirname(edf)))} "
              f"曲 ses-S001  true V={v_true:.3f} A={a_true:.3f} ===")
        outputs.clear()
        # 256Hz をリアルより速く流す（10倍速）。実機は push_sample を 256Hz で呼ぶだけ。
        for k in range(len(stream)):
            pred.push_sample(stream[k])
            if k % 25 == 0:
                time.sleep(0.001)
        time.sleep(0.4)
        if outputs:
            arr = np.array(outputs)
            for idx in np.linspace(0, len(arr) - 1, min(6, len(arr))).astype(int):
                print(f"   t≈{idx*pred.pred_dt:4.1f}s  V_pred={arr[idx,0]:.3f}  A_pred={arr[idx,1]:.3f}")
            print(f"   --> 最終 V={arr[-1,0]:.3f}(true {v_true:.3f})  "
                  f"A={arr[-1,1]:.3f}(true {a_true:.3f})  予測窓数={len(arr)}")
    pred.stop()


def main():
    ap = argparse.ArgumentParser(description="リアルタイム V/A 推論サーバ（OSC）")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--in-ip", default="127.0.0.1")
    ap.add_argument("--in-port", type=int, default=5005)
    ap.add_argument("--out-ip", default="127.0.0.1")
    ap.add_argument("--out-port", type=int, default=5006)
    ap.add_argument("--eeg-addr", default="/muse/eeg")
    ap.add_argument("--pred-hz", type=float, default=4.0, help="V/A更新レート")
    ap.add_argument("--ema", type=float, default=0.3, help="出力平滑化(0=平滑強,1=生)")
    ap.add_argument("--cal-sec", type=float, default=60.0, help="キャリブ秒数")
    ap.add_argument("--no-calib", action="store_true", help="起動時の自動キャリブを無効")
    ap.add_argument("--selftest", action="store_true", help="NeuroSenseで動作確認(OSC不要)")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        print(f"[ERROR] モデルがありません: {args.model}")
        print("        先に  python3 train_window.py  を実行してください。")
        sys.exit(1)

    if args.selftest:
        run_selftest(args.model, pred_hz=args.pred_hz, ema=args.ema)
    else:
        run_server(args.model, args.in_ip, args.in_port, args.out_ip, args.out_port,
                   args.eeg_addr, args.pred_hz, args.ema, args.cal_sec,
                   auto_calib=not args.no_calib)


if __name__ == "__main__":
    main()
