"""
Mind Monitor 経由のOSC(/muse/eeg)で録音する。各曲の感情に V/A アンカーを付け、
train_personal.py が本人モデルを学習する用の生EEGを保存する。

出力：
personal/recordings/calib_<timestamp>.npz  (感情ごとの生EEG + V/Aラベル)
"""

import os
import sys
import time
import subprocess
import argparse
import threading

import numpy as np

try:
    from pythonosc import dispatcher as osc_dispatcher
    from pythonosc import osc_server as osc_srv
except ImportError:
    print("[ERROR] python-osc が必要:  pip install python-osc")
    sys.exit(1)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # va_prediction_model/
MUSIC_DIR = os.path.join(ROOT, "MUSIC")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings")
CH_ORDER = ["TP9", "AF7", "AF8", "TP10"]
ORIG_FS = 256

# 感情 -> V/A アンカー（0〜1, Russell円環のざっくり座標。4象限をカバー）+ 自己誘発の手がかり
EMOTIONS = [
    {"name": "happiness", "file": "Happy_Commercial-Happiness.mp3", "V": 0.85, "A": 0.55,
     "induce": "幸せだった瞬間（大切な人と笑った等）を思い出し、その温かい喜びを感じ続けてください"},
    {"name": "excitement", "file": "Light-Excitement.mp3", "V": 0.75, "A": 0.90,
     "induce": "ワクワク・興奮した体験（旅行直前・勝利の瞬間等）を思い出し、高揚を体に感じ続けてください"},
    {"name": "sadness", "file": "Magic-Sadness.mp3", "V": 0.20, "A": 0.25,
     "induce": "悲しかった出来事を思い出し、その重く沈む感じに静かに浸り続けてください"},
    {"name": "anger", "file": "RisingDanger-Anger.mp3", "V": 0.20, "A": 0.80,
     "induce": "強く怒った/理不尽だった出来事を思い出し、その怒りを感じ続けてください"},
]

def song_duration(path):
    try:
        out = subprocess.check_output(["afinfo", path], stderr=subprocess.DEVNULL).decode()
        for line in out.splitlines():
            if "estimated duration" in line.lower():
                return float(line.split(":")[1].strip().split()[0])
    except Exception:
        pass
    return None


class EEGCollector:
    def __init__(self):
        self.buf = []
        self.recording = False
        self.lock = threading.Lock()

    def handle(self, address, *args):
        if self.recording and len(args) >= 4:
            with self.lock:
                self.buf.append([float(args[0]), float(args[1]),
                                 float(args[2]), float(args[3])])

    def start(self):
        with self.lock:
            self.buf = []
        self.recording = True

    def stop(self):
        self.recording = False
        with self.lock:
            if not self.buf:
                return np.empty((4, 0), dtype=np.float32)     # 受信0でも落ちないように
            return np.asarray(self.buf, dtype=np.float32).T   # (4, N)


def record_silent(coll, emo, rec_sec):
    total = rec_sec if rec_sec > 0 else 60.0
    print(f"\n=== {emo['name'].upper()}  (V={emo['V']} A={emo['A']}) ===")
    print(f"   ◆ {emo['induce']}")
    print(f"   準備ができたら Enter →{total:.0f}秒録音します（その間ずっとその感情を感じ続けて）")
    input("   Enter で開始… ")
    for c in (3, 2, 1):
        print(f"   {c}…"); time.sleep(1)
    coll.start()
    t0 = time.time()
    while time.time() - t0 < total:
        time.sleep(0.2)
        print(f"\r   録音中… {time.time()-t0:4.0f}/{total:.0f}s  受信={len(coll.buf)}", end="")
    eeg = coll.stop()
    print(f"\n   完了: EEG {eeg.shape} (≈{eeg.shape[1]/ORIG_FS:.0f}s)")
    if eeg.shape[1] < ORIG_FS * 15:
        print("   [警告] サンプルが少ない（OSC接続を確認）")
    return eeg


def record_one(coll, emo, rec_sec, warmup):
    path = os.path.join(MUSIC_DIR, emo["file"])
    if not os.path.exists(path):
        print(f"   [skip] 曲が無い: {path}"); return None
    dur = song_duration(path)
    total = rec_sec if rec_sec > 0 else (min(dur, 180.0) if dur else 120.0)

    print(f"\n=== {emo['name'].upper()}  (V={emo['V']} A={emo['A']}) ===")
    print(f"   曲: {emo['file']}  録音 {total:.0f}秒（最初の{warmup}秒は感情の立ち上げで除外）")
    for c in (3, 2, 1):
        print(f"   開始まで {c}..."); time.sleep(1)

    player = subprocess.Popen(["afplay", path])
    time.sleep(warmup)                       
    coll.start()
    t0 = time.time()
    while time.time() - t0 < total - warmup:
        time.sleep(0.2)
        n = len(coll.buf)
        print(f"\r   録音中… {time.time()-t0:4.0f}s  受信サンプル={n}", end="")
    eeg = coll.stop()
    print(f"\n   完了: EEG {eeg.shape} (≈{eeg.shape[1]/ORIG_FS:.0f}s @ {ORIG_FS}Hz)")
    try:
        player.terminate()
    except Exception:
        pass
    if eeg.shape[1] < ORIG_FS * 20:
        print("   [警告] サンプルが少ない。Mind Monitor送信先/ポート/TD競合を確認")
    return eeg


def main():
    ap = argparse.ArgumentParser(description="感情誘発EEGの録音（Mind Monitor OSC）")
    ap.add_argument("--in-ip", default="0.0.0.0")
    ap.add_argument("--in-port", type=int, default=5005)
    ap.add_argument("--eeg-addr", default="/muse/eeg")
    ap.add_argument("--sec", type=float, default=0,
                    help="各感情の録音秒（音楽:0=曲フル/最大180、無音:0=60秒）")
    ap.add_argument("--warmup", type=float, default=8.0, help="音楽モードで感情立ち上げに使う秒")
    ap.add_argument("--silent", action="store_true",
                    help="無音・自己誘発モード（音楽を流さず、感情を思い出して感じている間を録音）")
    args = ap.parse_args()

    coll = EEGCollector()
    disp = osc_dispatcher.Dispatcher()
    disp.map(args.eeg_addr, coll.handle)
    disp.map("/eeg", coll.handle)
    server = osc_srv.ThreadingOSCUDPServer((args.in_ip, args.in_port), disp)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    mode = "無音・自己誘発" if args.silent else "音楽誘発"
    print(f"[osc] 受信 {args.in_ip}:{args.in_port}  '{args.eeg_addr}'  モード={mode}")
    print("Mind Monitor を起動し、Muse装着・接続してください。")
    print("★ TouchDesigner が同ポートを使っているとここに届きません（録音中はTD側OSCを止める）")
    if args.silent:
        print("※ 無音モード: 音楽は流しません。各感情を“思い出して感じる”間を録音します。")
    input("準備ができたら Enter を押すと録音を開始します… ")

    # --- 事前チェック: OSCが本当に届いているか（3秒）。届かないなら即中止 ---
    print("OSC受信テスト中（3秒）…")
    coll.start(); time.sleep(3); test = coll.stop()
    if test.shape[1] == 0:
        print("[中止] OSCが1つも届いていません（受信0）。録音前に直しましょう：")
        print("  ① TouchDesigner を完全に閉じる（ポート5005の取り合い）")
        print(f"  ② Mind Monitor: Target IP=このPC / Port={args.in_port} / OSC Stream ON / Raw EEG有効")
        print("  ③ 何が届いているか確認:  python3 personal/osc_monitor.py")
        sys.exit(1)
    print(f"  OK: 3秒で {test.shape[1]} サンプル受信。録音を開始します。")

    recs = {}
    for emo in EMOTIONS:
        if args.silent:
            eeg = record_silent(coll, emo, args.sec)
        else:
            eeg = record_one(coll, emo, args.sec, args.warmup)
        if eeg is not None and eeg.shape[1] > 0:
            recs[emo["name"]] = eeg
        time.sleep(1)

    if not recs:
        print("\n[ERROR] 録音データが空です。OSC接続を確認してやり直してください。")
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = os.path.join(OUT_DIR, f"calib_{ts}.npz")
    save = {"fs": ORIG_FS, "ch_order": np.array(CH_ORDER),
            "emotions": np.array([e["name"] for e in EMOTIONS]),
            "anchors": np.array([[e["V"], e["A"]] for e in EMOTIONS], dtype=np.float32)}
    for name, eeg in recs.items():
        save[f"eeg_{name}"] = eeg
    np.savez_compressed(out, **save)
    print(f"\n保存: {out}")
    print(f"  録音できた感情: {list(recs.keys())}")
    print("次: python3 personal/train_personal.py  で本人モデルを学習")

if __name__ == "__main__":
    main()
