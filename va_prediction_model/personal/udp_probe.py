"""
udp_probe.py 
→MindMonitorから生UDPパケットが届くか確認
"""

import socket
import time
import argparse


def main():
    ap = argparse.ArgumentParser(description="生UDP受信の確認")
    ap.add_argument("--port", type=int, default=5005)
    ap.add_argument("--sec", type=float, default=20)
    args = ap.parse_args()

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(("0.0.0.0", args.port))
    except OSError as e:
        print(f"[ERROR] ポート{args.port}をbindできません: {e}（TD等が使用中なら閉じる）")
        return
    s.settimeout(0.5)
    print(f"[udp] 0.0.0.0:{args.port} を {args.sec:.0f}秒監視。Mind Monitor を送信状態に。")
    n = 0
    srcs = set()
    t0 = time.time()
    while time.time() - t0 < args.sec:
        try:
            data, addr = s.recvfrom(65535)
            n += 1
            srcs.add(addr[0])
            if n <= 3:
                print(f"\n  受信 {len(data)}byte from {addr[0]}:{addr[1]}")
        except socket.timeout:
            pass
        print(f"\r  UDPパケット数={n}  送信元={sorted(srcs)}", end="")
    print()
    if n == 0:
        print("=> 生UDPすら0個。")
        print("   Mind MonitorのTarget IPが今のPCのIPか")
        print("   Mind MonitorのOSC送信がONで、Museが接続済みか")
    else:
        print(f"=> UDPは{n}個届いている（送信元{sorted(srcs)}）。")
        print("   OSCで取れないなら python3 personal/osc_monitor.py で確認。")


if __name__ == "__main__":
    main()
