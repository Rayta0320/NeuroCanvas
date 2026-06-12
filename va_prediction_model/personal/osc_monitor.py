# Small diagnostic tool to check whether OSC messages are reaching this machine.
import argparse
import time
import threading
import collections

from pythonosc import dispatcher as osc_dispatcher
from pythonosc import osc_server as osc_srv


def main():
    ap = argparse.ArgumentParser(description="Monitor incoming OSC messages.")
    ap.add_argument("--ip", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5005)
    ap.add_argument("--sec", type=float, default=15)
    args = ap.parse_args()

    counts = collections.Counter()
    sample = {}

    def handle(addr, *a):
        counts[addr] += 1
        if addr not in sample:
            sample[addr] = [
                round(float(x), 1) if isinstance(x, (int, float)) else x
                for x in a[:6]
            ]

    disp = osc_dispatcher.Dispatcher()
    disp.set_default_handler(handle)  # Catch every OSC address.

    server = osc_srv.ThreadingOSCUDPServer((args.ip, args.port), disp)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    print(f"[monitor] Listening on {args.ip}:{args.port} for {args.sec:.0f} seconds.")
    print("Turn on OSC output in Mind Monitor while your Muse is connected.")
    print()

    t0 = time.time()
    try:
        while time.time() - t0 < args.sec:
            time.sleep(1)
            elapsed = int(time.time() - t0)
            print(
                f"\r  {elapsed:2d}s elapsed | "
                f"{len(counts)} address types | "
                f"{sum(counts.values())} total messages",
                end="",
                flush=True,
            )
    finally:
        server.shutdown()
        server.server_close()

    print("\n\n--- OSC addresses received, sorted by activity ---")

    if not counts:
        print("No OSC messages arrived.")
        print()
        print("Check these first:")
        print("  1. Close TouchDesigner so the port is free.")
        print("  2. Confirm the Target IP and Port in Mind Monitor.")
        print("  3. Make sure the phone and this computer are on the same Wi-Fi.")
        print("  4. Turn on Raw EEG and OSC streaming in Mind Monitor.")
        return

    for addr, c in counts.most_common(25):
        print(
            f"  {c:6d} messages  {addr:28s}  "
            f"{len(sample[addr])} args  example: {sample[addr]}"
        )

    if "/muse/eeg" in counts or "/eeg" in counts:
        print()
        print("Looks good: EEG data is arriving.")
        print("record_calibration.py should work with the current settings.")
    else:
        print()
        print("I did not see /muse/eeg or /eeg.")
        print("Look above for an address with 4 numeric EEG-like values.")
        print("Then pass that address explicitly, for example:")
        print("  python record_calibration.py --eeg-addr <address>")


if __name__ == "__main__":
    main()