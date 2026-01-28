#!/usr/bin/env python3
"""
predictive_metronome.py
"""
import csv
import io
import time
import threading
from flask import Flask, send_from_directory, jsonify, send_file
import requests
from collections import deque

# ---------------------------
# Configuration / constants
# ---------------------------

_all_step_times = []
_all_beat_times = []

# PhyPhox / network
PHY_URL = "http://192.168.86.41"
POLL_ENDPOINT = f"{PHY_URL}/get?accY&acc_time"

# Server polling (your computer)
POLL_S = 0.01              # Hz server polling

# Prediction / metronome
LEAD_SEC = -0.0001        # play beep this many seconds before predicted impact
MAX_DELAY = 2.0           # clamp: do not hand browser delays > this

# Step detection timing
DETECT_REFRACTORY = 0.60  # minimum seconds between distinct detections (avoid double triggers)

# Buffers / smoothing sizes
ACC_BUF_LEN = 8           # window for local extremum check
DERIV_BUF_LEN = 5         # smoothing window for derivative
STEPS_MAXLEN = 6          # store last N step timestamps for interval estimation

# Derivative thresholding (per-second units)
# Because derivative = (ay - prev_ay) / dt, with dt ~0.01, a change of 6-15 units yields 600-1500
DERIV_MAG_THRESHOLD = 20.0  # start here; tune up/down after inspecting logs
MIN_VALLEY_MAG = -10.0       # optional sanity: trough magnitude expected for many phone setups

INTERVALS_MAXLEN = 8
GAP_MULTIPLE = 2.0
MAX_INFERRED_STEPS = 2

# ---------------------------
# Flask app + shared state
# ---------------------------
app = Flask(__name__)

_state_lock = threading.Lock()
_next_delay = None               # None or float (seconds from "now")
_steps = deque(maxlen=STEPS_MAXLEN)
_intervals_real = deque(maxlen=INTERVALS_MAXLEN)

@app.route("/")
def index():
    return send_from_directory(".", "sound.html")

@app.route("/next_step", methods=["GET"])
def next_step():
    """Return next delay (seconds from now) or null. Clears stored delay once handed out."""
    global _next_delay
    with _state_lock:
        d = _next_delay
        _next_delay = None
    return jsonify({"delay": None if d is None else float(d)})

@app.route("/download_csv")
def download_csv():
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["type", "timestamp"])

    # steps
    for t in _all_step_times:
        writer.writerow(["step", t])

    # beats
    for t in _all_beat_times:
        writer.writerow(["beat", t])

    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype="text/csv",
        as_attachment=True,
        download_name="session_data.csv"
    )

# ---------------------------
# Detection / prediction loop
# ---------------------------

def robust_period(intervals):
    if len(intervals) < 3:
       return None
    s = sorted(intervals)
    return s[len(s)//2]

def step_detection_loop():
    global _next_delay, _steps

    print("Backend: polling PhyPhox at", POLL_ENDPOINT)
    # local buffers / state for detection
    acc_buf = deque(maxlen=ACC_BUF_LEN)    # recent raw accY
    deriv_buf = deque(maxlen=DERIV_BUF_LEN)  # recent derivatives (per-second)
    prev_ay = None
    prev_ts_phy = None
    last_step_time_phy = None  # local reference; also appended to _steps when detected

    session = requests.Session()

    while True:
        try:
            r = session.get(POLL_ENDPOINT, timeout=3.0)
            data = r.json()

            # phphox buffer format: data["buffer"]["accY"]["buffer"], data["buffer"]["acc_time"]["buffer"]
            buf = data.get("buffer", {})
            acc_buffer = buf.get("accY", {}).get("buffer", [])
            t_buffer = buf.get("acc_time", {}).get("buffer", [])

            if not acc_buffer or not t_buffer:
                # no valid data in response
                time.sleep(POLL_S)
                continue

            ay = acc_buffer[-1]
            ts_phy = t_buffer[-1]   # monotonic phy timestamps (seconds since measurement start on phone)

            # initialize previous sample on first run
            if prev_ts_phy is None:
                prev_ts_phy = ts_phy
                prev_ay = ay
                acc_buf.append(ay)
                # nothing to detect on the very first sample
                time.sleep(POLL_S)
                continue

            # compute dt from phy timestamps (use true phone dt)
            dt = ts_phy - prev_ts_phy
            if dt <= 0:
                # skip faulty/repeated timestamp sample but update prevs so we don't lock
                prev_ts_phy = ts_phy
                prev_ay = ay
                time.sleep(POLL_S)
                continue

            # derivative normalized to per-second units
            deriv = (ay - prev_ay) / dt
            deriv_buf.append(deriv)
            acc_buf.append(ay)

            # smoothed derivative
            avg_deriv = sum(deriv_buf) / len(deriv_buf)

            # spike condition: large magnitude derivative (either sign)
            spike = abs(avg_deriv) >= DERIV_MAG_THRESHOLD

            # local extremum check: ensure we are near a peak/trough to reduce false positives
            local_extremum = False
            a2 = None

            if len(acc_buf) >= 3:
                a1 = acc_buf[-3]
                a2 = acc_buf[-2]
                a3 = acc_buf[-1]
                # check if middle sample was a valley or peak
                if (a2 < a1 and a2 < a3) or (a2 > a1 and a2 > a3):
                    local_extremum = True

            # optional sanity: if avg_deriv negative we expect a trough sufficiently low
            magnitude_ok = True
            if a2 is not None and avg_deriv < 0:
                # if your data uses gravity ~ -9.8 baseline, tune MIN_VALLEY_MAG accordingly
                if a2 > MIN_VALLEY_MAG:
                    magnitude_ok = False

            detected = False
            # detection: spike + local extremum + refractory
            if spike and local_extremum and magnitude_ok:
                # timestamp to use for the detected event: use the middle sample a2's timestamp approximation
                # (we only have phy timestamps for the latest sample); use current ts_phy as the event time
                # since PhyPhox timestamps are monotonic and dense.
                if last_step_time_phy is None or (ts_phy - last_step_time_phy) >= DETECT_REFRACTORY:
                    detected = True
                    last_step_time_phy = ts_phy

            if detected:
                _steps.append(ts_phy)
                _all_step_times.append(ts_phy)
                # also store last_step_time_phy (keeps local refractory)
                print(f"Detected step at phy={ts_phy:.6f} ay={ay:.3f} avg_deriv={avg_deriv:.1f}")

                # prediction using last interval
                if len(_steps) >= 2:
                   raw_interval = _steps[-1] - _steps[-2]
                   _intervals_real.append(raw_interval)

                   T_hat = robust_period(_intervals_real)

                   inferred_steps = []

                   if T_hat is not None and raw_interval > GAP_MULTIPLE * T_hat:
                       # infer missed steps
                       m = round(raw_interval / T_hat)
                       m = max(1, min(m, MAX_INFERRED_STEPS + 1))

                       if m >= 2:
                           print(f"Missed-step inferred: raw={raw_interval:.3f}s, T̂={T_hat:.3f}s, m={m}")

                           for j in range(1, m):
                               t_fake = _steps[-2] + j * (raw_interval / m)
                               inferred_steps.append(t_fake)
                           for t_inf in inferred_steps:
                               _all_step_times.append(t_inf)

                   # choose effective last interval for prediction
                   if inferred_steps:
                       effective_interval = raw_interval / (len(inferred_steps) + 1)
                       last_step_time = inferred_steps[-1]
                   else:
                       effective_interval = raw_interval
                       last_step_time = _steps[-1]

                   predicted_next_phy = last_step_time + effective_interval

                   interval = _steps[-1] - _steps[-2]
                   predicted_next_phy = _steps[-1] + interval

                   # delay relative to now (server time). We use phy timestamps differences as proxy.
                   delay = predicted_next_phy - ts_phy
                   # subtract lead so beep happens slightly before predicted impact
                   delay = delay - LEAD_SEC

                   if delay < 0:
                       # already passed
                       print("Predicted delay already past (ignored).")
                   elif delay > MAX_DELAY:
                       print("Predicted delay too large (ignored):", f"{delay:.3f}")
                   else:
                       with _state_lock:
                           _next_delay = delay
                       print("Will send delay (s):", f"{delay:.3f}")
                       _all_beat_times.append(predicted_next_phy)
            # update prevs for next sample
            prev_ay = ay
            prev_ts_phy = ts_phy

        except Exception as e:
            print("ERR polling:", e)

        # respect server-side poll interval (your computer load)
        time.sleep(POLL_S)


# ---------------------------
# Start background thread + server
# ---------------------------
if __name__ == "__main__":
    t = threading.Thread(target=step_detection_loop, daemon=True)
    t.start()
    # Flask default reloader can spawn processes; disable reloader here.
    app.run(host="0.0.0.0", port=5000, threaded=True, use_reloader=False)

