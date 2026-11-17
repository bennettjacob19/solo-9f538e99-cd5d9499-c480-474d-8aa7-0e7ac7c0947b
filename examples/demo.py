#!/usr/bin/env python3
"""End-to-end demo of the hot-swap service, runnable against a live API.

Usage::

    python examples/demo.py [BASE_URL]

Covers the scenarios reviewers care about:

1. normal inference, then a successful hot swap;
2. a failed candidate load (old version keeps serving);
3. slow inference with a client timeout, followed by a swap and a third
   upload that is refused until the slow work finishes and the slot is freed;
4. device calibration: trial + publish, calibrated inference, a failed trial
   that keeps the old revision, and recalibration after a model swap.
"""

from __future__ import annotations

import json
import sys
import time

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
LABELS = ["pass", "scratch", "dent"]
W1 = [[1.2, -0.4, 0.0, 0.3], [-0.2, 0.9, 0.5, -0.1], [0.1, -0.8, 1.1, 0.2]]
W2 = [[0.3, 1.1, -0.2, 0.1], [1.3, -0.3, 0.1, -0.2], [-0.4, 0.2, 0.9, 0.7]]
FEATURES = [0.7, 0.2, 0.1, 0.5]
DEVICE = "line-3-station-a"
# Raw channel order as read by the sensor after replacement, and the affine
# correction that maps it back to what the model version expects.
PERMUTATION = [1, 0, 2, 3]
GAIN = [2.0, 1.0, 1.0, 0.5]
OFFSET = [-1.0, 0.0, 0.0, 0.1]

client = httpx.Client(base_url=BASE, timeout=60.0)


def raw(model_features):
    """Inverse of the calibration above: model-space -> raw device channels.

    Forward: calibrated[i] = raw[PERMUTATION[i]] * GAIN[i] + OFFSET[i].
    """
    return [
        model_features[1],
        (model_features[0] + 1.0) / 2.0,
        model_features[2],
        (model_features[3] - 0.1) / 0.5,
    ]


def show(label, resp):
    try:
        body = resp.json()
    except Exception:
        body = resp.text
    print(f"  {label:<42} {resp.status_code}  {json.dumps(body)[:160]}")
    return resp


def slots():
    st = client.get("/admin/status").json()
    layout = [
        (s["slot_id"], s["version"], s["process_state"], s["in_flight"])
        for s in st["slots"]
    ]
    print(f"  status: active={st['active_version']} slots={layout}")
    if st["calibrated_devices"]:
        print(f"  calibrated devices: {st['device_calibration_revisions']}")
    return st


def artifact(version, weights, **extra):
    return {"version": version, "weights": weights, "bias": [0.0, 0.0, 0.0], "labels": LABELS, **extra}


def active_version():
    return client.get("/admin/status").json()["active_version"]


def expected(model_features):
    body = client.post("/infer", json={"features": model_features}).json()
    return body["label"]


def calibration_payload(**extra):
    samples = [
        {"features": raw(f), "expected_label": expected(f)}
        for f in ([0.7, 0.2, 0.1, 0.5], [0.1, 0.8, 0.2, 0.4])
    ]
    return {
        "device_id": DEVICE,
        "model_version": active_version(),
        "channel_permutation": PERMUTATION,
        "gain": GAIN,
        "offset": OFFSET,
        "samples": samples,
        **extra,
    }


def main() -> int:
    print("\n== 0. baseline load ==")
    show("upload v1", client.post("/admin/models", json=artifact("v1", W1)))
    show("infer", client.post("/infer", json={"features": FEATURES}))
    slots()

    print("\n== 1. successful hot swap ==")
    show("upload v2", client.post("/admin/models", json=artifact("v2", W2)))
    show("infer after swap", client.post("/infer", json={"features": FEATURES}))
    slots()

    print("\n== 2. failed candidate load leaves v2 active ==")
    show(
        "upload bad-load",
        client.post(
            "/admin/models", json=artifact("bad-load", W1, simulate_load_failure=True)
        ),
    )
    show("infer (still v2)", client.post("/infer", json={"features": FEATURES}))
    show(
        "upload invalid artifact",
        client.post("/admin/models", json=artifact("broken", [[1.0, 2.0]])),
    )
    slots()

    print("\n== 3. device calibration: uncalibrated device is refused ==")
    show("device infer (no calibration)", client.post("/infer", json={"features": raw(FEATURES), "device_id": DEVICE}))
    show("upload v3", client.post("/admin/models", json=artifact("v3", W2)))
    show("calibrate device on v3", client.post("/admin/calibrations", json=calibration_payload()))
    show(
        "calibrated infer",
        client.post("/infer", json={"features": raw(FEATURES), "device_id": DEVICE}),
    )
    slots()

    print("\n== 4. a failed trial keeps the published revision ==")
    bad = calibration_payload(
        samples=[
            {"features": raw(FEATURES), "expected_label": expected([0.1, 0.8, 0.2, 0.4])},
            {"features": raw(FEATURES), "expected_label": expected(FEATURES)},
        ]
    )
    show("calibrate with a wrong label", client.post("/admin/calibrations", json=bad))
    show(
        "calibrated infer still works",
        client.post("/infer", json={"features": raw(FEATURES), "device_id": DEVICE}),
    )

    print("\n== 5. model swap invalidates the calibration until re-trialled ==")
    show("upload v4", client.post("/admin/models", json=artifact("v4", W1)))
    show(
        "device infer on uncalibrated v4",
        client.post("/infer", json={"features": raw(FEATURES), "device_id": DEVICE}),
    )
    show("recalibrate on v4", client.post("/admin/calibrations", json=calibration_payload()))
    show(
        "calibrated infer on v4",
        client.post("/infer", json={"features": raw(FEATURES), "device_id": DEVICE}),
    )
    slots()

    print("\n== 6. slow inference + client timeout ==")
    time.sleep(1.0)  # let earlier drains finish and free a slot
    show(
        "slow infer, timeout_ms=150",
        client.post("/infer", json={"features": FEATURES, "slow_ms": 6000, "timeout_ms": 150}),
    )
    st = slots()
    print(f"  in-flight work still running in model process: {st['total_in_flight']}")

    print("\n== 7. swap while the slow call is still running ==")
    show("upload v5", client.post("/admin/models", json=artifact("v5", W1)))
    slots()
    show("upload v6 (third process)", client.post("/admin/models", json=artifact("v6", W2)))
    show("infer (v5 already serving)", client.post("/infer", json={"features": FEATURES}))

    print("\n== 8. wait for the slow work to finish; slot is released ==")
    deadline = time.time() + 15
    while time.time() < deadline:
        st = slots()
        if st["slots_in_use"] == 1:
            break
        time.sleep(0.5)
    show("upload v6 now", client.post("/admin/models", json=artifact("v6", W2)))
    slots()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
