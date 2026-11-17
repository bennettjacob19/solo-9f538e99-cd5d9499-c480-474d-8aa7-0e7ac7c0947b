"""Integration tests against a real API process and real worker processes.

These exercise the guarantees that only show up with real OS processes:
version pinning, serving during candidate load, load-failure fallback, the
two-process cap, and the fact that a client timeout does *not* free the slot
until the worker has actually finished.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class ApiServer:
    def __init__(self, data_dir: Path, extra_env: dict | None = None) -> None:
        self.data_dir = data_dir
        self.port = _free_port()
        self.extra_env = extra_env or {}
        self.proc: subprocess.Popen | None = None
        self.log = open(data_dir / "api.log", "w", encoding="utf-8")

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self) -> "ApiServer":
        env = {
            **os.environ,
            **self.extra_env,
            "DATA_DIR": str(self.data_dir),
            "LOG_LEVEL": "WARNING",
        }
        self.proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
                "--log-level",
                "warning",
            ],
            cwd=str(ROOT),
            env=env,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.time() + 30
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError("API exited during startup:\n" + self._tail())
            try:
                if httpx.get(f"{self.base}/healthz", timeout=1.0).status_code == 200:
                    return self
            except httpx.HTTPError:
                pass
            time.sleep(0.05)
        raise TimeoutError("API did not start:\n" + self._tail())

    def __exit__(self, *exc) -> None:
        if self.proc is not None and self.proc.poll() is None:
            os.killpg(os.getpgid(self.proc.pid), 15)
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self.proc.pid), 9)
                self.proc.wait(10)
        # Any model workers belong to their own process groups; the API stops
        # them on shutdown, but reap stragglers that could poison later tests.
        subprocess.run(
            ["pkill", "-f", f"{self.data_dir}/workers"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.log.close()

    def _tail(self) -> str:
        self.log.flush()
        return self.log.name


@pytest.fixture(scope="module")
def api(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("qcdata")
    with ApiServer(data_dir) as server:
        yield server
    shutil.rmtree(data_dir, ignore_errors=True)


def artifact(version: str, weights, bias, labels, **extra):
    return {"version": version, "weights": weights, "bias": bias, "labels": labels, **extra}


W_A = [[1.0, 0.0], [0.0, 1.0]]
W_B = [[0.0, 1.0], [1.0, 0.0]]
LABELS = ["defect-A", "defect-B"]


def upload(api: ApiServer, payload, timeout=40.0):
    return httpx.post(f"{api.base}/admin/models", json=payload, timeout=timeout)


def infer(api: ApiServer, features, timeout_ms=None, slow_ms=0, timeout=40.0):
    body = {"features": features, "slow_ms": slow_ms}
    if timeout_ms is not None:
        body["timeout_ms"] = timeout_ms
    return httpx.post(f"{api.base}/infer", json=body, timeout=timeout)


def status(api: ApiServer):
    return httpx.get(f"{api.base}/admin/status", timeout=5.0).json()


def wait_for(predicate, timeout=10.0, interval=0.05):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    raise AssertionError("condition not met in time")


def slot_states(api: ApiServer):
    return [(s["slot_id"], s["version"], s["process_state"]) for s in status(api)["slots"]]


def test_01_initial_load_and_inference(api):
    resp = upload(api, artifact("v1", W_A, [0.0, 0.0], LABELS))
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["previous_version"] is None
    assert body["state"] == "active"

    result = infer(api, [0.9, 0.1])
    assert result.status_code == 200, result.text
    payload = result.json()
    assert payload["version"] == "v1"
    assert payload["label"] == "defect-A"
    assert payload["score"] == pytest.approx(0.9)

    st = status(api)
    assert st["active_version"] == "v1"
    assert st["slots_in_use"] == 1
    assert st["total_in_flight"] == 0
    assert st["total_completed"] >= 1


def test_02_feature_width_mismatch(api):
    resp = infer(api, [1.0, 2.0, 3.0])
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"] == "feature_dimension_mismatch"
    assert status(api)["total_in_flight"] == 0


def test_03_load_failure_keeps_old_version(api):
    before = status(api)["slots_in_use"]
    resp = upload(
        api,
        artifact("bad-load", W_B, [0.0, 0.0], LABELS, simulate_load_failure=True),
    )
    assert resp.status_code == 502, resp.text
    detail = resp.json()["detail"]
    assert detail["error"] == "candidate_load_failed"
    assert detail["fallback_version"] == "v1"

    st = status(api)
    assert st["active_version"] == "v1"
    assert st["slots_in_use"] == before  # candidate slot released
    failed = next(v for v in st["versions"] if v["version"] == "bad-load")
    assert failed["state"] == "failed"
    assert "simulated" in failed["error"]
    # Old version still answers with the old weights.
    assert infer(api, [0.9, 0.1]).json()["label"] == "defect-A"


def test_04_invalid_artifact_rejected(api):
    resp = upload(api, artifact("broken", [[1.0, 2.0]], [0.0, 0.0], ["a"]))
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"] == "invalid_artifact"


def test_05_old_version_serves_during_candidate_load(api):
    observations = []
    resp = {}

    def do_upload():
        resp["r"] = upload(
            api, artifact("v2", W_B, [0.0, 0.0], LABELS, slow_load_ms=1200)
        )

    import threading

    t = threading.Thread(target=do_upload)
    t.start()
    deadline = time.time() + 1.0
    while time.time() < deadline:
        r = infer(api, [0.9, 0.1])
        if r.status_code == 200:
            observations.append(r.json()["version"])
        time.sleep(0.1)
    t.join()
    assert resp["r"].status_code == 201, resp["r"].text
    assert observations, "old version did not answer during candidate load"
    assert set(observations) <= {"v1", "v2"}
    assert "v1" in observations

    st = status(api)
    assert st["active_version"] == "v2"
    v1 = next(v for v in st["versions"] if v["version"] == "v1")
    assert v1["state"] == "retired"


def test_06_timeout_does_not_release_slot(api):
    # Let the drain of v1 settle first so both slots are known.
    wait_for(lambda: status(api)["slots_in_use"] == 1)

    # Long enough that the upload's bounded drain-wait (3s) expires.
    resp = infer(api, [0.9, 0.1], timeout_ms=150, slow_ms=6000)
    assert resp.status_code == 504, resp.text
    detail = resp.json()["detail"]
    assert detail["error"] == "inference_timeout"
    assert detail["version"] == "v2"

    # Immediately after the client gave up the work is still running.
    st = status(api)
    assert st["total_in_flight"] == 1
    assert st["active_version"] == "v2"

    # Swap commits while the old request is still computing.
    resp2 = upload(api, artifact("v3", W_A, [0.0, 0.0], LABELS))
    assert resp2.status_code == 201, resp2.text
    st = status(api)
    assert st["active_version"] == "v3"
    states = dict((s["version"], s["process_state"]) for s in st["slots"] if s["version"])
    assert states.get("v2") == "draining"
    assert st["slots_in_use"] == 2

    # Third load must be refused while the drain is still held by real work.
    resp3 = upload(api, artifact("v4", W_B, [0.0, 0.0], LABELS))
    assert resp3.status_code == 429, resp3.text
    assert resp3.json()["detail"]["error"] == "capacity_exhausted"
    assert status(api)["active_version"] == "v3"  # never stopped the current version

    # New traffic already goes to the new version.
    assert infer(api, [0.9, 0.1]).json()["version"] == "v3"

    # Once the slow work actually finishes, the drained slot is released.
    wait_for(lambda: status(api)["slots_in_use"] == 1, timeout=15)
    st = status(api)
    v2 = next(v for v in st["versions"] if v["version"] == "v2")
    assert v2["state"] == "retired"
    assert v2["process_state"] == "idle"
    assert st["total_in_flight"] == 0


def test_07_third_version_loads_after_release(api):
    resp = upload(api, artifact("v4", W_B, [0.0, 0.0], LABELS))
    assert resp.status_code == 201, resp.text
    st = status(api)
    assert st["active_version"] == "v4"
    in_use = [s for s in st["slots"] if s["process_state"] != "idle"]
    assert len(in_use) == 2  # v4 active + v3 draining
    assert infer(api, [0.9, 0.1]).json()["version"] == "v4"


def test_08_concurrent_swaps_one_winner(api):
    wait_for(lambda: status(api)["slots_in_use"] == 1, timeout=10)
    import threading

    results = []

    def go(version, weights, slow):
        results.append(
            (
                version,
                upload(
                    api,
                    artifact(version, weights, [0.0, 0.0], LABELS, slow_load_ms=slow),
                ).status_code,
            )
        )

    t1 = threading.Thread(target=go, args=("c1", W_A, 1200))
    t2 = threading.Thread(target=go, args=("c2", W_B, 0))
    t1.start()
    time.sleep(0.25)
    t2.start()
    t1.join()
    t2.join()
    winners = [v for v, code in results if code == 201]
    losers = [v for v, code in results if code == 409]
    assert len(winners) == 1, results
    assert len(losers) == 1, results
    assert status(api)["active_version"] == winners[0]


def test_08b_duplicate_id_concurrent_uploads(api):
    """Two simultaneous uploads of the same id: exactly one wins, one 409."""
    wait_for(lambda: status(api)["slots_in_use"] == 1, timeout=10)
    import threading

    codes = []

    def go():
        codes.append(upload(api, artifact("dup", W_A, [0.0, 0.0], LABELS)).status_code)

    threads = [threading.Thread(target=go) for _ in range(2)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sorted(codes) == [201, 409], codes
    assert status(api)["active_version"] == "dup"
    # The loser must not leave a stale swap claim behind: another swap works.
    resp = upload(api, artifact("dup-next", W_B, [0.0, 0.0], LABELS))
    assert resp.status_code == 201, resp.text


def test_09_status_reports_processes_and_counts(api):
    st = status(api)
    assert st["capacity"] == 2
    assert 1 <= st["slots_in_use"] <= 2
    for slot in st["slots"]:
        assert slot["process_state"] in {"loading", "active", "draining", "idle"}
        if slot["version"] is not None:
            assert slot["alive"] is True
            assert slot["pid"] is not None
    active_slots = [s for s in st["slots"] if s["process_state"] == "active"]
    assert len(active_slots) == 1
    assert active_slots[0]["version"] == st["active_version"]


def test_10_duplicate_version_is_conflict(api):
    active = status(api)["active_version"]
    resp = upload(api, artifact(active, W_A, [0.0, 0.0], LABELS))
    assert resp.status_code == 409
    assert resp.json()["detail"]["error"] == "version_exists"


def test_11_stale_expected_version(api):
    resp = upload(
        api,
        artifact(
            f"stale-{uuid.uuid4().hex[:6]}",
            W_A,
            [0.0, 0.0],
            LABELS,
            expected_active_version="does-not-exist",
        ),
    )
    assert resp.status_code == 409
    assert resp.json()["detail"]["error"] == "swap_conflict"


def test_12_no_model_after_shutdown(api):
    # Sanity: the service still answers health and exposes /docs.
    assert httpx.get(f"{api.base}/healthz", timeout=5).status_code == 200
    assert httpx.get(f"{api.base}/docs", timeout=5).status_code == 200
    assert httpx.get(f"{api.base}/openapi.json", timeout=5).status_code == 200


# ---------------------------------------------------------------------------
# Device calibration
# ---------------------------------------------------------------------------

# A physically plausible station: raw channel 1 is a mirrored sensor with a
# different scale, and raw channel 0 carries an offset.  The model wants
# [ch1 * 2 - 5, ch0] (channel 0/1 swapped, gain, offset).
CAL_DEVICE = "station-7"
CAL_SWAP = [1, 0]
CAL_GAIN = [2.0, 1.0]
CAL_OFFSET = [-5.0, 0.0]


def raw_of_model_features(a: float, b: float) -> list[float]:
    """Inverse of the calibration above: model-space (a, b) -> raw channels."""
    return [b, (a + 5.0) / 2.0]


def label_for(api: ApiServer, a: float, b: float) -> str:
    """Expected label for a model-space point, from the currently active model.

    Deriving it from the untagged path keeps the calibration tests independent
    of which weights the earlier swap tests left active.
    """
    resp = infer(api, [a, b])
    assert resp.status_code == 200, resp.text
    return resp.json()["label"]


def default_samples(api: ApiServer) -> list[dict]:
    return [
        {
            "features": raw_of_model_features(a, b),
            "expected_label": label_for(api, a, b),
        }
        for a, b in ((0.9, 0.1), (0.1, 0.9))
    ]


def calibrate(api: ApiServer, **overrides):
    payload = {
        "device_id": CAL_DEVICE,
        "model_version": status(api)["active_version"],
        "channel_permutation": CAL_SWAP,
        "gain": CAL_GAIN,
        "offset": CAL_OFFSET,
        "samples": default_samples(api),
    }
    payload.update(overrides)
    return httpx.post(f"{api.base}/admin/calibrations", json=payload, timeout=60.0)


def device_infer(api: ApiServer, raw_features, **extra):
    body = {"features": raw_features, "device_id": CAL_DEVICE, **extra}
    return httpx.post(f"{api.base}/infer", json=body, timeout=40.0)


def test_20_calibrate_device_and_infer(api):
    active = status(api)["active_version"]

    # Without a calibration for the active version, device traffic is refused
    # but untagged traffic keeps working.
    resp = device_infer(api, [0.1, 0.545])
    assert resp.status_code == 409, resp.text
    detail = resp.json()["detail"]
    assert detail["error"] == "calibration_required"
    assert detail["device_id"] == CAL_DEVICE
    assert detail["version"] == active
    assert infer(api, [0.9, 0.1]).status_code == 200

    trial = calibrate(api)
    assert trial.status_code == 201, trial.text
    published = trial.json()
    assert published["revision"] == 1
    assert published["model_version"] == active
    assert published["supersedes"] is None
    assert published["trial"]["all_correct"] is True
    assert published["trial"]["total"] == 2
    assert published["trial"]["calibrated_features"][0] == pytest.approx([0.9, 0.1])

    # Raw device readings now score like the underlying model features.
    resp = device_infer(api, raw_of_model_features(0.9, 0.1))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["version"] == active
    assert body["label"] == label_for(api, 0.9, 0.1)
    assert body["calibration_revision"] == 1
    assert body["device_id"] == CAL_DEVICE
    assert body["calibrated_features"] == pytest.approx([0.9, 0.1])

    # Untagged requests are unchanged: no revision, raw features go straight in.
    plain = infer(api, [0.9, 0.1]).json()
    assert plain["calibration_revision"] is None
    assert plain["calibrated_features"] is None
    assert plain["label"] == body["label"]
    # The revision is listed and marked applicable.
    listing = httpx.get(f"{api.base}/admin/calibrations", timeout=5).json()
    ours = [r for r in listing if r["device_id"] == CAL_DEVICE]
    assert ours[-1]["revision"] == 1
    assert ours[-1]["applicable"] is True


def test_21_failed_trial_keeps_previous_revision(api):
    before = [
        r
        for r in httpx.get(f"{api.base}/admin/calibrations", timeout=5).json()
        if r["device_id"] == CAL_DEVICE
    ]
    assert before[-1]["revision"] == 1

    resp = calibrate(
        api,
        samples=[
            {
                "features": raw_of_model_features(0.9, 0.1),
                "expected_label": label_for(api, 0.9, 0.1),
            },
            {
                "features": raw_of_model_features(0.9, 0.1),
                "expected_label": label_for(api, 0.1, 0.9),
            },
        ],
    )
    assert resp.status_code == 422, resp.text
    detail = resp.json()["detail"]
    assert detail["error"] == "calibration_validation_failed"
    assert detail["published"] is False
    assert detail["trial"]["correct"] == 1
    assert detail["trial"]["total"] == 2

    after = [
        r
        for r in httpx.get(f"{api.base}/admin/calibrations", timeout=5).json()
        if r["device_id"] == CAL_DEVICE
    ]
    assert [r["revision"] for r in after] == [1]
    # The old revision still serves.
    resp = device_infer(api, raw_of_model_features(0.9, 0.1))
    assert resp.status_code == 200
    assert resp.json()["calibration_revision"] == 1


def test_22_invalid_calibration_shapes_rejected(api):
    bad_perm = calibrate(api, channel_permutation=[0, 0])
    assert bad_perm.status_code == 422
    assert bad_perm.json()["detail"]["error"] == "invalid_calibration"

    bad_gain = calibrate(api, gain=[1.0, 2.0, 3.0])
    assert bad_gain.status_code == 422
    assert bad_gain.json()["detail"]["error"] == "invalid_calibration"

    wrong_width = calibrate(
        api, samples=[{"features": [0.1], "expected_label": "defect-A"}]
    )
    assert wrong_width.status_code == 422
    assert wrong_width.json()["detail"]["error"] == "invalid_calibration"

    wrong_version = calibrate(api, model_version="not-active")
    assert wrong_version.status_code == 409
    assert wrong_version.json()["detail"]["error"] == "calibration_version_mismatch"


def test_23_new_version_requires_recalibration(api):
    wait_for(lambda: status(api)["slots_in_use"] == 1, timeout=10)
    old = status(api)["active_version"]
    resp = upload(
        api,
        artifact(
            f"cal-{uuid.uuid4().hex[:6]}",
            W_A,
            [0.0, 0.0],
            LABELS,
            expected_active_version=old,
        ),
    )
    assert resp.status_code == 201, resp.text

    # Old revision is retained in history but not applicable to the new version.
    listing = [
        r
        for r in httpx.get(f"{api.base}/admin/calibrations", timeout=5).json()
        if r["device_id"] == CAL_DEVICE
    ]
    assert listing[-1]["revision"] == 1
    assert listing[-1]["applicable"] is False

    refused = device_infer(api, raw_of_model_features(0.9, 0.1))
    assert refused.status_code == 409, refused.text
    assert refused.json()["detail"]["error"] == "calibration_required"

    # Untagged inference on the new version is unaffected.
    assert infer(api, [0.9, 0.1]).json()["version"] == status(api)["active_version"]

    republished = calibrate(api)
    assert republished.status_code == 201, republished.text
    body = republished.json()
    assert body["revision"] == 2  # monotonic across model versions
    assert body["supersedes"] == 1
    served = device_infer(api, raw_of_model_features(0.9, 0.1)).json()
    assert served["calibration_revision"] == 2


def test_24_trial_racing_a_swap_cannot_publish_old_version(api):
    wait_for(lambda: status(api)["slots_in_use"] == 1, timeout=10)
    old = status(api)["active_version"]
    earlier = [
        r
        for r in httpx.get(f"{api.base}/admin/calibrations", timeout=5).json()
        if r["device_id"] == CAL_DEVICE
    ][-1]["revision"]

    import threading

    result: dict = {}

    def do_calibration():
        # One sample, slowed to 2000ms, so the swap below definitely commits
        # while the trial is still running on the old version.
        result["cal"] = calibrate(
            api,
            model_version=old,
            slow_sample_ms=2000,
            samples=[
                {
                    "features": raw_of_model_features(0.9, 0.1),
                    "expected_label": label_for(api, 0.9, 0.1),
                }
            ],
        )

    thread = threading.Thread(target=do_calibration)
    thread.start()
    time.sleep(0.4)
    swap = upload(
        api,
        artifact(
            f"race-{uuid.uuid4().hex[:6]}",
            W_B,
            [0.0, 0.0],
            LABELS,
            expected_active_version=old,
        ),
    )
    assert swap.status_code == 201, swap.text
    thread.join(timeout=60)
    assert "cal" in result

    cal_resp = result["cal"]
    assert cal_resp.status_code == 409, cal_resp.text
    detail = cal_resp.json()["detail"]
    assert detail["error"] == "calibration_stale"
    assert detail["published"] is False

    # Nothing was published for the old trial; the new version stays
    # uncalibrated and the revision counter did not advance.
    listing = [
        r
        for r in httpx.get(f"{api.base}/admin/calibrations", timeout=5).json()
        if r["device_id"] == CAL_DEVICE
    ]
    assert listing[-1]["revision"] == earlier
    assert device_infer(api, raw_of_model_features(0.9, 0.1)).status_code == 409


def test_25_accepted_inference_keeps_revision_and_completes_after_swap(api):
    wait_for(lambda: status(api)["slots_in_use"] == 1, timeout=10)
    republished = calibrate(api)
    assert republished.status_code == 201, republished.text
    revision = republished.json()["revision"]
    active = status(api)["active_version"]

    # Slow device inference accepted on `active` and its revision...
    slow = device_infer(
        api,
        raw_of_model_features(0.7, 0.3),
        slow_ms=6000,
        timeout_ms=150,
    )
    assert slow.status_code == 504, slow.text
    assert slow.json()["detail"]["calibration_revision"] == revision
    st = status(api)
    assert st["total_in_flight"] == 1

    # ...then a swap commits and a new calibration revision is published for
    # the new version.  Neither may disturb the accepted request.
    new_version = f"pin-{uuid.uuid4().hex[:6]}"
    swap = upload(
        api,
        artifact(new_version, W_A, [0.0, 0.0], LABELS, expected_active_version=active),
    )
    assert swap.status_code == 201, swap.text
    assert status(api)["active_version"] == new_version

    recal = calibrate(api)
    assert recal.status_code == 201, recal.text
    new_revision = recal.json()["revision"]
    assert new_revision == revision + 1

    # New device traffic uses the new revision / new version.
    fresh = device_infer(api, raw_of_model_features(0.9, 0.1))
    assert fresh.status_code == 200
    assert fresh.json()["version"] == new_version
    assert fresh.json()["calibration_revision"] == new_revision

    # The slow request still holds the old version's slot until it finishes.
    assert status(api)["total_in_flight"] == 1
    wait_for(lambda: status(api)["slots_in_use"] == 1, timeout=15)


def test_26_trial_does_not_break_two_process_cap(api):
    wait_for(lambda: status(api)["slots_in_use"] == 1, timeout=10)
    assert status(api)["slots_in_use"] == 1
    resp = calibrate(
        api,
        samples=[
            {
                "features": raw_of_model_features(0.9, 0.1),
                "expected_label": label_for(api, 0.9, 0.1),
            }
        ],
    )
    assert resp.status_code == 201, resp.text
    st = status(api)
    assert st["slots_in_use"] == 1
    assert st["capacity"] == 2


# ---------------------------------------------------------------------------
# Persistence and request validation
# ---------------------------------------------------------------------------


def test_27_calibrations_survive_restart(tmp_path):
    """Published revisions are reloaded with the promoted model on restart."""
    data_dir = tmp_path / "restart-data"
    data_dir.mkdir()
    with ApiServer(data_dir) as first:
        boot = upload(
            first,
            artifact(
                "boot",
                W_A,
                [0.0, 0.0],
                LABELS,
                test_features=[0.9, 0.1],
            ),
        )
        assert boot.status_code == 201, boot.text
        published = calibrate(first)
        assert published.status_code == 201, published.text
        assert published.json()["revision"] == 1
        assert device_infer(first, raw_of_model_features(0.9, 0.1)).status_code == 200

    # Restart against the same data directory, re-bootstrapping the promoted
    # artifact exactly as docker-compose.yml documents.
    boot_env = {
        "BOOTSTRAP_VERSION": "boot",
        "BOOTSTRAP_MODEL": str(data_dir / "models" / "boot.json"),
    }
    with ApiServer(data_dir, extra_env=boot_env) as second:
        wait_for(lambda: status(second)["active_version"] == "boot", timeout=30)
        listing = httpx.get(f"{second.base}/admin/calibrations", timeout=5).json()
        ours = [r for r in listing if r["device_id"] == CAL_DEVICE]
        assert ours and ours[-1]["revision"] == 1
        assert ours[-1]["applicable"] is True
        served = device_infer(second, raw_of_model_features(0.9, 0.1))
        assert served.status_code == 200, served.text
        assert served.json()["calibration_revision"] == 1
        assert served.json()["version"] == "boot"


def test_28_device_id_and_sample_validation(api):
    wait_for(lambda: status(api)["slots_in_use"] == 1, timeout=10)
    bad_device = httpx.post(
        f"{api.base}/infer",
        json={"features": [0.1, 0.2], "device_id": "bad device/../x"},
        timeout=5,
    )
    assert bad_device.status_code == 422

    no_samples = httpx.post(
        f"{api.base}/admin/calibrations",
        json={
            "device_id": CAL_DEVICE,
            "model_version": status(api)["active_version"],
            "samples": [],
        },
        timeout=5,
    )
    assert no_samples.status_code == 422

    no_device_cal = httpx.post(
        f"{api.base}/admin/calibrations",
        json={
            "model_version": status(api)["active_version"],
            "samples": [{"features": [0.1, 0.2], "expected_label": "x"}],
        },
        timeout=5,
    )
    assert no_device_cal.status_code == 422
