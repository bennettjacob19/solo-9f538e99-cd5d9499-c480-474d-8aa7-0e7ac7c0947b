"""Unit tests for the shared registry state machine, including the drain gate."""

from __future__ import annotations

import multiprocessing

import pytest

from app.process_manager import (
    VERSION_ACTIVE,
    VERSION_FAILED,
    VERSION_RETIRED,
    CalibrationConflict,
    CalibrationRequired,
    FeatureMismatch,
    NoActiveModel,
    NoCapacity,
    Registry,
    SwapConflict,
)


@pytest.fixture(scope="module")
def manager():
    ctx = multiprocessing.get_context("fork")
    mgr = ctx.Manager()
    yield mgr
    mgr.shutdown()


@pytest.fixture()
def registry(manager):
    reg = Registry(manager)
    yield reg
    # Clear state between tests so the shared manager stays pristine.
    for slot in list(reg._slots.keys()):
        reg._slots[slot] = {
            "slot_id": slot,
            "version": None,
            "process_state": "idle",
            "pid": None,
            "in_flight": 0,
            "accepted": 0,
            "completed": 0,
            "started_at": None,
            "idle_since": 0.0,
            "generation": reg._slots[slot]["generation"],
        }
    for version in list(reg._versions.keys()):
        del reg._versions[version]
    for key in list(reg._claims.keys()):
        del reg._claims[key]
    for device in list(reg._calibrations.keys()):
        del reg._calibrations[device]
    reg._state["active_version"] = None


def _load(reg: Registry, version: str) -> str:
    slot = reg.reserve_any_slot(version)
    reg.set_pid(version, 1000)
    reg.set_fingerprint(version, f"fp-{version}")
    return slot


def _activate(reg: Registry, version: str, old: str | None = None, token: str = "t") -> None:
    reg.begin_swap(old, token)
    reg.mark_candidate_tested(version)
    reg.commit_swap(version, old, token)


def _publish(reg: Registry, device: str, version: str, revision_doc=None):
    info = reg.version_info(version)
    return reg.publish_calibration(
        device,
        version,
        info["fingerprint"],
        revision_doc or {"channel_permutation": [0, 1], "gain": [1.0, 1.0], "offset": [0.0, 0.0]},
        [{"features": [1.0, 2.0], "expected_label": "a"}],
    )


def test_first_load_claims_none_and_activates(registry):
    slot = _load(registry, "v1")
    assert slot == "slot-0"
    registry.mark_candidate_tested("v1")
    old = registry.begin_swap(None, "tok")
    registry.commit_swap("v1", old, "tok")
    assert registry.active_version() == "v1"
    assert registry.version_info("v1")["state"] == VERSION_ACTIVE
    assert registry.snapshot()["slots_in_use"] == 1


def test_second_swap_retires_old_but_keeps_slot_until_drained(registry):
    _load(registry, "v1")
    _activate(registry, "v1")
    _load(registry, "v2")
    _activate(registry, "v2", old="v1")

    assert registry.active_version() == "v2"
    assert registry.version_info("v1")["state"] == VERSION_RETIRED
    assert registry.snapshot()["slots_in_use"] == 2  # v1 still draining
    assert not registry.has_capacity()

    slot = registry.version_info("v1")["slot_id"]
    # In-flight work blocks release.
    version, _ = registry.accept_work()
    assert version == "v2"
    assert registry.release_slot(slot, "v1", process_exited=True) is True  # no in-flight on v1
    registry.complete_work("v2")


def test_no_capacity_when_both_slots_held(registry):
    _load(registry, "v1")
    _activate(registry, "v1")
    _load(registry, "v2")
    _activate(registry, "v2", old="v1")
    with pytest.raises(NoCapacity):
        registry.reserve_any_slot("v3")
    # The current version is untouched.
    assert registry.active_version() == "v2"


def test_in_flight_request_blocks_slot_release(registry):
    _load(registry, "v1")
    _activate(registry, "v1")
    _load(registry, "v2")
    _activate(registry, "v2", old="v1")

    # A request accepted on v1 *before* the swap is counted on v1.
    # (Simulated here by accepting on v1 while it was active is not possible
    # after commit, so drive the counter the way the API does.)
    registry._slots["slot-0"] = {**registry._slots["slot-0"], "in_flight": 1}
    assert registry.release_slot("slot-0", "v1", process_exited=True) is False
    assert registry.release_slot("slot-0", "v1", process_exited=False) is False
    registry.complete_work("v1")
    assert registry.release_slot("slot-0", "v1", process_exited=True) is True
    assert registry.has_capacity()


def test_concurrent_swap_claim_allows_one_winner(registry):
    _load(registry, "v1")
    _activate(registry, "v1")
    assert registry.begin_swap("v1", "a") == "v1"
    with pytest.raises(SwapConflict, match="already in progress"):
        registry.begin_swap("v1", "b")
    registry.end_swap("v1", "a")
    assert registry.begin_swap("v1", "b") == "v1"


def test_stale_expected_version_is_conflict(registry):
    _load(registry, "v1")
    _activate(registry, "v1")
    with pytest.raises(SwapConflict, match="precondition"):
        registry.begin_swap("v0", "token")


def test_commit_requires_smoke_test(registry):
    _load(registry, "v1")
    registry.begin_swap(None, "t")
    with pytest.raises(SwapConflict, match="smoke test"):
        registry.commit_swap("v1", None, "t")


def test_failed_candidate_frees_slot_and_can_be_retried(registry):
    _load(registry, "v1")
    _activate(registry, "v1")
    registry.begin_swap("v1", "t")
    _load(registry, "bad")
    registry.fail_candidate("bad", "boom")
    registry.end_swap("v1", "t")

    info = registry.version_info("bad")
    assert info["state"] == VERSION_FAILED
    assert info["error"] == "boom"
    assert registry.has_capacity()
    assert registry.active_version() == "v1"

    registry.begin_swap("v1", "t2")
    _load(registry, "bad")  # retry allowed, keeps previous error
    assert registry.version_info("bad")["attempt"] == 2
    registry.end_swap("v1", "t2")


def test_duplicate_live_version_rejected(registry):
    _load(registry, "v1")
    _activate(registry, "v1")
    with pytest.raises(SwapConflict, match="already exists"):
        registry.reserve_any_slot("v1")


def test_accepted_work_is_pinned_to_active_version(registry):
    _load(registry, "v1")
    _activate(registry, "v1")
    version, slot = registry.accept_work()
    assert (version, slot) == ("v1", "slot-0")

    _load(registry, "v2")
    _activate(registry, "v2", old="v1")
    # The pre-existing in-flight count belongs to v1, not the new active.
    assert registry.version_in_flight("v1") == 1
    assert registry.version_in_flight("v2") == 0
    registry.complete_work("v1")
    assert registry.version_in_flight("v1") == 0


def test_feature_mismatch_does_not_count_in_flight(registry):
    _load(registry, "v1")
    registry.set_model_shape("v1", 3, 2)
    _activate(registry, "v1")
    with pytest.raises(FeatureMismatch):
        registry.accept_work(2)
    assert registry.version_in_flight("v1") == 0
    version, _ = registry.accept_work(3)
    assert version == "v1"


def test_inference_without_model_is_rejected(registry):
    with pytest.raises(NoActiveModel):
        registry.accept_work()


def test_dead_active_worker_releases_slot(registry):
    _load(registry, "v1")
    _activate(registry, "v1")
    registry.accept_work()
    registry.settle_dead_worker("v1")
    assert registry.fail_active("v1", "crashed") is True
    assert registry.active_version() is None
    assert registry.has_capacity()
    assert registry.version_info("v1")["state"] == VERSION_FAILED


# ---------------------------------------------------------------------------
# Device calibrations
# ---------------------------------------------------------------------------


def test_device_inference_without_calibration_is_rejected(registry):
    _load(registry, "v1")
    _activate(registry, "v1")
    with pytest.raises(CalibrationRequired) as exc:
        registry.accept_inference("station-7", 2)
    assert exc.value.device_id == "station-7"
    assert exc.value.version == "v1"
    assert registry.version_in_flight("v1") == 0
    # Untagged inference still works unchanged.
    admission = registry.accept_inference(None, 2)
    assert admission.version == "v1"
    assert admission.calibration is None
    registry.complete_work("v1")


def test_calibration_pins_revision_at_admission(registry):
    _load(registry, "v1")
    _activate(registry, "v1")
    first = _publish(registry, "station-7", "v1")
    assert first["revision"] == 1

    admission = registry.accept_inference("station-7", 2)
    assert admission.calibration["revision"] == 1
    assert admission.version == "v1"

    # A newer revision published while the request is in flight does not
    # change what the already-admitted request would use.
    second = _publish(registry, "station-7", "v1")
    assert second["revision"] == 2
    assert admission.calibration["revision"] == 1

    later = registry.accept_inference("station-7", 2)
    assert later.calibration["revision"] == 2
    registry.complete_work("v1")
    registry.complete_work("v1")


def test_new_model_version_invalidates_old_calibration(registry):
    _load(registry, "v1")
    _activate(registry, "v1")
    _publish(registry, "station-7", "v1")

    _load(registry, "v2")
    _activate(registry, "v2", old="v1")
    with pytest.raises(CalibrationRequired):
        registry.accept_inference("station-7", 2)
    # The old revision is still recorded, just not applicable.
    assert registry.resolve_calibration("station-7", "v2", "fp-v2") is None
    assert registry.resolve_calibration("station-7", "v1", "fp-v1") is not None


def test_calibration_for_retried_version_with_new_weights_not_reused(registry):
    _load(registry, "v1")
    _activate(registry, "v1")
    _publish(registry, "station-7", "v1")

    # v2 starts with the same version id "v1"?  Simulate the relevant case:
    # same version id, different parameter fingerprint.
    registry._versions["v1"] = {**registry._versions["v1"], "fingerprint": "fp-other"}
    with pytest.raises(CalibrationRequired):
        registry.accept_inference("station-7", 2)
    assert registry.resolve_calibration("station-7", "v1", "fp-other") is None


def test_publication_rejected_after_swap_committed(registry):
    _load(registry, "v1")
    _activate(registry, "v1")
    started = registry.version_info("v1")
    _load(registry, "v2")
    _activate(registry, "v2", old="v1")
    # The trial ran against v1 (started fingerprint fp-v1) but by the time it
    # finished v2 is active.
    with pytest.raises(CalibrationConflict, match="active version is now"):
        registry.publish_calibration(
            "station-7", "v1", started["fingerprint"], {}, [{}]
        )
    assert registry.calibration_records("station-7") == []


def test_publication_rejected_when_weights_changed_under_same_version(registry):
    _load(registry, "v1")
    _activate(registry, "v1")
    registry._versions["v1"] = {**registry._versions["v1"], "fingerprint": "fp-new"}
    with pytest.raises(CalibrationConflict, match="parameters changed"):
        registry.publish_calibration("station-7", "v1", "fp-old", {}, [{}])
    assert registry.calibration_records("station-7") == []


def test_revisions_are_monotonic_and_latest_per_version(registry):
    _load(registry, "v1")
    _activate(registry, "v1")
    assert _publish(registry, "d", "v1")["revision"] == 1
    assert _publish(registry, "d", "v1")["revision"] == 2
    third = _publish(registry, "d", "v1")
    assert third["revision"] == 3
    assert third["supersedes"] == 2
    latest = registry.latest_calibration_for_device("d")
    assert latest["revision"] == 3


def test_trial_is_counted_as_in_flight_and_blocks_drain(registry):
    _load(registry, "v1")
    _activate(registry, "v1")
    info = registry.begin_trial("v1")
    assert info is not None and info["version"] == "v1"
    assert registry.version_in_flight("v1") == 1
    _load(registry, "v2")
    _activate(registry, "v2", old="v1")
    # v1 still holds trial work; its slot may not be released.
    assert registry.release_slot("slot-0", "v1", process_exited=True) is False
    registry.end_trial("v1")
    assert registry.release_slot("slot-0", "v1", process_exited=True) is True


def test_begin_trial_rejects_non_active_version(registry):
    _load(registry, "v1")
    _activate(registry, "v1")
    _load(registry, "v2")
    _activate(registry, "v2", old="v1")
    assert registry.begin_trial("v1") is None
    assert registry.version_in_flight("v1") == 0


def test_restored_calibrations_apply_to_matching_version(registry):
    _load(registry, "v1")
    _activate(registry, "v1")
    record = _publish(registry, "d", "v1")
    registry.restore_calibrations({"d2": [dict(record, device_id="d2")]})
    admission = registry.accept_inference("d2", 2)
    assert admission.calibration is not None
    assert admission.calibration["revision"] == record["revision"]
    registry.complete_work("v1")
