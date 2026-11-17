"""Edge QC classifier inference service with two-process hot model swapping.

Public surface (see ``/docs``):

* ``POST /infer``            -- classify a feature vector with a client timeout
* ``POST /admin/models``     -- upload a classifier artifact and hot-swap it in
* ``POST /admin/calibrations`` -- trial a per-device channel calibration on the
  current model version and publish it only if every reference sample passes
* ``GET  /admin/calibrations`` -- calibration revisions per device
* ``GET  /admin/versions``   -- version registry
* ``GET  /admin/status``     -- active version, in-flight counts, process states
* ``GET  /healthz``          -- liveness

The invariants live in ``app/process_manager.py``.  In short: every model
version runs in its own OS process, at most two such processes exist at once,
requests are pinned to the version active at admission, and a retired version
only releases its slot once every inference it actually accepted has finished
inside its worker -- regardless of whether the client is still waiting.

A device-tagged inference is additionally pinned to the calibration revision
that matched the active version *and* its exact model parameters at admission.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import multiprocessing
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import httpx
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

from app.calibration import CalibrationInvalid, CalibrationTransform
from app.models import ModelValidationError, load_model, save_artifact, validate_artifact
from app.process_manager import (
    STATE_DRAINING,
    STATE_IDLE,
    VERSION_ACTIVE,
    VERSION_FAILED,
    VERSION_LOADING,
    VERSION_RETIRED,
    CalibrationConflict,
    CalibrationRequired,
    FeatureMismatch,
    NoActiveModel,
    NoCapacity,
    Registry,
    SwapConflict,
    WorkerHandle,
)

logger = logging.getLogger("qc.service")
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
# Device ids appear in URLs and are persisted; keep them to a safe alphabet.
DEVICE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$")
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
MODEL_DIR = DATA_DIR / "models"
STAGING_DIR = DATA_DIR / "staging"
WORK_DIR = DATA_DIR / "workers"
CALIBRATION_PATH = DATA_DIR / "calibrations.json"
DEFAULT_TIMEOUT_MS = int(os.environ.get("DEFAULT_INFER_TIMEOUT_MS", "5000"))
MAX_TIMEOUT_MS = 600_000
# How long an upload may wait for a draining slot to be released before 429.
SWAP_WAIT_SECONDS = float(os.environ.get("SWAP_WAIT_SECONDS", "3"))
# How long shutdown waits for accepted inference to finish before killing workers.
SHUTDOWN_GRACE_SECONDS = float(os.environ.get("SHUTDOWN_GRACE_SECONDS", "20"))


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class InferRequest(BaseModel):
    features: List[float] = Field(..., description="Feature vector for the classifier")
    device_id: Optional[str] = Field(
        None,
        description=(
            "Optional QC station id. When present, traffic is accepted only if the "
            "active model version has a published calibration for this device; the "
            "raw features are transformed by that revision before scoring. Without "
            "it the request behaves exactly as before (no calibration)."
        ),
    )
    timeout_ms: Optional[int] = Field(
        None,
        ge=1,
        le=MAX_TIMEOUT_MS,
        description=(
            "How long the client is willing to wait, in milliseconds. On expiry the "
            "API returns 504; the model process keeps computing to completion."
        ),
    )
    timeout_seconds: Optional[float] = Field(
        None,
        gt=0,
        le=MAX_TIMEOUT_MS / 1000.0,
        description=(
            "Same as timeout_ms, expressed in seconds (e.g. 0.2). If both are "
            "provided, timeout_ms wins."
        ),
    )
    slow_ms: int = Field(
        0,
        ge=0,
        le=600_000,
        description=(
            "Demo hook: ask the model process to spend this long on the computation "
            "before scoring. Use with a small timeout_ms to demonstrate the 504 / "
            "drain behaviour."
        ),
    )

    @field_validator("features")
    @classmethod
    def _features_present(cls, value: List[float]) -> List[float]:
        if not value:
            raise ValueError("features must contain at least one value")
        return value

    @field_validator("device_id")
    @classmethod
    def _device_valid(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        if not DEVICE_RE.match(value):
            raise ValueError(
                "device_id must match ^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$"
            )
        return value


class InferResponse(BaseModel):
    version: str
    label: str
    score: float
    inference_id: str
    elapsed_ms: float
    device_id: Optional[str] = Field(
        None, description="Echo of the request's device id (None when omitted)"
    )
    calibration_revision: Optional[int] = Field(
        None,
        description=(
            "Calibration revision used to transform the raw features, pinned at "
            "admission. Null for requests without a device id."
        ),
    )
    calibrated_features: Optional[List[float]] = Field(
        None,
        description="The transformed feature vector actually scored (device requests only)",
    )


class ReferenceSample(BaseModel):
    features: List[float] = Field(
        ..., description="Raw, uncalibrated channel readings from the device"
    )
    expected_label: str = Field(..., description="Ground-truth label for this sample")

    @field_validator("features")
    @classmethod
    def _sample_features_present(cls, value: List[float]) -> List[float]:
        if not value:
            raise ValueError("sample features must contain at least one value")
        return value


class CalibrationRequest(BaseModel):
    device_id: str = Field(..., description="QC station / sensor unit id")
    model_version: str = Field(
        ...,
        description=(
            "Model version the calibration is trialled against. Must be the "
            "currently active version; the revision is only published for that "
            "version and its exact parameters."
        ),
    )
    channel_permutation: Optional[List[int]] = Field(
        None,
        description=(
            "Order of raw channels as the model expects them: calibrated[i] = "
            "raw[channel_permutation[i]]. Must be a permutation of 0..n-1; "
            "defaults to identity."
        ),
    )
    gain: Optional[Union[float, List[float]]] = Field(
        None, description="Per-channel multiplier or one scalar for all channels; default 1"
    )
    offset: Optional[Union[float, List[float]]] = Field(
        None, description="Per-channel additive bias or one scalar; default 0"
    )
    samples: List[ReferenceSample] = Field(
        ...,
        min_length=1,
        description=(
            "Reference samples with expected labels. ALL of them must be "
            "classified correctly by the given version after calibration; "
            "otherwise nothing is published and the previous revision stays valid."
        ),
    )
    slow_sample_ms: int = Field(
        0,
        ge=0,
        le=600_000,
        description=(
            "Demo hook: slow each trial inference down by this many ms, to observe "
            "a calibration trial interleaving with a model swap."
        ),
    )

    @field_validator("device_id")
    @classmethod
    def _device_valid(cls, value: str) -> str:
        if not DEVICE_RE.match(value):
            raise ValueError(
                "device_id must match ^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$"
            )
        return value


class SampleResult(BaseModel):
    index: int
    expected_label: str
    actual_label: str
    score: float
    correct: bool


class CalibrationTrialView(BaseModel):
    version: str
    device_id: str
    calibration_revision: Optional[int] = Field(
        None, description="Revision used for this trial (existing one, or null)"
    )
    all_correct: bool
    correct: int
    total: int
    sample_results: List[SampleResult]
    calibrated_features: List[List[float]]


class CalibrationPublishedView(BaseModel):
    device_id: str
    revision: int
    model_version: str
    model_fingerprint: str
    published_at: float
    supersedes: Optional[int]
    transform: Dict[str, Any]
    sample_count: int
    trial: CalibrationTrialView


class CalibrationRecordView(BaseModel):
    device_id: str
    revision: int
    model_version: str
    model_fingerprint: str
    published_at: float
    status: str
    sample_count: int
    supersedes: Optional[int]
    transform: Dict[str, Any]
    applicable: bool = Field(
        ...,
        description="True only if this revision matches the currently active version and weights",
    )


class TimeoutResponse(BaseModel):
    detail: Dict[str, Any]


class ModelUpload(BaseModel):
    version: str = Field(..., description="Unique version id, e.g. 'v3'")
    weights: List[List[float]] = Field(..., description="(n_classes x n_features) matrix")
    bias: List[float] = Field(..., description="Bias vector of length n_classes")
    labels: List[str] = Field(..., description="Class label for each weight row")
    test_features: Optional[List[float]] = Field(
        None,
        description="Vector used for the pre-activation smoke test; defaults to all zeros",
    )
    expected_active_version: Optional[str] = Field(
        None,
        description=(
            "Optimistic-concurrency guard: swap only if this is still the active "
            "version. Of concurrent swaps away from the same old version at most "
            "one succeeds."
        ),
    )
    slow_load_ms: int = Field(
        0,
        ge=0,
        le=60_000,
        description="Demo hook: delay worker readiness to observe the old version serving.",
    )
    simulate_load_failure: bool = Field(
        False,
        description="Demo hook: make the worker report a load failure after starting.",
    )


class SlotView(BaseModel):
    slot_id: str
    version: Optional[str]
    process_state: str
    pid: Optional[int]
    in_flight: int
    accepted: int
    completed: int
    alive: Optional[bool]


class StatusView(BaseModel):
    active_version: Optional[str]
    version: Optional[str]
    total_in_flight: int
    total_accepted: int
    total_completed: int
    slots_in_use: int
    capacity: int
    slots: List[SlotView]
    versions: List[Dict[str, Any]]
    calibrated_devices: List[str] = Field(
        default_factory=list,
        description="Devices with at least one published calibration revision",
    )
    device_calibration_revisions: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description=(
            "Per device, the latest published revision and whether it applies to "
            "the currently active version and parameters"
        ),
    )


class SwapResult(BaseModel):
    version: str
    previous_version: Optional[str]
    slot_id: str
    state: str
    process_state: str
    pid: Optional[int]
    activated_at: float
    test_label: str
    load_seconds: float
    model_fingerprint: Optional[str] = Field(
        None,
        description=(
            "Identity of the promoted parameters. Calibrations are bound to this "
            "value; device inference needs a calibration matching the new version "
            "and fingerprint."
        ),
    )


# ---------------------------------------------------------------------------
# Inference service
# ---------------------------------------------------------------------------


class InferenceService:
    def __init__(self) -> None:
        self.manager: Any = None
        self.registry: Registry
        self.handles: Dict[str, WorkerHandle] = {}
        self.client: Optional[httpx.AsyncClient] = None
        self._drain_task: Optional[asyncio.Task] = None
        self._closing = False
        self._draining: set[str] = set()
        self._background: set[asyncio.Task] = set()
        self._swap_lock = asyncio.Lock()  # serialises swaps inside this process
        # Serialises calibration trials/publications so revision numbers are
        # assigned in the same order the trials were accepted.
        self._calibration_lock = asyncio.Lock()
        self._lock_file: Any = None

    # -- lifecycle -------------------------------------------------------
    async def startup(self) -> None:
        for path in (DATA_DIR, MODEL_DIR, STAGING_DIR, WORK_DIR):
            path.mkdir(parents=True, exist_ok=True)
        self._lock_file = open(DATA_DIR / "api.lock", "a+")
        try:
            import fcntl

            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except ImportError:  # pragma: no cover - non-POSIX
            pass
        except OSError as exc:
            raise RuntimeError(
                "another API process already owns this data directory; the "
                "service must run as a single process so the two-slot budget "
                "has a single owner"
            ) from exc
        self.manager = _new_manager()
        self.registry = Registry(self.manager)
        self._load_calibrations()
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5.0, read=None, write=None, pool=None),
            limits=httpx.Limits(max_connections=512, max_keepalive_connections=128),
        )
        self._drain_task = asyncio.create_task(self._drain_loop())
        bootstrap_version = os.environ.get("BOOTSTRAP_VERSION")
        bootstrap_model = os.environ.get("BOOTSTRAP_MODEL")
        if bootstrap_version and bootstrap_model and Path(bootstrap_model).is_file():
            try:
                await self.swap_in_path(
                    version=bootstrap_version,
                    source_path=bootstrap_model,
                    expected_active=None,
                    test_features=None,
                    slow_load_ms=0,
                    simulate_load_failure=False,
                )
                logger.info("bootstrapped model %s", bootstrap_version)
            except Exception:
                logger.exception(
                    "bootstrap model %s failed; service starts without a model",
                    bootstrap_version,
                )

    async def shutdown(self) -> None:
        self._closing = True
        if self._drain_task is not None:
            self._drain_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._drain_task
            self._drain_task = None
        # Let accepted inference settle so no in-flight work is killed halfway
        # and no worker is left orphaned.  Bounded so shutdown cannot hang.
        if self._background:
            logger.info("waiting for %d in-flight inference call(s)", len(self._background))
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(
                    asyncio.gather(*self._background, return_exceptions=True),
                    timeout=SHUTDOWN_GRACE_SECONDS,
                )
        for task in list(self._background):
            task.cancel()
        for handle in list(self.handles.values()):
            await asyncio.to_thread(handle.stop, 5.0)
        self.handles.clear()
        if self.client is not None:
            await self.client.aclose()
            self.client = None
        if self.manager is not None:
            with contextlib.suppress(Exception):
                self.manager.shutdown()
            self.manager = None
        if self._lock_file is not None:
            with contextlib.suppress(Exception):
                self._lock_file.close()
            self._lock_file = None

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _artifact_path(version: str) -> Path:
        return MODEL_DIR / f"{version}.json"

    def _load_calibrations(self) -> None:
        """Restore published calibration revisions from disk (best effort).

        The registry lives in a manager process that dies with the API, so the
        revision history is persisted next to the promoted model artifacts.
        """
        try:
            with open(CALIBRATION_PATH, "r", encoding="utf-8") as fh:
                doc = json.load(fh)
        except FileNotFoundError:
            return
        except (OSError, json.JSONDecodeError):
            logger.exception("could not read %s; starting with no calibrations", CALIBRATION_PATH)
            return
        devices = doc.get("devices") if isinstance(doc, dict) else None
        if not isinstance(devices, dict):
            return
        restored: Dict[str, List[Dict[str, Any]]] = {}
        for device_id, records in devices.items():
            if isinstance(device_id, str) and isinstance(records, list):
                restored[device_id] = [rec for rec in records if isinstance(rec, dict)]
        self.registry.restore_calibrations(restored)
        logger.info("restored calibrations for %d device(s)", len(restored))

    def _persist_calibrations(self) -> None:
        """Write the whole revision history; call with the calibration lock held."""
        snapshot = self.registry.snapshot()["calibrations"]
        tmp = CALIBRATION_PATH.with_suffix(f".json.tmp.{os.getpid()}")
        try:
            CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump({"devices": snapshot}, fh, indent=2, sort_keys=True)
            os.replace(tmp, CALIBRATION_PATH)
        except OSError:
            logger.exception("failed to persist calibrations to %s", CALIBRATION_PATH)
            with contextlib.suppress(FileNotFoundError):
                tmp.unlink()

    @staticmethod
    def _model_fingerprint_from_path(path: str, version: str) -> str:
        return load_model(path, version=version).fingerprint()

    def _persist_artifact(self, version: str, source_path: str) -> None:
        """Copy the promoted artifact into the durable per-version location."""
        target = self._artifact_path(version)
        tmp = target.with_suffix(target.suffix + f".tmp.{os.getpid()}")
        with open(source_path, "rb") as src, open(tmp, "wb") as dst:
            dst.write(src.read())
        os.replace(tmp, target)

    async def _reserve_slot(self, version: str) -> str:
        """Reserve a process slot for a candidate, waiting briefly for a drain.

        A slot that is draining with zero in-flight work is about to be freed
        by the drain loop; waiting a couple of seconds avoids spurious 429s
        without ever exceeding two processes.  If the drain is held up by real
        in-flight work the wait expires and the caller gets 429.
        """
        deadline = time.monotonic() + SWAP_WAIT_SECONDS
        while True:
            try:
                return self.registry.reserve_any_slot(version)
            except SwapConflict as exc:
                # e.g. the version id was registered concurrently, or a retry
                # of a failed id raced a live one.
                raise HTTPException(
                    status_code=409,
                    detail={"error": "version_exists", "message": str(exc)},
                ) from exc
            except NoCapacity as exc:
                if time.monotonic() >= deadline:
                    raise HTTPException(
                        status_code=429,
                        detail={
                            "error": "capacity_exhausted",
                            "message": str(exc),
                            "active_version": self.registry.active_version(),
                        },
                        headers={"Retry-After": "1"},
                    ) from exc
                await asyncio.sleep(0.05)

    async def _salvage_candidate(
        self, version: str, handle: WorkerHandle, reason: str
    ) -> None:
        """Stop a failed candidate, wait for exit, then free its slot."""
        self.handles.pop(version, None)
        await asyncio.to_thread(handle.stop, 5.0)
        self.registry.fail_candidate(version, reason)
        logger.warning("candidate %s failed: %s", version, reason)

    async def _test_candidate(
        self, version: str, test_features: Optional[List[float]]
    ) -> Dict[str, Any]:
        handle = self.handles[version]
        # Test against the exact artifact file this candidate process loaded,
        # not a shared per-version path (concurrent uploads use staged files).
        artifact = load_model(handle.model_path, version=version)
        features = (
            list(test_features)
            if test_features is not None
            else [0.0] * artifact.n_features
        )
        if len(features) != artifact.n_features:
            raise RuntimeError(
                f"test_features must have {artifact.n_features} values, got {len(features)}"
            )
        assert self.client is not None
        resp = await self.client.post(
            handle.inference_url(),
            json={"features": features},
            timeout=httpx.Timeout(10.0),
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"candidate smoke test failed with HTTP {resp.status_code}: {resp.text[:300]}"
            )
        body = resp.json()
        if body.get("version") != version:
            raise RuntimeError("candidate reported a mismatched version")
        if body.get("label") not in artifact.labels:
            raise RuntimeError("candidate returned a label outside the label set")
        return body

    # -- swap ------------------------------------------------------------
    async def swap_in_path(
        self,
        version: str,
        source_path: str,
        expected_active: Optional[str],
        test_features: Optional[List[float]],
        slow_load_ms: int,
        simulate_load_failure: bool,
    ) -> SwapResult:
        """Load a candidate in a fresh process, smoke-test it, then promote it.

        The old version keeps serving until the promotion is a single atomic
        registry operation.  On any failure the candidate is stopped, its slot
        freed, and the old version is left untouched.
        """
        # Snapshot the active version *before* waiting on the in-process swap
        # lock.  ``expected_active`` replaces the snapshot when the caller
        # supplies one.  Competing swaps therefore carry the same old version,
        # and the registry claim (keyed by that version) lets exactly one win.
        observed = (
            expected_active
            if expected_active is not None
            else self.registry.active_version()
        )
        # Fail fast on a known-live version id before taking the swap claim, so
        # the common duplicate-upload case never disturbs the claim table.
        existing = self.registry.version_info(version)
        if existing is not None and existing["state"] != VERSION_FAILED:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "version_exists",
                    "message": f"version {version!r} already exists ({existing['state']})",
                },
            )
        token = uuid.uuid4().hex
        # Claim outside the in-process lock so that a second swap observing the
        # same old version fails immediately instead of queueing behind the
        # first (which may be slow to load) and then failing anyway.
        try:
            old_version = self.registry.begin_swap(observed, token)
        except SwapConflict as exc:
            raise HTTPException(
                status_code=409,
                detail={"error": "swap_conflict", "message": str(exc)},
            ) from exc

        async with self._swap_lock:
            slot_id: Optional[str] = None
            handle: Optional[WorkerHandle] = None
            committed = False
            try:
                # Re-check under the lock: a concurrent request may have
                # registered this version while we waited.
                slot_id = await self._reserve_slot(version)

                handle = WorkerHandle(
                    slot_id=slot_id,
                    version=version,
                    model_path=source_path,
                    work_dir=str(WORK_DIR),
                )
                self.handles[version] = handle
                handle.spawn(
                    slow_load_ms=slow_load_ms,
                    simulate_load_failure=simulate_load_failure,
                )
                self.registry.set_pid(version, handle.pid or -1)

                load_timeout = 30.0 + max(0, slow_load_ms) / 1000.0
                try:
                    ready = await asyncio.to_thread(handle.wait_ready, load_timeout)
                except Exception as exc:
                    await self._salvage_candidate(version, handle, str(exc))
                    raise HTTPException(
                        status_code=502,
                        detail={
                            "error": "candidate_load_failed",
                            "message": str(exc),
                            "fallback_version": self.registry.active_version(),
                        },
                    ) from exc
                if not ready.get("ok"):
                    reason = str(ready.get("error") or "worker reported load failure")
                    await self._salvage_candidate(version, handle, reason)
                    raise HTTPException(
                        status_code=502,
                        detail={
                            "error": "candidate_load_failed",
                            "message": reason,
                            "fallback_version": self.registry.active_version(),
                        },
                    )
                if ready.get("n_features") is not None:
                    self.registry.set_model_shape(
                        version,
                        int(ready["n_features"]),
                        int(ready.get("n_classes", 0)),
                    )
                # Record exactly what this candidate loaded.  Calibrations are
                # bound to this fingerprint, so a failed id retried with
                # different weights can never inherit an earlier calibration.
                try:
                    fingerprint = self._model_fingerprint_from_path(
                        handle.model_path, version
                    )
                except ModelValidationError as exc:  # pragma: no cover - worker loaded it
                    await self._salvage_candidate(version, handle, str(exc))
                    raise HTTPException(
                        status_code=502,
                        detail={
                            "error": "candidate_load_failed",
                            "message": str(exc),
                            "fallback_version": self.registry.active_version(),
                        },
                    ) from exc
                self.registry.set_fingerprint(version, fingerprint)

                try:
                    test_body = await self._test_candidate(version, test_features)
                except Exception as exc:
                    await self._salvage_candidate(
                        version, handle, f"smoke test failed: {exc}"
                    )
                    raise HTTPException(
                        status_code=502,
                        detail={
                            "error": "candidate_test_failed",
                            "message": str(exc),
                            "fallback_version": self.registry.active_version(),
                        },
                    ) from exc

                self.registry.mark_candidate_tested(version)
                try:
                    self.registry.commit_swap(version, old_version, token)
                except SwapConflict as exc:
                    await self._salvage_candidate(version, handle, str(exc))
                    raise HTTPException(
                        status_code=409,
                        detail={
                            "error": "swap_conflict",
                            "message": str(exc),
                            "active_version": self.registry.active_version(),
                        },
                    ) from exc
                committed = True
                # Keep a durable copy of what is now serving.  Staged uploads
                # are cleaned up by the endpoint; this snapshot is what a
                # future restart can reload.
                try:
                    self._persist_artifact(version, source_path)
                except Exception:
                    logger.exception("failed to persist artifact for %s", version)

                new_info = self.registry.version_info(version)
                assert new_info is not None
                logger.info(
                    "activated model %s in %s (previous=%s, pid=%s)",
                    version,
                    slot_id,
                    old_version,
                    handle.pid,
                )
                return SwapResult(
                    version=version,
                    previous_version=old_version,
                    slot_id=slot_id,
                    state=new_info["state"],
                    process_state=new_info["process_state"],
                    pid=handle.pid,
                    activated_at=float(new_info["activated_at"]),
                    test_label=test_body["label"],
                    load_seconds=float(ready.get("load_seconds", 0.0)),
                    model_fingerprint=new_info.get("fingerprint"),
                )
            except HTTPException:
                raise
            except Exception as exc:
                logger.exception("swap of %s failed", version)
                raise HTTPException(
                    status_code=500,
                    detail={"error": "swap_failed", "message": str(exc)},
                ) from exc
            finally:
                if not committed:
                    if handle is not None and version in self.handles:
                        await self._salvage_candidate(version, handle, "swap aborted")
                    elif slot_id is not None:
                        self.registry.fail_candidate(version, "swap aborted")
                self.registry.end_swap(old_version, token)

    # -- calibration -----------------------------------------------------
    async def calibrate(
        self, req: "CalibrationRequest"
    ) -> "CalibrationPublishedView":
        """Trial a device calibration on the current version, then publish it.

        Ordering rules that make interleaving safe:

        * The trial is bound to the version *and* parameter fingerprint read at
          admission.  All sample inferences execute inside that version's model
          process (counted in its in-flight work, so a draining process is not
          stopped underneath them).
        * Nothing is published unless **every** sample is classified with its
          expected label.
        * Publication re-checks the active version/fingerprint under the
          registry lock.  If a model swap or a parameter change committed while
          the trial ran, publication fails with 409 and the old revision stays
          in force -- an old-version trial can never become a new-version
          calibration.
        """
        # Admit the trial against the requested version in one atomic step: the
        # version must be active right now, its fingerprint is pinned, and the
        # whole trial is counted as in-flight model work (so a drain cannot
        # stop the process underneath it).
        trial_info = self.registry.begin_trial(req.model_version)
        if trial_info is None:
            active = self.registry.active_version()
            if active is None:
                raise HTTPException(
                    status_code=503,
                    detail={"error": "no_active_model", "message": "no model is active"},
                )
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "calibration_version_mismatch",
                    "message": (
                        f"calibration names model_version {req.model_version!r}, but "
                        f"the active version is {active!r}; calibrate against the "
                        "version that will serve the device"
                    ),
                    "active_version": active,
                },
            )

        version = req.model_version
        fingerprint = trial_info.get("fingerprint")
        handle = self.handles.get(version)
        info = self.registry.version_info(version)
        if (
            not fingerprint
            or handle is None
            or info is None
            or info.get("n_features") is None
        ):
            # An active version always has a worker, a shape and a fingerprint
            # (recorded before its promotion).  If not, it is mid-teardown.
            self.registry.end_trial(version)
            raise HTTPException(
                status_code=503,
                detail={"error": "worker_missing", "message": f"worker for {version} is gone"},
            )
        n_features = int(info["n_features"])

        try:
            transform = CalibrationTransform.build(
                n_features,
                req.channel_permutation,
                req.gain,
                req.offset,
            )
        except CalibrationInvalid as exc:
            self.registry.end_trial(version)
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "invalid_calibration",
                    "message": str(exc),
                    "model_version": version,
                    "expected_features": n_features,
                },
            ) from exc

        expected_width = transform.n_features
        for index, sample in enumerate(req.samples):
            if len(sample.features) != expected_width:
                self.registry.end_trial(version)
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error": "invalid_calibration",
                        "message": (
                            f"sample {index} has {len(sample.features)} raw features; "
                            f"version {version!r} expects {expected_width}"
                        ),
                    },
                )

        # Transform every sample up front.  A NaN/infinite raw reading is a
        # client error, detected before any model work is done.
        calibrated: List[List[float]] = []
        try:
            calibrated = [transform.apply(s.features) for s in req.samples]
        except CalibrationInvalid as exc:
            self.registry.end_trial(version)
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "invalid_calibration",
                    "message": f"reference samples are not usable: {exc}",
                },
            ) from exc

        results: List[SampleResult] = []
        worker_error: Optional[str] = None

        async def _trial() -> None:
            nonlocal worker_error
            try:
                await self._run_trial_samples(version, handle, req, calibrated, results)
            except (RuntimeError, httpx.HTTPError) as exc:
                worker_error = str(exc)
            finally:
                # The in-flight count must clear only when the model process is
                # actually done, even if the calibration client disconnects.
                self.registry.end_trial(version)

        task = asyncio.create_task(_trial())
        self._background.add(task)
        task.add_done_callback(self._background.discard)
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            # The trial keeps running to completion against the pinned version;
            # nothing is published for this (abandoned) request.
            raise

        if worker_error is not None:
            raise HTTPException(
                status_code=502,
                detail={
                    "error": "calibration_trial_failed",
                    "message": worker_error,
                    "model_version": version,
                    "published": False,
                },
            )

        all_correct = all(r.correct for r in results)
        existing = self.registry.resolve_calibration(req.device_id, version, fingerprint)
        trial = CalibrationTrialView(
            version=version,
            device_id=req.device_id,
            calibration_revision=existing["revision"] if existing else None,
            all_correct=all_correct,
            correct=sum(1 for r in results if r.correct),
            total=len(results),
            sample_results=results,
            calibrated_features=calibrated,
        )
        if not all_correct:
            logger.warning(
                "calibration for device %s on %s failed %d/%d sample(s); revision kept",
                req.device_id,
                version,
                len(results) - trial.correct,
                len(results),
            )
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "calibration_validation_failed",
                    "message": (
                        f"{len(results) - trial.correct} of {len(results)} reference "
                        "sample(s) were misclassified; nothing was published and the "
                        "previous revision, if any, remains in force"
                    ),
                    "device_id": req.device_id,
                    "model_version": version,
                    "published": False,
                    "trial": trial.model_dump(),
                },
            )

        async with self._calibration_lock:
            try:
                record = self.registry.publish_calibration(
                    device_id=req.device_id,
                    model_version=version,
                    model_fingerprint=fingerprint,
                    transform_doc=transform.to_doc(),
                    samples=[s.model_dump() for s in req.samples],
                )
            except CalibrationConflict as exc:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error": "calibration_stale",
                        "message": str(exc),
                        "active_version": self.registry.active_version(),
                        "published": False,
                    },
                ) from exc
            self._persist_calibrations()

        logger.info(
            "published calibration r%d for device %s on %s",
            record["revision"],
            req.device_id,
            version,
        )
        return CalibrationPublishedView(
            device_id=record["device_id"],
            revision=int(record["revision"]),
            model_version=record["model_version"],
            model_fingerprint=record["model_fingerprint"],
            published_at=float(record["published_at"]),
            supersedes=record.get("supersedes"),
            transform=record["transform"],
            sample_count=int(record["sample_count"]),
            trial=trial,
        )

    async def _run_trial_samples(
        self,
        version: str,
        handle: WorkerHandle,
        req: "CalibrationRequest",
        calibrated: List[List[float]],
        results: List["SampleResult"],
    ) -> None:
        """Score every calibrated reference sample in the pinned model process."""
        assert self.client is not None
        url = handle.inference_url()
        for index, sample in enumerate(req.samples):
            features = calibrated[index]
            payload: Dict[str, Any] = {"features": features}
            if req.slow_sample_ms:
                payload["slow_ms"] = req.slow_sample_ms
            resp = await self.client.post(
                url, json=payload, timeout=httpx.Timeout(60.0)
            )
            if resp.status_code != 200:
                raise RuntimeError(
                    f"model process returned HTTP {resp.status_code} for sample "
                    f"{index}: {resp.text[:300]}"
                )
            body = resp.json()
            if body.get("version") != version:
                raise RuntimeError(
                    f"model process reported version {body.get('version')!r} while "
                    f"trialling {version!r}"
                )
            actual = str(body.get("label"))
            results.append(
                SampleResult(
                    index=index,
                    expected_label=sample.expected_label,
                    actual_label=actual,
                    score=float(body.get("score", 0.0)),
                    correct=actual == sample.expected_label,
                )
            )

    def calibration_records(self, device_id: Optional[str]) -> List["CalibrationRecordView"]:
        snapshot = self.registry.snapshot()
        active = snapshot["active_version"]
        active_info = snapshot["versions"].get(active) if active else None
        active_fp = active_info.get("fingerprint") if active_info else None
        records: List[CalibrationRecordView] = []
        for dev, entries in snapshot["calibrations"].items():
            if device_id is not None and dev != device_id:
                continue
            for rec in entries:
                records.append(
                    CalibrationRecordView(
                        device_id=dev,
                        revision=int(rec["revision"]),
                        model_version=rec["model_version"],
                        model_fingerprint=rec["model_fingerprint"],
                        published_at=float(rec["published_at"]),
                        status=str(rec.get("status", "published")),
                        sample_count=int(rec["sample_count"]),
                        supersedes=rec.get("supersedes"),
                        transform=rec["transform"],
                        applicable=(
                            rec["model_version"] == active
                            and rec["model_fingerprint"] == active_fp
                        ),
                    )
                )
        records.sort(key=lambda r: (r.device_id, r.revision))
        return records

    # -- draining --------------------------------------------------------
    async def _drain_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(0.05)
                await self.drain_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("drain loop error")

    async def drain_once(self) -> None:
        snapshot = self.registry.snapshot()
        for version, info in snapshot["versions"].items():
            handle = self.handles.get(version)
            # Unexpected worker death: a crashed worker cannot be part of the
            # two-process budget any more, but its slot must still be reclaimed
            # so the machine does not appear full forever.
            if handle is not None and not handle.alive():
                self._reap_dead_worker(version, info)

            if info["state"] not in (VERSION_RETIRED, VERSION_FAILED):
                continue
            if info["process_state"] not in (STATE_DRAINING, STATE_IDLE):
                continue
            slot = next(
                (s for s in snapshot["slots"] if s["slot_id"] == info["slot_id"]),
                None,
            )
            if slot is None or slot["version"] != version:
                continue
            if version in self._draining:
                continue
            # The drain gate: only stop the process once no accepted request
            # is still running inside it.
            if self.registry.version_in_flight(version) > 0:
                continue
            self._draining.add(version)
            try:
                if handle is not None:
                    await asyncio.to_thread(handle.stop, 10.0)
                    await asyncio.to_thread(handle.wait_exit, 10.0)
                released = self.registry.release_slot(
                    slot["slot_id"], version, process_exited=True
                )
                if released:
                    self.handles.pop(version, None)
                    logger.info("drained %s; slot %s released", version, slot["slot_id"])
            except Exception:
                logger.exception("failed to drain %s", version)
            finally:
                self._draining.discard(version)

    def _reap_dead_worker(self, version: str, info: Dict[str, Any]) -> None:
        """Reclaim bookkeeping for a worker process that died unexpectedly."""
        self.handles.pop(version, None)
        if info["state"] in (VERSION_ACTIVE, VERSION_LOADING):
            self.registry.settle_dead_worker(version)
            if self.registry.fail_active(version, "worker process exited unexpectedly"):
                logger.error("active model %s died; service has no active model", version)
        elif info["state"] in (VERSION_RETIRED, VERSION_FAILED):
            self.registry.settle_dead_worker(version)
            with contextlib.suppress(Exception):
                self.registry.release_slot(info["slot_id"], version, process_exited=True)

    # -- inference -------------------------------------------------------
    async def infer(
        self,
        features: List[float],
        timeout_ms: int,
        slow_ms: int = 0,
        device_id: Optional[str] = None,
    ) -> InferResponse:
        try:
            admission = self.registry.accept_inference(device_id, len(features))
        except FeatureMismatch as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "feature_dimension_mismatch",
                    "message": str(exc),
                    "version": exc.version,
                    "expected_features": exc.expected,
                    "received_features": len(features),
                },
            ) from exc
        except CalibrationRequired as exc:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "calibration_required",
                    "message": str(exc),
                    "device_id": exc.device_id,
                    "version": exc.version,
                    "action": (
                        "POST /admin/calibrations with this device_id and "
                        f"model_version={exc.version!r}; inference without device_id "
                        "continues on the uncalibrated model"
                    ),
                },
            ) from exc
        except NoActiveModel as exc:
            raise HTTPException(
                status_code=503,
                detail={"error": "no_active_model", "message": str(exc)},
            ) from exc

        version = admission.version
        calibration = admission.calibration
        calibrated_features: Optional[List[float]] = None
        revision: Optional[int] = None
        if calibration is not None:
            transform = CalibrationTransform.from_doc(calibration["transform"])
            try:
                features = transform.apply(features)
            except CalibrationInvalid as exc:  # pragma: no cover - validated at publish
                self.registry.complete_work(version)
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": "calibration_corrupt",
                        "message": str(exc),
                        "version": version,
                        "calibration_revision": calibration["revision"],
                    },
                ) from exc
            calibrated_features = features
            revision = int(calibration["revision"])

        handle = self.handles.get(version)
        if handle is None:  # pragma: no cover - registry/handle invariant
            self.registry.complete_work(version)
            raise HTTPException(
                status_code=503,
                detail={"error": "worker_missing", "message": f"worker for {version} is gone"},
            )

        payload: Dict[str, Any] = {"features": features}
        if slow_ms:
            payload["slow_ms"] = slow_ms
        inference_id = uuid.uuid4().hex

        # Shield the worker call from this request's timeout/cancellation.  The
        # in-flight counter must only clear when the model process is actually
        # done; a 504 or disconnected client must not fake completion.
        task = asyncio.create_task(self._proxy_inference(version, handle, payload))
        self._background.add(task)
        task.add_done_callback(self._background.discard)

        started = time.perf_counter()
        try:
            result = await asyncio.wait_for(asyncio.shield(task), timeout_ms / 1000.0)
        except asyncio.TimeoutError:
            elapsed = (time.perf_counter() - started) * 1000.0
            logger.warning(
                "inference on %s exceeded %dms (request %s); worker continues",
                version,
                timeout_ms,
                inference_id,
            )
            raise HTTPException(
                status_code=504,
                detail={
                    "error": "inference_timeout",
                    "version": version,
                    "calibration_revision": revision,
                    "timeout_ms": timeout_ms,
                    "elapsed_ms": round(elapsed, 3),
                    "note": (
                        "The client stopped waiting, but the model process continues "
                        "the computation; its slot is not released until that work "
                        "actually finishes."
                    ),
                },
            )
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=502,
                detail={
                    "error": "worker_unreachable",
                    "version": version,
                    "message": str(exc),
                },
            ) from exc
        except RuntimeError as exc:
            raise HTTPException(
                status_code=502,
                detail={"error": "worker_error", "version": version, "message": str(exc)},
            ) from exc

        elapsed = (time.perf_counter() - started) * 1000.0
        return InferResponse(
            version=version,
            label=result["label"],
            score=float(result["score"]),
            inference_id=inference_id,
            elapsed_ms=round(elapsed, 3),
            device_id=device_id,
            calibration_revision=revision,
            calibrated_features=calibrated_features,
        )

    async def _proxy_inference(
        self, version: str, handle: WorkerHandle, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            assert self.client is not None
            resp = await self.client.post(handle.inference_url(), json=payload)
            if resp.status_code != 200:
                raise RuntimeError(
                    f"worker {version} returned HTTP {resp.status_code}: {resp.text[:300]}"
                )
            return resp.json()
        finally:
            # Runs once, when the worker call actually settles.  This is the
            # only place a request leaves the drain gate.
            self.registry.complete_work(version)

    # -- status ----------------------------------------------------------
    def status(self) -> StatusView:
        snapshot = self.registry.snapshot()
        slots: List[SlotView] = []
        total_in_flight = 0
        total_accepted = 0
        total_completed = 0
        active = snapshot["active_version"]
        active_info = snapshot["versions"].get(active) if active else None
        active_fp = active_info.get("fingerprint") if active_info else None
        device_revisions: Dict[str, Dict[str, Any]] = {}
        for device_id, records in snapshot["calibrations"].items():
            if not records:
                continue
            latest = records[-1]
            device_revisions[device_id] = {
                "revision": int(latest["revision"]),
                "model_version": latest["model_version"],
                "published_at": float(latest["published_at"]),
                "applicable": (
                    latest["model_version"] == active
                    and latest["model_fingerprint"] == active_fp
                ),
                "revision_count": len(records),
            }
        for slot in snapshot["slots"]:
            handle = self.handles.get(slot["version"]) if slot["version"] else None
            alive = handle.alive() if handle is not None else None
            total_in_flight += int(slot["in_flight"])
            total_accepted += int(slot["accepted"])
            total_completed += int(slot["completed"])
            slots.append(
                SlotView(
                    slot_id=slot["slot_id"],
                    version=slot["version"],
                    process_state=slot["process_state"],
                    pid=slot["pid"],
                    in_flight=int(slot["in_flight"]),
                    accepted=int(slot["accepted"]),
                    completed=int(slot["completed"]),
                    alive=alive,
                )
            )
        return StatusView(
            active_version=active,
            version=active,
            total_in_flight=total_in_flight,
            total_accepted=total_accepted,
            total_completed=total_completed,
            slots_in_use=snapshot["slots_in_use"],
            capacity=snapshot["capacity"],
            slots=slots,
            versions=sorted(
                snapshot["versions"].values(), key=lambda v: v.get("created_at", 0)
            ),
            calibrated_devices=sorted(device_revisions),
            device_calibration_revisions=device_revisions,
        )


def _new_manager() -> Any:
    """Create the registry manager, preferring ``fork`` on POSIX.

    The manager server is started once during startup, before the API handles
    requests; using ``fork`` avoids re-importing the uvicorn entry point in a
    spawned child.  On platforms without ``fork`` the default context is used.
    """
    try:
        ctx = multiprocessing.get_context("fork")
    except ValueError:  # pragma: no cover - Windows / spawn-only platforms
        ctx = multiprocessing.get_context()
    return ctx.Manager()


service = InferenceService()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    await service.startup()
    try:
        yield
    finally:
        await service.shutdown()


app = FastAPI(
    title="Edge QC Classifier Inference Service",
    version="1.1.0",
    description=(
        "Linear classifier inference with live model replacement on an edge QC "
        "machine that can host at most **two model processes**.\n\n"
        "* `POST /admin/models` uploads `weights`, `bias` and `labels`; the new "
        "version is loaded in a free process, smoke-tested, then promoted. The "
        "old version keeps serving during load and keeps serving if the new one "
        "fails.\n"
        "* `POST /admin/calibrations` trials a per-device channel permutation / "
        "gain / offset against the **currently active** version using reference "
        "samples with expected labels. A new revision is published only if every "
        "sample is classified correctly; otherwise nothing changes. A calibration "
        "is bound to one device, one model version and that version's exact "
        "parameters.\n"
        "* `POST /infer` returns the top-scoring label and the version. Requests "
        "are pinned to the version active when accepted. With `device_id` the "
        "request is also pinned to the calibration revision that matched the "
        "active version at admission; without it, the request behaves exactly as "
        "before. `timeout_ms` (or `timeout_seconds`) bounds the *client wait "
        "only*: on expiry you get 504 while the worker keeps working and keeps "
        "its slot.\n"
        "* `GET /admin/calibrations` lists calibration revisions per device and "
        "whether each still applies to the active version.\n"
        "* `GET /admin/status` shows the active version, in-flight counts and each "
        "process slot's state (`loading` / `active` / `draining` / `idle`).\n\n"
        "**Demos**\n"
        "- Calibrate a device: `POST /admin/calibrations` with raw `samples` and "
        "their `expected_label`s; then `POST /infer` with raw features and the "
        "same `device_id`. Swap the model and the old revision stops applying: "
        "device-tagged inference returns 409 `calibration_required` until the new "
        "version is calibrated.\n"
        "- Slow inference: `POST /infer` with `slow_ms: 3000`, `timeout_ms: 200`, "
        "then watch `/admin/status` and try uploading another version: while the "
        "drained version still works, only one slot is free.\n"
        "- Load failure: upload with `simulate_load_failure: true` (or malformed "
        "weights). You get 502 and the old version remains active.\n"
        "- Capacity: trigger a slow in-flight request, swap once, then try a third "
        "upload -> 429 until the slow work finishes and the slot is released."
    ),
    lifespan=lifespan,
)


@app.get("/healthz", tags=["ops"], summary="Liveness probe")
def healthz() -> Dict[str, Any]:
    active = service.registry.active_version()
    return {"ok": True, "active_version": active}


@app.post(
    "/infer",
    response_model=InferResponse,
    responses={
        504: {
            "model": TimeoutResponse,
            "description": "Client timeout; the worker keeps computing and holds its slot.",
        },
        409: {
            "description": (
                "device_id supplied but the active version has no matching "
                "calibration (calibration_required)"
            )
        },
        503: {"description": "No active model"},
        502: {"description": "Worker unreachable or errored"},
    },
    tags=["inference"],
    summary="Classify a feature vector (optionally via a device calibration)",
)
async def infer(req: InferRequest) -> InferResponse:
    if req.timeout_ms is not None:
        timeout_ms = req.timeout_ms
    elif req.timeout_seconds is not None:
        timeout_ms = max(1, int(round(req.timeout_seconds * 1000)))
    else:
        timeout_ms = DEFAULT_TIMEOUT_MS
    return await service.infer(
        req.features, timeout_ms, req.slow_ms, device_id=req.device_id
    )


@app.post(
    "/admin/calibrations",
    response_model=CalibrationPublishedView,
    status_code=201,
    responses={
        409: {
            "description": (
                "Calibration names a version that is no longer active, or the model "
                "changed while the trial ran (nothing published)"
            )
        },
        422: {
            "description": (
                "Invalid permutation/gain/offset, wrong sample width, or at least "
                "one reference sample misclassified (nothing published)"
            )
        },
        502: {"description": "The model process failed while trialling samples"},
        503: {"description": "No active model"},
    },
    tags=["calibration"],
    summary="Trial and publish a per-device channel calibration",
)
async def calibrate(payload: CalibrationRequest) -> CalibrationPublishedView:
    if not VERSION_RE.match(payload.model_version):
        raise HTTPException(
            status_code=422,
            detail="model_version must match ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$",
        )
    return await service.calibrate(payload)


@app.get(
    "/admin/calibrations",
    response_model=List[CalibrationRecordView],
    tags=["calibration"],
    summary="Calibration revision history per device",
)
def list_calibrations(
    device_id: Optional[str] = Query(
        None, description="Only return revisions for this device"
    ),
) -> List[CalibrationRecordView]:
    return service.calibration_records(device_id)


@app.post(
    "/admin/models",
    response_model=SwapResult,
    status_code=201,
    responses={
        409: {"description": "Version exists or swap conflict / stale expectation"},
        429: {"description": "Both model process slots are still occupied"},
        502: {"description": "Candidate load or smoke test failed; old version kept"},
        422: {"description": "Artifact is structurally invalid"},
    },
    tags=["admin"],
    summary="Upload a classifier and hot-swap it in",
)
async def upload_model(payload: ModelUpload) -> SwapResult:
    if not VERSION_RE.match(payload.version):
        raise HTTPException(
            status_code=422,
            detail="version must match ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$",
        )
    try:
        validate_artifact(payload.model_dump())
    except ModelValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={"error": "invalid_artifact", "message": str(exc)},
        ) from exc

    target = STAGING_DIR / f"{payload.version}-{uuid.uuid4().hex[:12]}.json"
    try:
        save_artifact(str(target), payload.version, payload.model_dump())
    except (OSError, ModelValidationError) as exc:
        raise HTTPException(
            status_code=500,
            detail={"error": "artifact_write_failed", "message": str(exc)},
        ) from exc

    try:
        result = await service.swap_in_path(
            version=payload.version,
            source_path=str(target),
            expected_active=payload.expected_active_version,
            test_features=payload.test_features,
            slow_load_ms=payload.slow_load_ms,
            simulate_load_failure=payload.simulate_load_failure,
        )
    finally:
        # The worker process holds its own loaded copy in memory; the staging
        # file is only needed for the load.  Published artifacts live under
        # ``models/<version>.json`` and are never overwritten by an upload.
        with contextlib.suppress(FileNotFoundError):
            target.unlink()
    return result


@app.get("/admin/versions", tags=["admin"], summary="Version registry")
def versions() -> Dict[str, Any]:
    snapshot = service.registry.snapshot()
    return {
        "active_version": snapshot["active_version"],
        "capacity": snapshot["capacity"],
        "slots_in_use": snapshot["slots_in_use"],
        "versions": sorted(
            snapshot["versions"].values(), key=lambda v: v.get("created_at", 0)
        ),
        "calibrations": {
            device_id: [dict(rec) for rec in records]
            for device_id, records in snapshot["calibrations"].items()
        },
    }


@app.get(
    "/admin/status",
    response_model=StatusView,
    tags=["admin"],
    summary="Active version, in-flight counts and process states",
)
def status() -> StatusView:
    return service.status()
