"""Per-device channel calibration: validation and feature transformation.

A calibration revision belongs to exactly one ``(device_id, model_version)``
pair.  It describes how raw values read from a device's sensor channels are
turned into the feature vector the model version expects:

    calibrated[i] = raw[channel_permutation[i]] * gain[i] + offset[i]

``channel_permutation`` must therefore be a permutation of ``0..n-1``.  The
gain and offset may be given per channel, or as one scalar applied to every
channel.  Omitted fields default to the identity transform (identity order,
gain 1, offset 0).

The transform is deliberately immutable: an inference admitted with a
revision keeps using that revision's numbers even if a newer revision is
published while it is still running.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union


class CalibrationInvalid(ValueError):
    """The submitted permutation / gain / offset is not a usable calibration."""


NumberOrList = Optional[Union[float, List[float]]]


def _as_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalibrationInvalid(f"{name} must contain only numbers")
    number = float(value)
    if not math.isfinite(number):
        raise CalibrationInvalid(f"{name} contains NaN or infinite values")
    return number


def _normalise_affine(
    value: NumberOrList, n_features: int, name: str, default: float
) -> Tuple[float, ...]:
    if value is None:
        values: Iterable[Any] = [default] * n_features
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        values = [value] * n_features
    elif isinstance(value, (list, tuple)):
        if len(value) != n_features:
            raise CalibrationInvalid(
                f"{name} must have {n_features} values (one per channel), "
                f"got {len(value)}"
            )
        values = value
    else:
        raise CalibrationInvalid(
            f"{name} must be a number or a list of {n_features} numbers"
        )
    return tuple(_as_float(v, name) for v in values)


def _normalise_permutation(
    value: Optional[Sequence[int]], n_features: int
) -> Tuple[int, ...]:
    if value is None:
        return tuple(range(n_features))
    if not isinstance(value, (list, tuple)):
        raise CalibrationInvalid(
            "channel_permutation must be a list of channel indices"
        )
    if len(value) != n_features:
        raise CalibrationInvalid(
            f"channel_permutation must have {n_features} entries, got {len(value)}"
        )
    indices: List[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise CalibrationInvalid(
                "channel_permutation entries must be integers"
            )
        indices.append(item)
    if sorted(indices) != list(range(n_features)):
        raise CalibrationInvalid(
            f"channel_permutation must be a permutation of 0..{n_features - 1}"
        )
    return tuple(indices)


@dataclass(frozen=True)
class CalibrationTransform:
    """An immutable raw-feature -> model-feature mapping."""

    permutation: Tuple[int, ...]
    gain: Tuple[float, ...]
    offset: Tuple[float, ...]

    @classmethod
    def build(
        cls,
        n_features: int,
        channel_permutation: Optional[Sequence[int]] = None,
        gain: NumberOrList = None,
        offset: NumberOrList = None,
    ) -> "CalibrationTransform":
        if n_features < 1:
            raise CalibrationInvalid("model version has no input features")
        return cls(
            permutation=_normalise_permutation(channel_permutation, n_features),
            gain=_normalise_affine(gain, n_features, "gain", 1.0),
            offset=_normalise_affine(offset, n_features, "offset", 0.0),
        )

    @property
    def n_features(self) -> int:
        return len(self.permutation)

    def apply(self, features: Sequence[float]) -> List[float]:
        if len(features) != self.n_features:
            raise CalibrationInvalid(
                f"expected {self.n_features} raw features, got {len(features)}"
            )
        calibrated: List[float] = []
        for i in range(self.n_features):
            raw = _as_float(features[self.permutation[i]], "features")
            calibrated.append(raw * self.gain[i] + self.offset[i])
        return calibrated

    def to_doc(self) -> Dict[str, Any]:
        return {
            "channel_permutation": list(self.permutation),
            "gain": list(self.gain),
            "offset": list(self.offset),
            "n_features": self.n_features,
        }

    @classmethod
    def from_doc(cls, doc: Dict[str, Any]) -> "CalibrationTransform":
        n_features = int(doc.get("n_features", len(doc.get("channel_permutation", []))))
        return cls.build(
            n_features,
            doc.get("channel_permutation"),
            doc.get("gain"),
            doc.get("offset"),
        )

    @classmethod
    def identity(cls, n_features: int) -> "CalibrationTransform":
        return cls.build(n_features)
