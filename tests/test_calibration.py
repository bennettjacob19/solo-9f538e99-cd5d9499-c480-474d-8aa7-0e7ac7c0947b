"""Unit tests for calibration validation and the raw -> model feature transform."""

from __future__ import annotations

import pytest

from app.calibration import CalibrationInvalid, CalibrationTransform


def test_identity_defaults():
    transform = CalibrationTransform.build(3)
    assert transform.permutation == (0, 1, 2)
    assert transform.gain == (1.0, 1.0, 1.0)
    assert transform.offset == (0.0, 0.0, 0.0)
    assert transform.apply([1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0]


def test_permutation_gain_offset_order():
    # calibrated[0] = raw[2]*10 + 1, calibrated[1] = raw[0]*2 - 1,
    # calibrated[2] = raw[1]*1 + 0
    transform = CalibrationTransform.build(
        3, channel_permutation=[2, 0, 1], gain=[10, 2, 1], offset=[1, -1, 0]
    )
    assert transform.apply([5.0, 7.0, 3.0]) == [31.0, 9.0, 7.0]


def test_scalar_gain_and_offset_broadcast():
    transform = CalibrationTransform.build(2, gain=2.0, offset=-0.5)
    assert transform.apply([1.0, 2.0]) == [1.5, 3.5]
    assert transform.gain == (2.0, 2.0)


@pytest.mark.parametrize(
    "permutation,fragment",
    [
        ([0, 1], "must have 3 entries"),
        ([0, 0, 1], "permutation of 0..2"),
        ([0, 1, 3], "permutation of 0..2"),
        ([0, 1.0, 2], "must be integers"),
        ([0, True, 2], "must be integers"),
        ("abc", "must be a list"),
    ],
)
def test_bad_permutations_rejected(permutation, fragment):
    with pytest.raises(CalibrationInvalid) as exc:
        CalibrationTransform.build(3, channel_permutation=permutation)
    assert fragment in str(exc.value)


@pytest.mark.parametrize(
    "gain,offset,fragment",
    [
        ([1.0, 2.0], None, "gain must have 3 values"),
        (None, [0.0] * 4, "offset must have 3 values"),
        (float("nan"), None, "finite"),
        ([1.0, float("inf"), 1.0], None, "finite"),
        ("x", None, "must be a number"),
        ([1.0, True, 1.0], None, "numbers"),
    ],
)
def test_bad_gain_or_offset_rejected(gain, offset, fragment):
    with pytest.raises(CalibrationInvalid) as exc:
        CalibrationTransform.build(3, gain=gain, offset=offset)
    assert fragment in str(exc.value)


def test_apply_rejects_wrong_width_and_nan():
    transform = CalibrationTransform.build(2)
    with pytest.raises(CalibrationInvalid, match="expected 2"):
        transform.apply([1.0])
    with pytest.raises(CalibrationInvalid, match="finite"):
        transform.apply([1.0, float("nan")])


def test_document_roundtrip():
    transform = CalibrationTransform.build(2, [1, 0], [3.0, 4.0], [0.5, -0.5])
    doc = transform.to_doc()
    restored = CalibrationTransform.from_doc(doc)
    assert restored == transform
    assert doc["channel_permutation"] == [1, 0]
    assert doc["n_features"] == 2


def test_zero_features_rejected():
    with pytest.raises(CalibrationInvalid, match="no input features"):
        CalibrationTransform.build(0)
