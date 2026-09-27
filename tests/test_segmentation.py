import sys
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np

from smg.segmentation import SamSession


def test_sam_session_forwards_clicks_and_never_selects_transparency(monkeypatch):
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
        inference_mode=lambda: nullcontext(),
        autocast=lambda *args, **kwargs: nullcontext(),
        bfloat16=object(),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    rgba = np.full((4, 5, 4), 255, np.uint8)
    rgba[0, 0, 3] = 0
    session = SamSession(rgba)
    captured = {}

    class Predictor:
        def predict(self, **kwargs):
            captured.update(kwargs)
            mask = np.zeros((1, 4, 5), bool)
            mask[0, 0, 0] = True
            mask[0, 2, 3] = True
            return mask, np.array([0.9], np.float32), None

    session._predictor = Predictor()
    mask = session.predict([(3, 2), (1, 1)], [1, 0])
    assert captured["multimask_output"] is False
    assert np.array_equal(captured["point_coords"], [[3, 2], [1, 1]])
    assert np.array_equal(captured["point_labels"], [1, 0])
    assert mask.dtype == bool and mask.shape == (4, 5)
    assert mask[2, 3]
    assert not mask[0, 0]


def test_sam_session_requires_an_add_click():
    rgba = np.full((2, 2, 4), 255, np.uint8)
    session = SamSession(rgba)
    with np.testing.assert_raises_regex(ValueError, "добавляющий"):
        session.predict([(0, 0)], [0])
