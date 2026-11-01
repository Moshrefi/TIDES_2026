"""Regression coverage for reverse AD with genuinely frozen 4DGS appearance."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tides.camera import Intrinsics, PoseTrajectory
from tides.projection import CONTINUOUS_KEYS
from tides.renderer import TangentRenderer
from tides.scene.base import ToyScene


class FrozenAppearanceScene(ToyScene):
    def params(self, times):
        p = super().params(times)
        # Unlike ToyScene's +0*t, these have no autograd connection at all,
        # matching upstream 4DGS with no_do=True and SH degree zero.
        p['opacity_raw'] = self.opacity_raw.reshape(-1)
        p['shs'] = self.shs
        return p


class GridSampleScene(FrozenAppearanceScene):
    def params(self, times):
        p = super().params(times)
        plane = times.new_tensor([[[[-1., 0., 1.], [-1., 0., 1.]]]])
        grid = torch.stack([times*.2, torch.zeros_like(times)], -1).reshape(1, 1, -1, 2)
        shift = F.grid_sample(plane, grid, align_corners=True).reshape(-1)
        p['means'] = self.means + torch.stack([shift, shift*0, shift*0], -1)
        return p


def renderer(scene):
    return TangentRenderer(scene, PoseTrajectory.static(np.eye(3), np.zeros(3)),
        Intrinsics(40., 40., 23.5, 17.5, 48, 36), mode='reverse')


@pytest.mark.parametrize('scene_type', [FrozenAppearanceScene, GridSampleScene])
def test_reverse_constants_keep_dynamic_geometry_derivatives(scene_type):
    r = renderer(scene_type.random(n=8, seed=17))
    t, h = .37, .002
    projected, tangent = r.primitives(t)
    torch.testing.assert_close(tangent['opacity'], torch.zeros_like(projected['opacity']), rtol=0, atol=0)
    torch.testing.assert_close(tangent['color'], torch.zeros_like(projected['color']), rtol=0, atol=0)
    assert tangent['means2d'].abs().max() > .01  # Do not replace all derivatives with zeros.
    plus, _ = r.primitives(t+h, with_tangent=False)
    minus, _ = r.primitives(t-h, with_tangent=False)
    for key in CONTINUOUS_KEYS:
        assert torch.isfinite(tangent[key]).all()
        assert not tangent[key].requires_grad
        torch.testing.assert_close(tangent[key], (plus[key]-minus[key])/(2*h), rtol=.03, atol=.003)


def test_fully_disconnected_projection_is_reported(monkeypatch):
    r = renderer(FrozenAppearanceScene.random(n=8, seed=17))
    disconnected, _ = r.primitives(.37, with_tangent=False)
    monkeypatch.setattr(r, '_project', lambda *args, **kwargs: disconnected)
    with pytest.warns(RuntimeWarning, match='accidentally detached'):
        _, tangent = r.primitives(.37)
    assert all(torch.count_nonzero(value) == 0 for value in tangent.values())


def test_reverse_does_not_suppress_autograd_errors(monkeypatch):
    r = renderer(FrozenAppearanceScene.random(n=8, seed=17))
    def failed_grad(*args, **kwargs):
        raise RuntimeError('unexpected backward failure')
    monkeypatch.setattr(torch.autograd, 'grad', failed_grad)
    with pytest.raises(RuntimeError, match='unexpected backward failure'):
        r.primitives(.37)
