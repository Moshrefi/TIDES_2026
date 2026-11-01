"""CPU-only regression tests; run with unittest even without PyTorch/CUDA."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('audit_scene', ROOT/'scripts'/'audit_prepared_scene.py')
audit_scene = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_scene)


class PreparedAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.work = Path(self.tmp.name)
        sparse = self.work/'scene'/'sparse'/'0'
        sparse.mkdir(parents=True)
        images = self.work/'scene'/'images'
        images.mkdir()
        # Irregular seconds must map to index/N, not linear normalization of seconds.
        self.meta = dict(frame_times_s=[0, .01, .029], scene_times=[0, 1/3, 2/3],
            t_range=[0, .029], frame_interval=.0145, scale=1.,
            poses=dict(times_s=[0,.01,.029], R_wc=[np.eye(3).tolist()]*3,
                       t_wc=[[0,0,0],[-1,0,0],[-2,0,0]]),
            K_train=dict(width=5,height=3,fx=10,fy=11,cx=2,cy=1),
            T_render_rgb=np.eye(4).tolist())
        self.write_meta()
        (sparse/'cameras.txt').write_text('1 PINHOLE 5 3 10 11 2 1\n')
        (sparse/'images.txt').write_text(''.join(f'{i+1} 1 0 0 0 {-i} 0 0 1 {i:06d}.png\n\n' for i in range(3)))
        (sparse/'points3D.ply').write_bytes(b'fixture; no image or point decoding in this algebra audit')
        for i in range(3):
            (images/f'{i:06d}.png').touch()

    def tearDown(self):
        self.tmp.cleanup()

    def write_meta(self):
        (self.work/'tides_meta.json').write_text(json.dumps(self.meta))

    def test_irregular_time_index_over_n_is_valid(self):
        result = audit_scene.audit(self.work)
        self.assertEqual(result['frames'], 3)
        self.assertAlmostEqual(result['scene_time_last'], 2/3)

    def test_normalized_seconds_is_rejected(self):
        self.meta['scene_times'] = [0, .01/.029, 1]
        self.write_meta()
        with self.assertRaisesRegex(ValueError, 'index/N'):
            audit_scene.audit(self.work)

    def test_duplicate_seconds_is_rejected(self):
        self.meta['frame_times_s'][1] = 0
        self.write_meta()
        with self.assertRaisesRegex(ValueError, 'strictly increasing'):
            audit_scene.audit(self.work)

    def test_pose_direction_mismatch_is_rejected(self):
        self.meta['poses']['t_wc'][2] = [2, 0, 0]
        self.write_meta()
        with self.assertRaisesRegex(ValueError, 'pose differs'):
            audit_scene.audit(self.work)

    def test_stale_images_and_binary_models_are_rejected(self):
        stray = self.work/'scene'/'images'/'000003.png'
        stray.touch()
        with self.assertRaisesRegex(ValueError, 'stale training images'):
            audit_scene.audit(self.work)
        stray.unlink()
        (self.work/'scene'/'sparse'/'0'/'images.bin').touch()
        with self.assertRaisesRegex(ValueError, 'binary COLMAP'):
            audit_scene.audit(self.work)

    def test_nonrigid_extrinsic_is_rejected(self):
        self.meta['T_render_rgb'][0][0] = 2
        self.write_meta()
        with self.assertRaisesRegex(ValueError, 'Nonrigid'):
            audit_scene.audit(self.work)


if __name__ == '__main__':
    unittest.main()
