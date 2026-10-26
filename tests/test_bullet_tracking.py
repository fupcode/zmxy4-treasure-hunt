"""子弹跨帧匹配、漏检保留和距离排序。"""

from types import SimpleNamespace
import unittest

import numpy as np

from zaomeng_hanghai.core.entities import Bullet, Bullets


class BulletTrackingTests(unittest.TestCase):
    def test_bullet_update_sorts_matched_missing_and_new_bullets(self):
        env = SimpleNamespace(bullets=Bullets())
        env.player = SimpleNamespace(x=0, y=0, xn=0, yn=0)
        env.bullets.extend([Bullet((0.8, 0, 2)), Bullet((0.6, 0, 2))])
        env.bullets.update([(0.79, 0, 2), (0.1, 0, 1)], env.player)
        np.testing.assert_allclose([b.xn for b in env.bullets], [0.1, 0.6, 0.79])
        self.assertEqual([b.disappear_count for b in env.bullets], [0, 1, 0])
        self.assertEqual(env.bullets.new_bullets, [env.bullets[0]])


if __name__ == "__main__":
    unittest.main()
