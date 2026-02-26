import unittest
import math


class TestConsistencyModel(unittest.TestCase):
    """Tests for the consistency model parameterization (pure math, no GPU)."""

    # Defaults from Config / Consistency.__init__
    DATA_STD = 0.5
    TIME_MIN  = 0.002
    TIME_MAX  = 80.0

    def _c_skip(self, t):
        return self.DATA_STD**2 / ((t - self.TIME_MIN)**2 + self.DATA_STD**2)

    def _c_out(self, t):
        return self.DATA_STD * t / math.sqrt(t**2 + self.DATA_STD**2)

    def test_c_skip_is_one_at_t_min(self):
        """c_skip(t_min) == 1: output equals input at zero noise (boundary condition)."""
        self.assertAlmostEqual(self._c_skip(self.TIME_MIN), 1.0, places=9)

    def test_c_out_is_small_at_t_min(self):
        """c_out(t_min) << 1: network contribution is negligible at zero noise."""
        self.assertLess(self._c_out(self.TIME_MIN), 0.01)

    def test_c_skip_is_small_at_t_max(self):
        """c_skip(t_max) ≈ 0: output dominated by network at maximum noise."""
        self.assertLess(self._c_skip(self.TIME_MAX), 1e-3)


if __name__ == "__main__":
    unittest.main()
