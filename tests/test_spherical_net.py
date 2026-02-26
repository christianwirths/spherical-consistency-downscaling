import unittest

import numpy as np
import torch

from src.consistency_model.spherical_net import (
    DirectNeighConv,
    SphericalUNetWrapper,
    build_equiangular_neighbours,
)


class TestSphericalNet(unittest.TestCase):
    """Basic tests for the spherical graph network components."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        H, W = 8, 16
        cls.H, cls.W = H, W
        model = SphericalUNetWrapper(
            in_channels=2, out_channels=2,
            image_height=H, image_width=W,
            channel_list=(8, 8), spherical_depth=2, time_emb_dim=16,
        )
        model.eval()
        with torch.no_grad():
            cls.wrapper_out = model(torch.randn(1, 2, H, W), torch.tensor([1.0]))

    def test_neighbour_indices_shape_and_range(self):
        H, W = 4, 8
        neigh = build_equiangular_neighbours(H, W)
        V = H * W
        self.assertEqual(neigh.shape, (V, 9), "incorrect neighbour array shape")
        self.assertTrue(np.all(neigh >= 0) and np.all(neigh < V),
                        "neighbour index out of range")

    def test_direct_neigh_conv_output_shape(self):
        H, W = 4, 8
        neigh = torch.from_numpy(build_equiangular_neighbours(H, W))
        conv = DirectNeighConv(in_ch=3, out_ch=6)
        x = torch.randn(2, H * W, 3)
        out = conv(neigh, x)
        self.assertEqual(out.shape, (2, H * W, 6), "incorrect output shape")
        self.assertFalse(torch.isnan(out).any(), "output contains NaN")

    def test_wrapper_output_shape(self):
        self.assertEqual(self.wrapper_out.sample.shape, (1, 2, self.H, self.W),
                         "incorrect output shape")

    def test_wrapper_output_no_nan(self):
        self.assertFalse(torch.isnan(self.wrapper_out.sample).any(),
                         "output contains NaN")


if __name__ == "__main__":
    unittest.main()
