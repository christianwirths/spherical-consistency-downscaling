import unittest
import math
import torch

from src.configuration import Config
from src.sde_model.model import SDEModel
from src.sde_model.loss import VELoss


class TestSDEModel(unittest.TestCase):
    """Runs some basic tests for the SDE model. """

    def test_output_shape(self):

        config = Config()
        config.in_channels = 1
        config.out_channels = 1
        model = SDEModel(config)

        x = torch.rand(1,1,32,32)
        t = torch.rand(1)
        y = model.net(x,t)

        self.assertEqual(y.shape, (1,1,32,32), "incorrect output shape")


    def test_loss_is_not_nan(self):

        config = Config()
        config.in_channels = 1
        config.out_channels = 1
        model = SDEModel(config)
        x = torch.rand(1,1,32,32)

        loss_fn = VELoss(model.marginal_prob_std)
        loss = loss_fn(model.net, x)

        self.assertFalse(math.isnan(loss), "loss is NaN")


if __name__ == "__main__":
    unittest.main()