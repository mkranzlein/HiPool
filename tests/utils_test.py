import torch

import hipool.utils as utils


def test_loss_fun():
    """Trivial test for setting up GitHub Actions.
    
    asdf
    
    """
    output = utils.loss_fun(torch.randn((5, 3)), torch.randn((5, 3)))
    print(type(output))
    assert type(output) == torch.Tensor