from nnutils.determinism import seed_all
import torch

from nnutils.models.resnet import ResNet, Bottleneck
from nnutils.models.resnet_generalized import ResNet as ResNetGeneralized, Bottleneck as BottleneckGeneralized

def test_resnet_and_resnet_generalized_match():
    seed_all(37, cuda=False)
    orig_resnet = ResNet(Bottleneck, [3, 4, 6, 3], skip_after_nonlin=True)
    seed_all(37, cuda=False)
    generalized_resnet = ResNetGeneralized(BottleneckGeneralized, [3, 4, 6, 3], skip_after_nonlin=True)
    # Check if the number of parameters match
    assert sum(p.numel() for p in orig_resnet.parameters()) == sum(p.numel() for p in generalized_resnet.parameters())
    # Dummy input
    x = torch.randn(1, 3, 224, 224)
    # Check if the outputs match
    orig_y = orig_resnet(x)
    gen_y = generalized_resnet(x)
    assert orig_y.shape == gen_y.shape
    assert torch.allclose(orig_y, gen_y, atol=1e-5)