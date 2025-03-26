from nnutils.determinism import seed_all
import torch

from nnutils.models.resnet import ResNet, Bottleneck
from nnutils.models.resnet_generalized import ResNet as ResNetGeneralized, Bottleneck as BottleneckGeneralized

def test_resnet_and_resnet_generalized_match():
    seed_all(37, cuda=False)
    orig_resnet = ResNet(Bottleneck, [3, 4, 6, 3], skip_after_nonlin=True)
    seed_all(37, cuda=False)
    generalized_resnet = ResNetGeneralized(BottleneckGeneralized, [3, 4, 6, 3])
    # Check if the number of parameters match
    # assert len(orig_resnet.layer1) == len(generalized_resnet.layer1.layers)
    # for i, l in enumerate(orig_resnet.layer1):
    #     assert isinstance(l, Bottleneck)
    #     assert isinstance(generalized_resnet.layer1.layers[i], BottleneckGeneralized)
    #     # Check if the parameters match
    #     for name, param in l.named_parameters():
    #         gen_params = dict(generalized_resnet.layer1.layers[i].named_parameters())
    #         assert param.shape == gen_params[name].shape, f"Parameter {name} does not match at layer {i}."
            # assert torch.allclose(param, gen_params[name])
    # assert sum(p.numel() for p in orig_resnet.layer1.parameters()) == sum(p.numel() for p in generalized_resnet.layer1.parameters())
    assert sum(p.numel() for p in orig_resnet.parameters()) == sum(p.numel() for p in generalized_resnet.parameters())
    # Dummy input
    x = torch.randn(1, 3, 224, 224)
    # Check if the outputs match
    orig_y = orig_resnet(x)
    gen_y = generalized_resnet(x)
    assert orig_y.shape == gen_y.shape
    assert torch.allclose(orig_y, gen_y, atol=1e-5)