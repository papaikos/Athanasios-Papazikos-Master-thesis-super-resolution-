import json
from unittest.mock import patch
import torch
import torch.nn.functional as F
from src.swinir_train import preflight


class SmallRGB(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1)
    def forward(self, x):
        return F.interpolate(self.conv(x), scale_factor=4, mode='bilinear', align_corners=False)


def test_fp32_preflight_runs_without_cuda_autocast_and_checks_gradients(tmp_path):
    lr = torch.rand(1, 3, 8, 8)
    hr = torch.rand(1, 3, 32, 32)
    hm, lm = torch.ones(1, 1, 32, 32), torch.ones(1, 1, 8, 8)
    net = SmallRGB()
    # Data/device boundaries are replaced; real forward, loss and gradients are exercised.
    with patch('src.swinir_train.load_pair', return_value=(lr, hr, hm, lm)), patch('torch.cuda.get_device_name', return_value='test CPU'):
        preflight(net, None, tmp_path, torch.device('cpu'), amp=False)
    receipt = json.loads((tmp_path/'preflight.json').read_text())
    assert receipt['passed']
    assert receipt['precision'] == 'fp32'
    assert receipt['comparison_mean_abs'] == 0
    assert all(p.grad is None for p in net.parameters())
