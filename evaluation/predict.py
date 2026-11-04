"""Super-resolve an RGB image with one of the released SwinIR models.

Same inference as the thesis comparisons: the RGB image is divided by 255 and
the whole frame goes through the model in FP32, with no tiling, resizing or
enhancement. The x4 models (D1, D2, D3) map 10 m to a 2.5 m grid; the x10
model (D3x10) maps 10 m to a 1 m grid. The added detail is predicted, not
observed.

  python evaluation/predict.py --weights swinir_x4_D3_best_step82000.pth --scale 4 \
      --input sentinel2_rgb.png --output sr.png
"""
import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

REPO = Path(__file__).resolve().parents[1]
SOURCE = {4: REPO / 'training/x4/source', 10: REPO / 'training/x10/source'}


def load_input(path):
    """Return a float32 (3, H, W) array in [0, 1]."""
    if path.suffix.lower() == '.npy':
        array = np.load(path).astype(np.float32)
        if array.ndim != 3 or array.shape[0] != 3:
            raise ValueError('A .npy input must have shape (3, H, W) with values in [0, 1]')
        return array
    with Image.open(path) as image:
        if image.mode != 'RGB':
            raise ValueError(f'8-bit RGB image required, got mode {image.mode}')
        return np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--weights', type=Path, required=True, help='.pth file from the GitHub Release')
    parser.add_argument('--scale', type=int, choices=(4, 10), required=True, help='4 for D1/D2/D3, 10 for D3x10')
    parser.add_argument('--input', type=Path, required=True, help='8-bit RGB image, or a (3, H, W) float .npy')
    parser.add_argument('--output', type=Path, required=True, help='output PNG')
    parser.add_argument('--save-npy', action='store_true', help='also save the raw FP32 prediction')
    parser.add_argument('--device', default='cpu', help='cpu (as in the thesis) or cuda')
    args = parser.parse_args()

    source = SOURCE[args.scale]
    sys.path.insert(0, str(source))
    from research.model_factory import make_model

    model, info = make_model(source / 'vendor/network_swinir.py')
    state = torch.load(args.weights, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(args.device).eval().requires_grad_(False)

    lr = torch.from_numpy(load_input(args.input))[None].to(args.device)
    with torch.inference_mode():
        sr = model(lr)[0].float().cpu().numpy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(np.clip(sr.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)).save(args.output)
    if args.save_npy:
        np.save(args.output.with_suffix('.npy'), sr)
    print(f'{args.input.name}: {tuple(lr.shape[-2:])} -> {tuple(sr.shape[-2:])} '
          f'(x{args.scale}, {info["parameter_count"]:,} parameters) -> {args.output}')


if __name__ == '__main__':
    main()
