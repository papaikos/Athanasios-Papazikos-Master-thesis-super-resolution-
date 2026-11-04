"""Which x4 experiment this process runs: set MARITIME_DATASET to d1, d2 or d3.

D1, D2 and D3 share the same code, model and schedule. Only the dataset folder
and the LR degradation recipe differ.
"""
import os

RECIPES = {'d1': 'd1-bicubic-aa-v1',
           'd2': 'd2-gaussian-area-context-v1',
           'd3': 'd3-variable-psf-noise-v1'}

KEY = os.environ.get('MARITIME_DATASET', 'd2').strip().lower()
if KEY not in RECIPES:
    raise ValueError('MARITIME_DATASET must be d1, d2 or d3')
NAME = KEY.upper()
RECIPE = RECIPES[KEY]
