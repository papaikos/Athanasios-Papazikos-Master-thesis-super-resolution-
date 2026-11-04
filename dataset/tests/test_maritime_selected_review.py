import io

import numpy as np
import pytest

from scripts.maritime_canonical_screening import canonical_arrays
from scripts.maritime_selected_review import bound_review_rgb


def witness():
    context=np.random.default_rng(12).random((3,544,544),dtype=np.float32)
    raw=io.BytesIO()
    np.savez_compressed(raw,hr=context[:,16:528,16:528],hr_context=context)
    data=raw.getvalue();_,rgb,hashes=canonical_arrays(data)
    return data,rgb,dict(sample_id='synthetic-selected',**hashes)


def test_actual_review_pixels_match_both_selected_hashes():
    raw,rgb,row=witness()
    assert np.array_equal(bound_review_rgb(raw,row),rgb)


@pytest.mark.parametrize('field',['hr_float32_sha256','hr_uint8_chw_sha256'])
def test_wrong_selected_image_witness_is_rejected(field):
    raw,_,row=witness();row[field]='0'*64
    with pytest.raises(ValueError,match='Selected HR hash mismatch'):
        bound_review_rgb(raw,row)
