import numpy as np
import pytest
from scripts.maritime_cloud_segmentation_pilot import model_input

def test_official_rgb_input_is_256_chw_and_global_minmax():
 a=np.zeros((512,512,3),dtype=np.uint8);a[:,:,0]=100;a[:,:,1]=150;a[:,:,2]=200
 x=model_input(a);assert x.shape==(3,256,256);assert x.dtype==np.float32;assert np.all(x[0]==0) and np.all(x[1]==.5) and np.all(x[2]==1)

def test_constant_image_deferred_instead_of_division_by_zero():
 with pytest.raises(ValueError,match='constant'):model_input(np.full((512,512,3),100,dtype=np.uint8))

def test_other_input_representation_rejected():
 with pytest.raises(ValueError,match='uint8'):model_input(np.zeros((512,512,3),dtype=np.float32))
