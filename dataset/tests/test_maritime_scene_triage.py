import pytest
from scripts.maritime_scene_triage import calibrate_threshold, source_holdout_evaluation

def test_calibration_target_is_training_only():
 assert calibrate_threshold([{'score':.1,'review_reject':True},{'score':.2,'review_reject':True},{'score':.9,'review_reject':False}],1)==.1

def test_source_holdout_does_not_fit_on_heldout_reject():
 rows=[{'source_kind':'a','score':.2,'review_reject':True},{'source_kind':'b','score':.3,'review_reject':True},{'source_kind':'c','score':-.5,'review_reject':True},{'source_kind':'c','score':.4,'review_reject':False}]
 e=source_holdout_evaluation(rows)['c'];assert e['threshold']==.2;assert e['heldout_reject_missed']==1;assert e['heldout_pending_flagged']==1;assert e['training_count']==2

def test_no_training_rejects_cannot_claim_calibration():
 with pytest.raises(ValueError,match='reject'):calibrate_threshold([{'score':.5,'review_reject':False}])
