import importlib
import pytest

m = importlib.import_module("reports.maritime-expansion-20261007.earthview-adapter.bounded_population_quality_v5_on_gce")

def test_frozen_pin_union_preserves_prior_and_disjoint_closures():
    prior = [{"sample_id":"old", "candidate_pin":{"uri":"gs://b/old","sha256":"a"*64}}]
    novel = [{"sample_id":"new", "candidate_pin":{"uri":"gs://b/new","sha256":"b"*64}}]
    assert m.frozen_source_pins(prior,[novel]) == [prior[0]["candidate_pin"],novel[0]["candidate_pin"]]
    with pytest.raises(ValueError): m.frozen_source_pins(prior,[prior])
    with pytest.raises(ValueError): m.frozen_source_pins(prior,[[{"sample_id":"new","candidate_pin":{"uri":"gs://b/new"}}]])

def test_reused_policy_requires_same_audit_and_all_locks():
    audit={"schema":"root-earthview-development-actual-hr-visual-audit-v1","quality_accepted_count":0,"final_eligible":False,"previous_excluded_sample_ids":[str(i) for i in range(31)],"newly_excluded_sample_ids":[str(i) for i in range(31,75)],"explicit_excluded_sample_ids":[str(i) for i in range(75)],"marine_content_unresolved_sample_ids":[str(i) for i in range(75,100)]}
    pin={"sha256":"a"*64};config={"audit_pin":pin,"explicit_observed_exclusions":audit["explicit_excluded_sample_ids"],"marine_content_unresolved_ids":audit["marine_content_unresolved_sample_ids"]}
    excluded,held=m.reused_policy(audit,pin,config)
    assert len(excluded)==75 and len(held)==25
    with pytest.raises(ValueError):m.reused_policy(audit,{"sha256":"b"*64},config)
    with pytest.raises(ValueError):m.reused_policy(audit,pin,{**config,"explicit_observed_exclusions":[]})
