from copy import deepcopy
from types import SimpleNamespace

import pytest
from rasterio.crs import CRS

from scripts import maritime_export_scheduler as scheduler


class Expression:
    def __init__(self, calls, value=None):
        self.calls = calls
        self.value = value

    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            if name == "projection":
                return Expression(self.calls, dict(crs="EPSG:32644", transform=[.8, 0, 100, 0, -.8, 3000]))
            if name == "coordinates":
                return Expression(self.calls, [[[100, 360], [2740, 360], [2740, 3000], [100, 3000], [100, 360]]])
            return self
        return call

    def getInfo(self):
        return self.value


class FakeEE:
    def __init__(self):
        self.calls = []
        self.exports = []
        self.remote = []
        self.starts = []
        self.uncertain = False
        self.next_id = 0
        self.data = SimpleNamespace(getTaskList=lambda: deepcopy(self.remote), newTaskId=self.new_id)
        self.Reducer = SimpleNamespace(min=lambda: "minimum")
        self.Geometry = SimpleNamespace(Rectangle=lambda *args, **kwargs: "native-rectangle")
        self.batch = SimpleNamespace(Export=SimpleNamespace(image=SimpleNamespace(toCloudStorage=self.export)))

    def Projection(self, crs):
        self.calls.append(('Projection', (crs,), {}))
        return SimpleNamespace(crs=crs, transform=[1,0,0,0,1,0])

    def Image(self, asset):
        self.calls.append(("Image", (asset,), {}))
        return Expression(self.calls)

    def new_id(self):
        self.next_id += 1
        return [f"request-{self.next_id}"]

    def export(self, **kwargs):
        self.exports.append(kwargs)
        task = SimpleNamespace(id=None)
        def start():
            self.starts.append(task._request_id)
            task.id = task._request_id
            if self.uncertain:
                raise OSError("connection lost after start")
            self.remote.append(dict(id=task.id, description=kwargs["description"], state="READY"))
        task.start = start
        return task


def state():
    return scheduler.initial_state({"asset_ids": ["a", "b"]}, [], "gs://bucket/new-export", "project")


def test_native_affine_and_original_mask_no_brightness_filter():
    ee = FakeEE()
    plan = scheduler.asset_plan(ee, "a")
    rgb, mask = scheduler.export_images(ee, "a")
    assert plan["crs_transform"] == [.8, 0, 100, 0, -.8, 3000]
    assert plan["width"] == plan["height"] == 3300
    names = [name for name, _, _ in ee.calls]
    assert "mask" in names and "reduce" in names and "eq" in names
    assert "gt" not in names and "resample" not in names and "reproject" not in names
    assert ("unmask", (0,), {"sameFootprint": True}) in ee.calls


def test_original_source_inventory_skips_existing_assets():
    row = dict(uri="gs://bucket/old/raw.tif", generation="7", bytes=50, crc32c="crc",
               asset_id="a", source_id="s", catalog_id="c", grid={}, geometry={})
    current = scheduler.initial_state({"asset_ids": ["a", "b"]}, [row], "gs://bucket/new-export", "p")
    assert current["assets"]["a"]["state"] == "SKIPPED_EXISTING"
    assert current["assets"]["b"]["state"] == "PENDING"
    assert scheduler.initial_state({"asset_ids": ["a", "b"]}, {"sources": [row]},
        "gs://bucket/new-export", "p")["assets"]["a"]["state"] == "SKIPPED_EXISTING"


def test_at_most_two_active_tasks_and_durable_state_before_start():
    ee = FakeEE()
    current = state()
    snapshots = []
    scheduler.tick(ee, current, lambda value: snapshots.append(deepcopy(value)), lambda job: [])
    assert len(ee.starts) == 2
    assert all(entry["state"] == "READY" for entry in current["assets"]["a"]["tasks"].values())
    submitting = [snapshot for snapshot in snapshots if any(
        task["state"] == "SUBMITTING" for asset in snapshot["assets"].values()
        for task in asset.get("tasks", {}).values())]
    assert len(submitting) >= 2
    assert current["assets"]["b"]["state"] == "PENDING"
    for options in ee.exports:
        assert options["crsTransform"] == [.8, 0, 100, 0, -.8, 3000]
        assert "scale" not in options and "dimensions" not in options
        assert options["fileDimensions"] == 8192


def test_uncertain_start_is_reconciled_and_never_resubmitted():
    ee = FakeEE()
    ee.uncertain = True
    current = state()
    with pytest.raises(RuntimeError, match="uncertain"):
        scheduler.tick(ee, current, lambda value: None, lambda job: [])
    assert len(ee.starts) == 1
    ee.uncertain = False
    with pytest.raises(RuntimeError, match="uncertain"):
        scheduler.tick(ee, current, lambda value: None, lambda job: [])
    assert len(ee.starts) == 1
    task = current["assets"]["a"]["tasks"]["rgb"]
    ee.remote.append(dict(id=ee.starts[0], description=task["description"], state="RUNNING"))
    scheduler.tick(ee, current, lambda value: None, lambda job: [])
    assert task["state"] == "RUNNING"
    assert ee.starts.count(ee.starts[0]) == 1


def test_unrelated_active_tasks_also_consume_capacity():
    ee = FakeEE()
    ee.remote = [dict(id="x", description="outside", state="RUNNING"),
                 dict(id="y", description="outside-2", state="READY")]
    scheduler.tick(ee, state(), lambda value: None, lambda job: [])
    assert ee.starts == []


def test_output_file_tiling_does_not_invent_resampling_grid():
    assert scheduler.tile_count(20000, 10000) == 6
    assert scheduler.tile_count(8192, 8192) == 1


def test_completed_exports_pair_original_mask_tiles_and_pin_objects():
    ee = FakeEE()
    current = state()
    del current["assets"]["b"]
    scheduler.tick(ee, current, lambda value: None, lambda job: [])
    for remote in ee.remote:
        remote["state"] = "COMPLETED"
    def pins(job):
        return [dict(uri="gs://bucket/" + job["file_prefix"] + "-0000000000-0000000000.tif",
                     generation="7", bytes=50, crc32c="crc")]
    scheduler.tick(ee, current, lambda value: None, pins)
    assert current["status"] == "complete"
    source = current["assets"]["a"]["source_objects"][0]
    assert source["producer_mask"]["generation"] == "7"
    assert source["mask_provenance"] == "producer_original_min_rgb_mask"
    assert source["license"] == "CC-BY-4.0"
    assert len(ee.starts) == 2


def test_failed_exports_have_three_attempt_limit_and_distinct_destinations():
    ee = FakeEE()
    current = state()
    del current["assets"]["b"]
    for _ in range(4):
        for task in ee.remote:
            task.update(state="FAILED", error_message="Transient worker failure")
        scheduler.tick(ee, current, lambda value: None, lambda job: [])
    assert current["assets"]["a"]["state"] == "FAILED_EXHAUSTED"
    assert len(ee.starts) == 6
    assert len({options["fileNamePrefix"] for options in ee.exports}) == 6
    assert current["status"] == "complete_with_failures"


def test_permission_failed_task_blocks_retries():
    ee = FakeEE()
    current = state()
    scheduler.tick(ee, current, lambda value: None, lambda job: [])
    ee.remote[0].update(state="FAILED", error_message="Permission denied writing bucket")
    with pytest.raises(RuntimeError, match="permission"):
        scheduler.tick(ee, current, lambda value: None, lambda job: [])
    assert current["status"] == "blocked_permission"
    assert len(ee.starts) == 2
    with pytest.raises(RuntimeError, match="permission"):
        scheduler.tick(ee, current, lambda value: None, lambda job: [])
    assert len(ee.starts) == 2


def test_uncertain_duplicate_task_matches_fail_closed():
    ee = FakeEE()
    current = state()
    scheduler.tick(ee, current, lambda value: None, lambda job: [])
    ee.remote.append({**ee.remote[0], "id": "unexpected-second-task"})
    with pytest.raises(RuntimeError, match="uncertain"):
        scheduler.tick(ee, current, lambda value: None, lambda job: [])
    assert len(ee.starts) == 2


class MemoryBlob:
    def __init__(self, name):
        self.name = name
        self.data = None
        self.generation = 1
        self.crc32c = "crc"
        self.size = 0

    def exists(self):
        return self.data is not None

    def reload(self):
        pass

    def download_as_bytes(self, **kwargs):
        assert kwargs == dict(if_generation_match=1, checksum="crc32c")
        return self.data

    def upload_from_string(self, data, **kwargs):
        assert kwargs == dict(if_generation_match=0, checksum="crc32c")
        assert self.data is None
        self.data = data
        self.size = len(data)


class MemoryStorage:
    def __init__(self):
        self.objects = {}

    def bucket(self, bucket):
        return SimpleNamespace(blob=lambda name: self.objects.setdefault(name, MemoryBlob(name)))

    def list_blobs(self, bucket, prefix):
        return [blob for name, blob in self.objects.items() if name.startswith(prefix) and blob.data is not None]


def test_immutable_cloud_journal_recovers_uncertain_start_and_detects_config_conflict(tmp_path):
    storage = MemoryStorage()
    journal = scheduler.CloudJournal(storage, "gs://bucket/new-export", tmp_path)
    ee = FakeEE()
    ee.uncertain = True
    current = state()
    with pytest.raises(RuntimeError, match="uncertain"):
        scheduler.tick(ee, current, journal.save, journal.pin_outputs)
    recovered = journal.load(state())
    assert recovered["assets"]["a"]["tasks"]["rgb"]["request_id"] == ee.starts[0]
    assert recovered["assets"]["a"]["tasks"]["rgb"]["state"] == "START_UNCERTAIN"
    with pytest.raises(RuntimeError, match="uncertain"):
        scheduler.tick(FakeEE(), recovered, journal.save, journal.pin_outputs)
    assert (tmp_path / "export-state.json").exists()
    changed = scheduler.initial_state({"asset_ids": ["a", "b"]}, [], "gs://bucket/new-export", "other-project")
    with pytest.raises(ValueError, match="configuration conflict"):
        journal.load(changed)


def test_rgb_mask_tile_mismatch_blocks_source_publication():
    ee = FakeEE()
    current = state()
    del current["assets"]["b"]
    scheduler.tick(ee, current, lambda value: None, lambda job: [])
    for remote in ee.remote:
        remote["state"] = "COMPLETED"
    def pins(job):
        suffix = "-0000000000-0000000000.tif" if "rgb" in job["file_prefix"] else "-0000000000-0000008192.tif"
        return [dict(uri="gs://bucket/" + job["file_prefix"] + suffix,
                     generation="7", bytes=50, crc32c="crc")]
    with pytest.raises(RuntimeError, match="tile grids differ"):
        scheduler.tick(ee, current, lambda value: None, pins)
    assert current["assets"]["a"]["source_objects"] == []


def test_expired_poll_deadline_does_not_start_exports():
    ee = FakeEE()
    scheduler.tick(ee, state(), lambda value: None, lambda job: [], deadline=0)
    assert ee.starts == []


def test_actual_ee_wkt_only_projection_retains_native_wkt_and_affine():
    ee = FakeEE()
    native_wkt = CRS.from_epsg(32752).to_wkt()
    actual = dict(type="Projection", wkt=native_wkt,
                  transform=[.8, 0, 432496, 0, -.8, 9621396.8])
    class WktExpression(Expression):
        def __getattr__(self, name):
            if name == "projection":
                return lambda: Expression(self.calls, actual)
            return super().__getattr__(name)
    ee.Image = lambda asset: WktExpression(ee.calls)
    plan = scheduler.asset_plan(ee, "SKYSAT/GEN-A/PUBLIC/ORTHO/RGB/s01_20150314T021941Z")
    assert CRS.from_user_input(plan["crs"]) == CRS.from_epsg(32752)
    assert plan["native_wkt"] == native_wkt
    assert plan["crs_transform"] == actual["transform"]


def test_typed_native_projection_comparison_does_not_depend_on_wkt_text():
    affine = [.8, 0, 0, 0, -.8, 3000]
    crs_code = scheduler.native_projection(dict(crs="EPSG:32752", transform=affine))
    wkt = scheduler.native_projection(dict(wkt=CRS.from_epsg(32752).to_wkt(), transform=affine))
    assert crs_code == wkt
    with pytest.raises(ValueError, match="CRS or WKT"):
        scheduler.native_projection(dict(type="Projection", transform=affine))


def test_real_like_metric_wkt_unit_m_and_base_crs_bounds():
    wkt=('PROJCS["UTM Zone 52, Southern Hemisphere",GEOGCS["WGS 84",'
        'DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],'
        'PRIMEM["Greenwich",0],UNIT["degree",0.017453292519943295]],'
        'PROJECTION["Transverse_Mercator"],PARAMETER["central_meridian",129],'
        'PARAMETER["latitude_of_origin",0],PARAMETER["scale_factor",0.9996],'
        'PARAMETER["false_easting",500000],PARAMETER["false_northing",10000000],'
        'UNIT["m",1],AXIS["x",EAST],AXIS["y",NORTH]]')
    assert CRS.from_wkt(wkt).linear_units=='m'
    ee=FakeEE(); affine=[.8,0,432496,0,-.8,9621396.8]
    rectangle=[432496,9618756.8,435136,9621396.8]
    class RealExpression(Expression):
        def __getattr__(self,name):
            if name=='projection':
                return lambda:Expression(self.calls,dict(type='Projection',wkt=wkt,transform=affine))
            if name=='bounds':
                def bounds(**kwargs):
                    self.calls.append(('bounds',(),kwargs))
                    assert kwargs['proj'].transform==[1,0,0,0,1,0]
                    return self
                return bounds
            if name=='coordinates':
                return lambda:Expression(self.calls,[[[rectangle[0],rectangle[1]],
                    [rectangle[2],rectangle[1]],[rectangle[2],rectangle[3]],
                    [rectangle[0],rectangle[3]],[rectangle[0],rectangle[1]]]])
            return super().__getattr__(name)
    ee.Image=lambda asset:RealExpression(ee.calls)
    plan=scheduler.asset_plan(ee,'a')
    assert plan['native_bbox']==rectangle
    assert plan['crs_transform']==affine
    assert plan['width']==plan['height']==3300
    assert plan['geometry_coordinate_space']=='native_base_crs_metres'


def test_exports_are_explicitly_bounded_after_unmask():
    ee=FakeEE(); rectangles=[]
    ee.Geometry.Rectangle=lambda bbox,**kwargs:rectangles.append((bbox,kwargs)) or 'bounded-region'
    current=state()
    scheduler.tick(ee,current,lambda value:None,lambda job:[])
    assert rectangles
    assert all(options['region']=='bounded-region' for options in ee.exports)
    assert sum(name=='clip' and args==('bounded-region',) for name,args,_ in ee.calls)==2
    assert all(rectangle[1]==dict(proj='EPSG:32644',geodesic=False) for rectangle in rectangles)


def test_foot_units_and_invalid_bounds_still_rejected():
    ee=FakeEE()
    feet=CRS.from_epsg(2227).to_wkt()
    class Feet(Expression):
        def __getattr__(self,name):
            if name=='projection':
                return lambda:Expression(self.calls,dict(wkt=feet,transform=[.8,0,0,0,-.8,0]))
            return super().__getattr__(name)
    ee.Image=lambda asset:Feet(ee.calls)
    with pytest.raises(ValueError,match='metric'):scheduler.asset_plan(ee,'a')
    class Invalid(Expression):
        def __getattr__(self,name):
            if name=='coordinates':return lambda:Expression(self.calls,[[[float('inf'),0],[1,1],[0,0]]])
            return super().__getattr__(name)
    ee.Image=lambda asset:Invalid(ee.calls)
    with pytest.raises(ValueError,match='bounded'):scheduler.asset_plan(ee,'a')
