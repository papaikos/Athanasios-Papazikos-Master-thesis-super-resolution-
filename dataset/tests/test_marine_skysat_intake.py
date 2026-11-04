import pytest
from scripts.marine_skysat_intake import (
    cropped_grid,
    candidate_assets,
    global_assignments,
)
from src.marine_acquire import digest


def test_native_crop_reservation_and_alignment():
    grid = {
        "crsCode": "EPSG:32644",
        "dimensions": {"width": 1000, "height": 1000},
        "affineTransform": {
            "scaleX": 0.8,
            "scaleY": -0.8,
            "translateX": 1000,
            "translateY": 2000,
        },
    }
    result = cropped_grid(grid, [1008.1, 1983.9, 1023.9, 1991.9])
    assert result["dimensions"] == {"width": 20, "height": 11}
    assert result["affineTransform"]["translateX"] == 1008
    assert result["affineTransform"]["translateY"] == 1992
    assert grid["dimensions"]["width"] == 1000


def test_crop_outside_native_grid_rejected():
    grid = {
        "dimensions": {"width": 10, "height": 10},
        "affineTransform": {
            "scaleX": 1,
            "scaleY": -1,
            "translateX": 0,
            "translateY": 10,
        },
    }
    with pytest.raises(ValueError, match="outside"):
        cropped_grid(grid, [20, 20, 30, 30])


def test_prior_exclusion_and_frozen_split():
    assets = [
        {
            "id": str(i),
            "properties": {"catalogID": str(i)},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [i * 10, 0],
                        [i * 10 + 1, 0],
                        [i * 10 + 1, 1],
                        [i * 10, 1],
                        [i * 10, 0],
                    ]
                ],
            },
        }
        for i in range(5)
    ]
    assignment = global_assignments(assets)
    train = next(a for a in assets if assignment[a["id"]]["split"] == "train")
    prior = {"sources": [{"asset_id": train["id"], **assignment[train["id"]]}]}
    prior["sha256"] = digest(prior)
    roi = {"bbox": [-1, -1, 100, 2]}
    selected = candidate_assets(assets, [prior], roi)
    assert train["id"] not in {x["asset"]["id"] for x in selected}
    assert all(x["split"] == "train" for x in selected)
    prior["sources"][0]["split"] = "test"
    prior["sha256"] = digest({k: v for k, v in prior.items() if k != "sha256"})
    with pytest.raises(ValueError, match="split changed"):
        candidate_assets(assets, [prior], roi)
