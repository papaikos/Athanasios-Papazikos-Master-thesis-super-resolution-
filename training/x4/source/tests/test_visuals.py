import json
import torch
from PIL import Image


def test_panels_freeze_ids_ranges_and_preserve_training_state(tmp_path):
    from research.visuals import write_panels

    class Dataset:
        members = [("shard.tar", name + ".npz") for name in ["z", "a", "b", "c"]]

        def __len__(self):
            return len(self.members)

        def __getitem__(self, index):
            return {"sample_id": self.members[index][1][:-4],
                    "lr": torch.full((3, 4, 4), 0.5),
                    "hr": torch.full((3, 16, 16), 0.5)}

    model = torch.nn.Upsample(scale_factor=4, mode="nearest")
    model.train()
    rng = torch.get_rng_state().clone()
    files = write_panels(model, Dataset(), "cpu", tmp_path, 0)
    assert model.training and torch.equal(rng, torch.get_rng_state())
    selection = json.loads((tmp_path / "config" / "fixed-visual-identities.json").read_text())
    assert selection["sample_ids"] == ["a", "b", "c"]
    assert selection["rgb_display_range"] == [0, 1]
    assert selection["error_display_range"] == [0, 0.25]
    assert len(files) == 3
    for path in files:
        with Image.open(path) as panel:
            assert panel.size == (64, 64)
