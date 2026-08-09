from sky_v1.data.toy_generator import ToyDataGenerator
from sky_v1.data.datasets import Phase1Dataset, Phase2AlignDataset, Phase3DistillDataset
from sky_v1.data.collator import SkyDataCollator

def test_toy_generator_produces_n_samples():
    g = ToyDataGenerator(n=10, seed=42)
    s = list(g.generate_all())
    assert len(s) == 10
    sample = s[0]
    for k in ("text_ids","image","audio","video","three_d_points","text_labels"):
        assert k in sample

def test_phase1_dataset_getitem_no_crash():
    g = ToyDataGenerator(n=4, seed=1)
    ds = Phase1Dataset(list(g.generate_all()), phase="text")
    assert len(ds) == 4
    item = ds[0]
    assert "input_ids" in item and "labels" in item

def test_collator_batch_has_expected_keys():
    g = ToyDataGenerator(n=3, seed=1)
    ds = Phase2AlignDataset(list(g.generate_all()))
    col = SkyDataCollator(max_seq_len=128)
    batch = col([ds[i] for i in range(3)])
    assert "inputs" in batch
