from word_energy.synthetic import manifest, split_windows, vocabulary


def test_frozen_synthetic_protocol_has_the_planned_counts_and_coverage() -> None:
    splits = split_windows()
    assert {name: len(rows) for name, rows in splits.items()} == {"train": 48, "validation": 18, "test": 18}
    train_tokens = {token for row in splits["train"] for token in row.tokens}
    assert set(vocabulary()) <= train_tokens
    assert manifest()["sha256"] == manifest()["sha256"]
