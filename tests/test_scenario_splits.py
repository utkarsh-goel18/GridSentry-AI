from ml.dataset import get_scenario_splits


def _assert_disjoint_and_complete(train, val, test):
    assert not (train & val)
    assert not (train & test)
    assert not (val & test)
    assert len(train | val | test) == 171


def test_default_scenario_split_preserves_existing_partition_sizes():
    train, val, test = get_scenario_splits()
    _assert_disjoint_and_complete(train, val, test)
    assert (len(train), len(val), len(test)) == (117, 27, 27)


def test_distance_holdout_keeps_entire_distance_out_of_training_and_validation():
    held_out_distance = 20
    train, val, test = get_scenario_splits(
        split_strategy="distance", holdout_value=held_out_distance
    )
    _assert_disjoint_and_complete(train, val, test)
    assert len(test) == 57
    assert all(scenario[0] == held_out_distance for scenario in test)
    assert all(scenario[0] != held_out_distance for scenario in train | val)


def test_resistance_holdout_keeps_entire_resistance_out_of_training_and_validation():
    held_out_resistance = 50.0
    train, val, test = get_scenario_splits(
        split_strategy="resistance", holdout_value=held_out_resistance
    )
    _assert_disjoint_and_complete(train, val, test)
    assert len(test) == 57
    assert all(scenario[1] == held_out_resistance for scenario in test)
    assert all(scenario[1] != held_out_resistance for scenario in train | val)


def test_invalid_holdout_value_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        get_scenario_splits(split_strategy="distance", holdout_value=999)
