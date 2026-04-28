import pytest
import pandas as pd
from correlation_analyzer import run_correlations, print_correlations, TRAITS


def _make_df(racial, emotional, domestic, figurative):
    return pd.DataFrame({
        "character": [f"char_{i}" for i in range(len(racial))],
        "racial_consciousness": racial,
        "emotional_register": emotional,
        "domestic_score": domestic,
        "figurative_per_1000_words": figurative,
    })


@pytest.fixture
def basic_results():
    df = _make_df([1, 2, 3, 4], [4, 3, 2, 1], [1, 3, 2, 4], [1, 2, 3, 4])
    return run_correlations(df)


def test_run_correlations_returns_list_of_three(basic_results):
    assert isinstance(basic_results, list)
    assert len(basic_results) == 3


def test_each_dict_has_correct_keys(basic_results):
    for item in basic_results:
        assert set(item.keys()) == {"trait", "r", "p_value"}


def test_r_values_are_floats_in_range(basic_results):
    for item in basic_results:
        assert isinstance(item["r"], float)
        assert -1.0 <= item["r"] <= 1.0


def test_p_values_are_floats_in_range(basic_results):
    for item in basic_results:
        assert isinstance(item["p_value"], float)
        assert 0.0 <= item["p_value"] <= 1.0


def test_all_three_trait_names_present(basic_results):
    trait_names = {item["trait"] for item in basic_results}
    assert trait_names == set(TRAITS)


def test_perfect_positive_correlation():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    df = _make_df(vals, vals, vals, vals)
    results = run_correlations(df)
    for item in results:
        assert item["r"] == pytest.approx(1.0, abs=1e-9)


def test_perfect_negative_correlation():
    forward = [1.0, 2.0, 3.0, 4.0, 5.0]
    backward = [5.0, 4.0, 3.0, 2.0, 1.0]
    df = _make_df(backward, backward, backward, forward)
    results = run_correlations(df)
    for item in results:
        assert item["r"] == pytest.approx(-1.0, abs=1e-9)


def test_print_correlations_output(capsys, basic_results):
    print_correlations(basic_results)
    captured = capsys.readouterr()
    assert "Correlation Results:" in captured.out
    for trait in TRAITS:
        assert trait in captured.out
    assert "r=" in captured.out
    assert "p=" in captured.out


def test_run_correlations_raises_on_nan():
    df = _make_df([1, float("nan"), 3, 4], [4, 3, 2, 1], [1, 3, 2, 4], [1, 2, 3, 4])
    with pytest.raises(ValueError, match="NaN"):
        run_correlations(df)


def test_run_correlations_preserves_trait_order():
    df = _make_df([1, 2, 3, 4], [4, 3, 2, 1], [1, 3, 2, 4], [1, 2, 3, 4])
    results = run_correlations(df)
    assert [item["trait"] for item in results] == list(TRAITS)
