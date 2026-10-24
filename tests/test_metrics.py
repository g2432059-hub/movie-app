import pytest

from metrics import (
    genre_count,
    high_threshold,
    precision_at_k,
    rmse_macro,
    rmse_pooled,
)


def test_rmse_pooled():
    assert round(rmse_pooled([3.5, 3, 4], [4, 3, 5]), 4) == 0.6455


def test_rmse_macro():
    user_pairs = [
        ([3.5, 3, 4], [4, 3, 5]),
        ([2, 2], [2, 2]),
    ]
    assert round(rmse_macro(user_pairs), 4) == 0.3227


def test_high_threshold_top25_and_relevant_count():
    eval_ratings = [5, 5, 4, 4, 4, 4, 3, 3, 2, 1]
    threshold = high_threshold(eval_ratings, "top25")

    assert threshold == 4
    assert sum(rating >= threshold for rating in eval_ratings) == 6


def test_high_threshold_top25_returns_none_for_constant_ratings():
    assert high_threshold([3, 3, 3], "top25") is None


def test_high_threshold_fixed4():
    assert high_threshold([1, 2, 3], "fixed4") == 4


def test_precision_at_k_counts_relevant_items_in_top_k():
    recommended_ids = list(range(1, 11))
    relevant_ids = {1, 3, 5}

    assert precision_at_k(recommended_ids, relevant_ids) == 0.3


def test_precision_at_k_returns_zero_for_empty_relevant_ids():
    assert precision_at_k([1, 2, 3], set()) == 0.0


def test_precision_at_k_uses_k_as_denominator_when_recommendations_are_short():
    assert precision_at_k([1, 2, 3], {1, 2, 3}, k=10) == 0.3


def test_genre_count():
    assert genre_count([["Action", "Comedy"], ["Action"], ["Drama"]]) == 3


def test_genre_count_returns_zero_for_empty_input():
    assert genre_count([]) == 0


@pytest.mark.parametrize(
    ("function", "args"),
    [
        (rmse_pooled, ([], [])),
        (rmse_macro, ([],)),
        (high_threshold, ([], "fixed4")),
    ],
)
def test_rmse_and_threshold_functions_reject_empty_input(function, args):
    with pytest.raises(ValueError):
        function(*args)
