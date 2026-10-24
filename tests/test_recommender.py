from types import SimpleNamespace

import pytest

import recommender
from recommender import rerank, validate_inputs


@pytest.fixture
def stub_model(monkeypatch):
    """既知のカタログ・対応表を持つ読込済みモデルを準備する。"""
    model = SimpleNamespace(
        movie_catalog={
            1: ("A", ["Action"]),
            2: ("B", ["Action"]),
            3: ("C", ["Comedy"]),
            4: ("D", ["Drama"]),
        },
        movie_to_index={1: 0, 2: 1, 3: 2},
    )
    monkeypatch.setattr(recommender, "load_model", lambda name: model)
    return model


@pytest.mark.parametrize(
    ("items", "message"),
    [
        ([(1, 5), (2, 4)], "3本以上"),
        ([(1, 5), (1, 4), (2, 3)], "重複"),
        ([(1, 5), (2, 6), (3, 4)], "1〜5"),
        ([(1, 5), (2, 0), (3, 4)], "1〜5"),
    ],
)
def test_validate_inputs_rejects_invalid_items(stub_model, items, message):
    with pytest.raises(ValueError, match=message):
        validate_inputs(items, loaded_model=stub_model)


def test_validate_inputs_rejects_movie_missing_from_model_map(stub_model):
    with pytest.raises(ValueError, match="対応表"):
        validate_inputs(
            [(1, 5), (2, 4), (4, 3)],
            loaded_model=stub_model,
        )


def test_validate_inputs_rejects_movie_missing_from_catalog(stub_model):
    with pytest.raises(ValueError, match="カタログ"):
        validate_inputs(
            [(1, 5), (2, 4), (5, 3)],
            loaded_model=stub_model,
        )


def test_rerank_uses_prediction_order_when_lambda_is_zero():
    candidates = [
        (1, 5.0, ["Action"]),
        (2, 4.9, ["Action"]),
        (3, 4.0, ["Comedy"]),
    ]

    assert [movie_id for movie_id, _, _ in rerank(candidates, lam=0, k=2)] == [1, 2]


def test_rerank_rewards_unselected_genres():
    candidates = [
        (1, 5.0, ["Action"]),
        (2, 4.9, ["Action"]),
        (3, 4.0, ["Comedy"]),
    ]

    assert [movie_id for movie_id, _, _ in rerank(candidates, lam=1, k=2)] == [1, 3]
