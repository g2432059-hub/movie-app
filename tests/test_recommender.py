import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import recommender
from recommender import predict_ratings, recommend, rerank, validate_inputs


def test_fold_in_user_defaults_to_l2_one():
    assert inspect.signature(recommender.fold_in_user).parameters["l2"].default == 1.0


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


def test_recommend_ranks_raw_predictions_but_returns_clipped_values(monkeypatch):
    class FakeNetwork:
        def eval(self):
            return self

        def predict_with_user_params(self, user_vec, item_idx):
            return torch.tensor([3.0, 3.5, 4.0, 5.2, 5.4])

    model = SimpleNamespace(
        name="ncf",
        model=FakeNetwork(),
        movie_to_index={movie_id: movie_id - 1 for movie_id in range(1, 6)},
        index_to_movie={index: movie_id for movie_id in range(1, 6) for index in [movie_id - 1]},
        movie_catalog={
            1: ("A", ["Action"]),
            2: ("B", ["Action"]),
            3: ("C", ["Action"]),
            4: ("Lower raw score", ["Action"]),
            5: ("Higher raw score", ["Action"]),
        },
        item_counts=np.ones(5, dtype=np.int64),
    )
    monkeypatch.setattr(recommender, "load_model", lambda name: model)
    monkeypatch.setattr(
        recommender,
        "fold_in_user",
        lambda loaded, item_idx, ratings: (user_params, 0.0, 0.0),
    )

    user_params = {"user_vec": torch.tensor([0.0])}
    np.testing.assert_allclose(
        predict_ratings(model, user_params, clip=False),
        [3.0, 3.5, 4.0, 5.2, 5.4],
    )
    assert predict_ratings(model, user_params).tolist() == [3.0, 3.5, 4.0, 5.0, 5.0]

    recommendations = recommend([(1, 5), (2, 4), (3, 3)], lam=0)

    assert [title for title, _, _ in recommendations] == [
        "Higher raw score",
        "Lower raw score",
    ]
    assert [prediction for _, prediction, _ in recommendations] == [5.0, 5.0]
