from types import SimpleNamespace

import numpy as np

import evaluate
from recommender import rerank


def test_ncf_evaluation_ranks_raw_scores_and_clips_rmse(monkeypatch):
    movie_ids = np.arange(1, 6, dtype=np.int64)
    split_data = {
        7: {
            "_used_input": [[1, 5], [2, 4], [3, 3]],
            "input": [[1, 5], [2, 4], [3, 3]],
            "eval": [[5, 5]],
        }
    }
    training_data = {
        "catalog": {
            movie_id: (f"Movie {movie_id}", ["Drama"])
            for movie_id in movie_ids
        },
        "item_counts": {int(movie_id): 1 for movie_id in movie_ids},
        "input_movies_removed": 0,
        "users_with_reduced_input": 0,
        "users_excluded_for_short_input": 0,
    }
    loaded_model = SimpleNamespace(
        movie_to_index={int(movie_id): int(movie_id - 1) for movie_id in movie_ids}
    )
    monkeypatch.setattr(evaluate, "load_model", lambda name: loaded_model)
    monkeypatch.setattr(evaluate, "validate_inputs", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        evaluate,
        "fold_in_user",
        lambda model, item_idx, ratings: ({}, 0.0, 0.0),
    )

    def fake_predict_ratings(model, user_params, clip=True):
        assert clip is False
        return np.asarray([3.0, 3.5, 4.0, 5.2, 5.4])

    monkeypatch.setattr(evaluate, "predict_ratings", fake_predict_ratings)
    predictions, loaded = evaluate._model_predictions(
        "ncf",
        split_data,
        training_data,
    )

    ranking_inputs = []
    ranked_movie_ids = []
    original_rerank = rerank

    def capture_rerank(candidates, lam, k=10):
        ranking_inputs.extend(candidates)
        ranked = original_rerank(candidates, lam=lam, k=k)
        ranked_movie_ids.extend(movie_id for movie_id, _, _ in ranked)
        return ranked

    monkeypatch.setattr(evaluate, "rerank", capture_rerank)
    summary, details = evaluate._evaluate_configuration(
        "ncf",
        0.0,
        split_data,
        training_data,
        predictions,
        loaded,
    )

    assert [(movie_id, score) for movie_id, score, _ in ranking_inputs] == [
        (4, 5.2),
        (5, 5.4),
    ]
    assert ranked_movie_ids == [5, 4]
    assert summary["RMSE pooled"] == 0.0
    assert details[0]["RMSE"] == 0.0
