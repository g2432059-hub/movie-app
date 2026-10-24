import argparse
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import numpy as np
import torch

from data_utils import load_movies
from models import MF, NCF
from recommender import (
    LoadedModel,
    fold_in_user,
    load_model,
    predict_ratings,
    recommend,
    rerank,
    to_movie_index,
    validate_inputs,
)


MODELS_DIR = Path("models")
MOVIES_PATH = Path("data/ml-1m/movies.dat")
SAMPLE_MOVIES = (
    (260, 5),   # Star Wars: Episode IV - A New Hope
    (1196, 5),  # Star Wars: Episode V - The Empire Strikes Back
    (1210, 5),  # Star Wars: Episode VI - Return of the Jedi
    (2571, 4),  # Matrix, The
    (589, 4),   # Terminator 2: Judgment Day
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """check_model.pyの引数を解析する。

    Args:
        argv: 引数列。Noneならsys.argvを使う。

    Returns:
        model名を持つargparse.Namespace。
    """
    parser = argparse.ArgumentParser(description="推薦モデルのfold-inと予測を確認する")
    parser.add_argument("--model", choices=("ncf", "mf", "mean"), default="ncf")
    return parser.parse_args(argv)


def select_sample(
    model: LoadedModel,
) -> tuple[list[tuple[int, int]], list[tuple[int, str, list[str]]]]:
    """Sci-Fi入力映画をカタログと学習ID対応表で検証する。

    Args:
        model: load_model()が返すモデル情報。

    Returns:
        (movie_id, rating)の入力列と、(movie_id, title, genres)列。

    Raises:
        ValueError: 固定映画がカタログまたは学習ID対応表にない場合。
    """
    sample: list[tuple[int, int]] = []
    titles: list[tuple[int, str, list[str]]] = []
    for movie_id, rating in SAMPLE_MOVIES:
        if movie_id not in model.movie_catalog:
            raise ValueError(f"サンプル映画ID {movie_id} はmovies.datにありません。")
        if movie_id not in model.movie_to_index:
            raise ValueError(
                f"サンプル映画ID {movie_id} は学習データの映画ID対応表にありません。"
            )
        title, genres = model.movie_catalog[movie_id]
        if "Sci-Fi" not in genres:
            raise ValueError(f"サンプル映画ID {movie_id} ({title}) はSci-Fi映画ではありません。")
        sample.append((movie_id, rating))
        titles.append((movie_id, title, genres))
    return sample, titles


def _raw_predictions(
    model: LoadedModel,
    user_params: dict[str, torch.Tensor],
) -> np.ndarray:
    """利用者パラメータからclip前の全映画予測を計算する。

    Args:
        model: load_model()が返すモデル情報。
        user_params: fold_in_user()が返す利用者パラメータ。

    Returns:
        映画index順のclip前予測値。
    """
    model.model.eval()
    item_idx = torch.arange(len(model.movie_to_index), dtype=torch.long)
    with torch.no_grad():
        if model.name == "mf":
            predictions = model.model.predict_with_user_params(
                user_params["user_vec"],
                user_params["user_bias"],
                item_idx,
            )
        else:
            predictions = model.model.predict_with_user_params(
                user_params["user_vec"],
                item_idx,
            )
    return predictions.cpu().numpy()


def _run_mean_check() -> None:
    """mean checkpointで入力、全映画予測、再ランキングを確認する。"""
    checkpoint = torch.load(
        MODELS_DIR / "mean.pt",
        map_location="cpu",
        weights_only=False,
    )
    catalog_frame = load_movies(MOVIES_PATH)
    catalog = {
        int(row.movie_id): (str(row.title), list(row.genres))
        for row in catalog_frame.itertuples(index=False)
    }
    movie_to_index = checkpoint["movie_to_index"]
    items: list[tuple[int, int]] = []
    for movie_id, rating in SAMPLE_MOVIES:
        if movie_id not in catalog:
            raise ValueError(f"サンプル映画ID {movie_id} はmovies.datにありません。")
        if movie_id not in movie_to_index:
            raise ValueError(
                f"サンプル映画ID {movie_id} は学習データの映画ID対応表にありません。"
            )
        title, genres = catalog[movie_id]
        if "Sci-Fi" not in genres:
            raise ValueError(f"サンプル映画ID {movie_id} ({title}) はSci-Fi映画ではありません。")
        items.append((movie_id, rating))

    print("入力(title, rating):")
    for (movie_id, rating) in items:
        print(f"  ({catalog[movie_id][0]}, {rating})")
    item_idx, ratings = to_movie_index(
        items,
        SimpleNamespace(movie_to_index=movie_to_index),
    )
    print(
        f"変換後index: dtype={item_idx.dtype}, shape={item_idx.shape}; "
        f"評価: min={ratings.min()}, max={ratings.max()}"
    )
    print("fold-in損失: N/A (meanモデルにはfold-inがありません)")

    item_means = checkpoint["item_means"].cpu().numpy()
    item_counts = checkpoint["item_counts"].cpu().numpy()
    index_to_movie = checkpoint["index_to_movie"]

    def predict_all() -> np.ndarray:
        return np.clip(item_means, 1.0, 5.0)

    started = time.perf_counter()
    raw_predictions = item_means.copy()
    initial_elapsed = time.perf_counter() - started
    warm_started = time.perf_counter()
    predictions = predict_all()
    warm_elapsed = time.perf_counter() - warm_started

    top_indices = sorted(
        range(len(raw_predictions)),
        key=lambda index: (-float(raw_predictions[index]), index_to_movie[index]),
    )[:5]
    print("生予測上位5件(movie_id, title, pred, genres):")
    for index in top_indices:
        movie_id = index_to_movie[index]
        title, genres = catalog[movie_id]
        print(f"  ({movie_id}, {title}, {raw_predictions[index]:.4f}, {genres})")

    seen_ids = {movie_id for movie_id, _ in items}
    candidates = [
        (movie_id, float(predictions[index]), catalog[movie_id][1])
        for index, movie_id in index_to_movie.items()
        if item_counts[index] > 0 and movie_id not in seen_ids and movie_id in catalog
    ]
    for lam in (0.0, 0.25):
        print(f"mean最終推薦 上位5件 (lambda={lam:g}):")
        for movie_id, prediction, genres in rerank(candidates, lam=lam, k=5):
            print(f"  ({catalog[movie_id][0]}, {prediction:.4f}, {list(genres)})")
    print(
        "全映画予測時間: "
        f"初回={initial_elapsed:.6f}秒, ウォームアップ後={warm_elapsed:.6f}秒"
    )


def main(argv: Sequence[str] | None = None) -> None:
    """サンプル入力でモデルのfold-in・予測・推薦を表示する。

    Args:
        argv: CLI引数。Noneならsys.argvを使う。

    Returns:
        None。
    """
    args = parse_args(argv)
    device = "CUDA" if torch.cuda.is_available() else "CPU"
    print(f"モデル: {args.model} / 実行デバイス: {device}")

    if args.model == "mean":
        _run_mean_check()
        return
    model = load_model(args.model)
    items, sample_titles = select_sample(model)
    validate_inputs(items, model_name=args.model, loaded_model=model)

    print("入力(title, rating):")
    for (_, rating), (_, title, _) in zip(items, sample_titles, strict=True):
        print(f"  ({title}, {rating})")

    item_idx, ratings = to_movie_index(items, model)
    print(
        f"変換後index: dtype={item_idx.dtype}, shape={item_idx.shape}; "
        f"評価: min={ratings.min()}, max={ratings.max()}"
    )

    start = time.perf_counter()
    user_params, initial_loss, final_loss = fold_in_user(
        model,
        item_idx,
        ratings,
    )
    first_raw_predictions = predict_ratings(model, user_params, clip=False)
    first_elapsed = time.perf_counter() - start

    warm_start = time.perf_counter()
    user_params, initial_loss, final_loss = fold_in_user(
        model,
        item_idx,
        ratings,
    )
    raw_predictions = _raw_predictions(model, user_params)
    predictions = predict_ratings(model, user_params)
    warm_elapsed = time.perf_counter() - warm_start

    print(f"fold-in損失: 最初={initial_loss:.6f}, 最後={final_loss:.6f}")
    print("クリップ前の生予測上位5件(movie_id, pred, genres):")
    top_indices = sorted(
        range(len(first_raw_predictions)),
        key=lambda index: (
            -float(first_raw_predictions[index]),
            model.index_to_movie[index],
        ),
    )[:5]
    for index in top_indices:
        movie_id = model.index_to_movie[index]
        title, genres = model.movie_catalog[movie_id]
        raw_prediction = float(first_raw_predictions[index])
        print(f"  ({movie_id}, {title}, {raw_prediction:.4f}, {genres})")

    for lam in (0.0, 0.25):
        print(f"recommend() 最終結果 上位5件 (lambda={lam:g}):")
        final_recommendations = recommend(
            items,
            lam=lam,
            model_name=args.model,
        )
        for title, prediction, genres in final_recommendations[:5]:
            print(f"  ({title}, {prediction:.4f}, {genres})")

    print(
        "fold-in + 全映画予測時間: "
        f"初回={first_elapsed:.6f}秒, ウォームアップ後={warm_elapsed:.6f}秒"
    )


if __name__ == "__main__":
    main()
