import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from data_utils import build_id_maps, load_movies, load_ratings
from metrics import (
    genre_count,
    high_threshold,
    precision_at_k,
    rmse_macro,
    rmse_pooled,
)
from recommender import (
    LoadedModel,
    fold_in_user,
    load_model,
    predict_ratings,
    rerank,
    validate_inputs,
)


DATA_DIR = Path("data/ml-1m")
SPLITS_DIR = Path("splits")
RESULTS_DIR = Path("results")
SEED = 42
SHRINK = 10.0
ALL_NCF_LAMBDAS = (0.0, 0.1, 0.25, 0.5, 1.0)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """評価CLIの引数を解析する。

    Args:
        argv: 引数列。Noneならsys.argvを使う。

    Returns:
        split、model、lam、limit、all設定のNamespace。
    """
    parser = argparse.ArgumentParser(description="MovieLens推薦モデルを評価する")
    parser.add_argument("split", type=Path, help="評価する分割 (例: splits/dev)")
    parser.add_argument(
        "--model",
        choices=("mean", "popular", "random", "mf", "ncf"),
        default="ncf",
    )
    parser.add_argument("--lam", type=float, default=0.25)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--all", action="store_true", help="全モデルと指定λを評価")
    args = parser.parse_args(argv)
    if args.lam < 0:
        parser.error("--lam must be non-negative")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    return args


def _resolve_split_path(split_path: Path) -> tuple[Path, str, bool]:
    """分割パスを正規化し、分割名とtest判定を返す。

    Args:
        split_path: 分割ディレクトリ、分割名、または *_inputs.json のパス。

    Returns:
        (正規化済みパス, 分割名, test対象か)の組。
    """
    resolved = split_path.expanduser().resolve()
    if resolved.suffix.lower() == ".json":
        split_name = resolved.stem
        if split_name.endswith("_inputs"):
            split_name = split_name[: -len("_inputs")]
    else:
        split_name = resolved.name

    is_test = split_name.lower().startswith("test")
    return resolved, split_name, is_test


def load_split_data(
    split_path: Path,
    limit: int | None = None,
) -> tuple[str, dict[int, dict[str, list[list[int]]]]]:
    """指定された分割の入力/eval JSONだけを読み込む。

    Args:
        split_path: 分割ディレクトリ、分割名、または *_inputs.json のパス。
        limit: 先頭から読み込む利用者数。Noneなら全利用者。

    Returns:
        分割名と、user_idをキーにしたinput/eval辞書。
    """
    resolved, split_name, _ = _resolve_split_path(split_path)
    if resolved.suffix.lower() == ".json":
        json_path = resolved
    elif resolved.is_dir():
        json_path = resolved / f"{split_name}_inputs.json"
    else:
        json_path = resolved.parent / f"{split_name}_inputs.json"
    with json_path.open(encoding="utf-8") as split_file:
        raw_data = json.load(split_file)
    entries = list(raw_data.items())
    if limit is not None:
        entries = entries[:limit]
    data = {
        int(user_id): {
            "input": [[int(movie_id), int(rating)] for movie_id, rating in value["input"]],
            "eval": [[int(movie_id), int(rating)] for movie_id, rating in value["eval"]],
        }
        for user_id, value in entries
    }
    return split_name, data


def load_training_data() -> dict[str, Any]:
    """train利用者の評価だけから映画統計とカタログを作る。

    Returns:
        train評価、映画カタログ、映画ごとの評価件数/平均、
        全体平均を含む辞書。
    """
    train_users_path = SPLITS_DIR / "train_users.txt"
    train_users = {
        int(line)
        for line in train_users_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    ratings = load_ratings(DATA_DIR / "ratings.dat")
    train_ratings = ratings.loc[ratings["user_id"].isin(train_users)].copy()
    if train_ratings.empty:
        raise ValueError("train利用者の評価が見つかりません。")

    movies = load_movies(DATA_DIR / "movies.dat")
    catalog = {
        int(row.movie_id): (str(row.title), list(row.genres))
        for row in movies.itertuples(index=False)
    }
    grouped = train_ratings.groupby("movie_id")["rating"].agg(["sum", "count"])
    global_mean = float(train_ratings["rating"].mean())
    item_counts = {int(movie_id): int(row["count"]) for movie_id, row in grouped.iterrows()}
    item_means = {
        int(movie_id): (
            float(row["sum"]) + SHRINK * global_mean
        )
        / (int(row["count"]) + SHRINK)
        for movie_id, row in grouped.iterrows()
    }
    return {
        "ratings": train_ratings,
        "catalog": catalog,
        "item_counts": item_counts,
        "item_means": item_means,
        "global_mean": global_mean,
    }


def _model_predictions(
    model_name: str,
    split_data: dict[int, dict[str, list[list[int]]]],
    training_data: dict[str, Any],
    seed: int = SEED,
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], LoadedModel | None]:
    """モデル別に映画スコアを作り、MF/NCFのfold-inは各利用者一度だけ行う。

    Args:
        model_name: mean, popular, random, mf, ncfのいずれか。
        split_data: 評価対象の利用者ごとのinput/evalデータ。
        training_data: train評価から作成した映画統計。
        seed: random順位の固定seed。

    Returns:
        user_idごとの映画ID→score配列と、読込済みMF/NCFモデル。
    """
    catalog = training_data["catalog"]
    movie_ids = np.asarray(
        sorted(
            movie_id
            for movie_id in catalog
            if movie_id in training_data["item_counts"]
        ),
        dtype=np.int64,
    )

    if model_name in {"mf", "ncf"}:
        loaded_model = load_model(model_name)
        movie_scores: dict[int, np.ndarray] = {}
        for user_id, user_data in split_data.items():
            items = [tuple(row) for row in user_data["_used_input"]]
            validate_inputs(items, model_name=model_name, loaded_model=loaded_model)
            item_idx = np.asarray(
                [loaded_model.movie_to_index[movie_id] for movie_id, _ in items],
                dtype=np.int64,
            )
            input_ratings = np.asarray(
                [rating for _, rating in items],
                dtype=np.float32,
            )
            user_params, _, _ = fold_in_user(
                loaded_model,
                item_idx,
                input_ratings,
            )
            predictions = predict_ratings(loaded_model, user_params)
            movie_scores[user_id] = np.asarray(
                [
                    predictions[loaded_model.movie_to_index[int(movie_id)]]
                    for movie_id in movie_ids
                ],
                dtype=np.float64,
            )
        return {user_id: (movie_ids, scores) for user_id, scores in movie_scores.items()}, loaded_model

    if model_name == "mean":
        base_scores = np.asarray(
            [training_data["item_means"][int(movie_id)] for movie_id in movie_ids],
            dtype=np.float64,
        )
    elif model_name == "popular":
        base_scores = np.asarray(
            [training_data["item_counts"][int(movie_id)] for movie_id in movie_ids],
            dtype=np.float64,
        )
    else:
        base_scores = np.random.RandomState(seed).random_sample(len(movie_ids))

    return {
        user_id: (movie_ids, base_scores.copy())
        for user_id in split_data
    }, None


def _evaluate_configuration(
    model_name: str,
    lam: float,
    split_data: dict[int, dict[str, list[list[int]]]],
    training_data: dict[str, Any],
    predictions: dict[int, tuple[np.ndarray, np.ndarray]],
    loaded_model: LoadedModel | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """1つのモデルとλについて集計指標・利用者別詳細を作る。

    Args:
        model_name: 評価するモデル名。
        lam: 再ランキングの新規性重み。
        split_data: 利用者ごとの入力・評価データ。
        training_data: train評価から作成した統計。
        predictions: _model_predictions()の戻り値。
        loaded_model: MF/NCFの読込済みモデル。その他はNone。

    Returns:
        (サマリー辞書, 利用者ごとの詳細辞書のリスト)。
    """
    catalog = training_data["catalog"]
    train_counts = training_data["item_counts"]
    predictive = model_name in {"mean", "mf", "ncf"}
    pooled_preds: list[float] = []
    pooled_trues: list[float] = []
    macro_pairs: list[tuple[list[float], list[float]]] = []
    precision_top25: list[float] = []
    precision_fixed4: list[float] = []
    top25_positive_ratios: list[float] = []
    fixed4_positive_ratios: list[float] = []
    genre_counts: list[int] = []
    details: list[dict[str, Any]] = []
    top25_excluded_users = 0
    no_eval_users = 0
    excluded_eval_ratings = 0

    for user_id, user_data in split_data.items():
        input_items = user_data["_used_input"]
        original_input_items = user_data["input"]
        eval_items = user_data["eval"]
        eval_known = [
            (int(movie_id), int(rating))
            for movie_id, rating in eval_items
            if int(movie_id) in train_counts
        ]
        excluded_eval_ratings += len(eval_items) - len(eval_known)
        if not eval_known:
            no_eval_users += 1

        movie_ids, scores = predictions[user_id]
        score_by_movie = {
            int(movie_id): float(score)
            for movie_id, score in zip(movie_ids, scores, strict=True)
        }
        seen_ids = {int(movie_id) for movie_id, _ in original_input_items}
        candidates = [
            (
                int(movie_id),
                float(score),
                catalog[int(movie_id)][1],
            )
            for movie_id, score in score_by_movie.items()
            if int(movie_id) not in seen_ids
        ]
        recommended = rerank(candidates, lam=lam, k=10)
        recommended_ids = [movie_id for movie_id, _, _ in recommended]
        recommended_genres = [genres for _, _, genres in recommended]
        user_genre_count = genre_count(recommended_genres)
        genre_counts.append(user_genre_count)

        eval_ratings = [rating for _, rating in eval_known]
        eval_ids = [movie_id for movie_id, _ in eval_known]
        threshold = high_threshold(eval_ratings, "top25") if eval_ratings else None
        threshold4 = high_threshold(eval_ratings, "fixed4") if eval_ratings else None
        top25_relevant = (
            {
                movie_id
                for movie_id, rating in eval_known
                if threshold is not None and rating >= threshold
            }
            if threshold is not None
            else set()
        )
        fixed4_relevant = (
            {
                movie_id
                for movie_id, rating in eval_known
                if threshold4 is not None and rating >= threshold4
            }
            if threshold4 is not None
            else set()
        )
        p_top25 = precision_at_k(recommended_ids, top25_relevant, k=10)
        p_fixed4 = precision_at_k(recommended_ids, fixed4_relevant, k=10)
        precision_fixed4.append(p_fixed4)
        if eval_ratings:
            fixed4_positive_ratios.append(len(fixed4_relevant) / len(eval_ratings))
        if threshold is None:
            if eval_ratings and all(rating == eval_ratings[0] for rating in eval_ratings):
                top25_excluded_users += 1
        else:
            precision_top25.append(p_top25)
            top25_positive_ratios.append(len(top25_relevant) / len(eval_ratings))

        user_preds: list[float] = []
        user_trues: list[float] = []
        if predictive:
            user_preds = [score_by_movie[movie_id] for movie_id in eval_ids]
            user_trues = [float(rating) for rating in eval_ratings]
            if user_trues:
                pooled_preds.extend(user_preds)
                pooled_trues.extend(user_trues)
                macro_pairs.append((user_preds, user_trues))

        details.append(
            {
                "user_id": user_id,
                "評価用件数": len(eval_known),
                "高評価件数": len(top25_relevant),
                "Precision": p_top25 if threshold is not None else np.nan,
                "RMSE": (
                    rmse_pooled(user_preds, user_trues)
                    if predictive and user_trues
                    else np.nan
                ),
                "ジャンル数": user_genre_count,
                "入力5本のうち4点以上の本数": sum(
                    int(rating) >= 4 for _, rating in original_input_items
                ),
                "入力に使った本数": len(input_items),
                "入力の評価がすべて同じか": (
                    len({int(rating) for _, rating in original_input_items}) == 1
                ),
            }
        )

    if predictive and pooled_trues:
        rmse_pooled_value: float | str = rmse_pooled(pooled_preds, pooled_trues)
        rmse_macro_value: float | str = rmse_macro(macro_pairs)
    elif predictive:
        rmse_pooled_value = "N/A"
        rmse_macro_value = "N/A"
    else:
        rmse_pooled_value = "N/A"
        rmse_macro_value = "N/A"

    catalog_movies = set(catalog)
    train_movies = set(train_counts)
    summary = {
        "model": model_name,
        "lam": lam,
        "利用者数": len(split_data),
        "評価対象利用者数": len(macro_pairs) if predictive else len(split_data) - no_eval_users,
        "RMSE pooled": rmse_pooled_value,
        "RMSE macro": rmse_macro_value,
        "Precision@10 top25": (
            float(np.mean(precision_top25)) if precision_top25 else "N/A"
        ),
        "Precision@10 fixed4": (
            float(np.mean(precision_fixed4)) if precision_fixed4 else "N/A"
        ),
        "ジャンル数平均": float(np.mean(genre_counts)) if genre_counts else 0.0,
        "top25除外利用者数": top25_excluded_users,
        "評価なし利用者数": no_eval_users,
        "高評価割合 top25 平均": (
            float(np.mean(top25_positive_ratios)) if top25_positive_ratios else "N/A"
        ),
        "高評価割合 top25 中央値": (
            float(np.median(top25_positive_ratios)) if top25_positive_ratios else "N/A"
        ),
        "高評価割合 fixed4 平均": (
            float(np.mean(fixed4_positive_ratios)) if fixed4_positive_ratios else "N/A"
        ),
        "高評価割合 fixed4 中央値": (
            float(np.median(fixed4_positive_ratios)) if fixed4_positive_ratios else "N/A"
        ),
        "train外映画数": len(catalog_movies - train_movies),
        "除外評価件数": excluded_eval_ratings,
        "入力から除いた映画の延べ件数": training_data["input_movies_removed"],
        "入力が減った利用者数": training_data["users_with_reduced_input"],
        "入力3本未満で対象外にした利用者数": training_data[
            "users_excluded_for_short_input"
        ],
    }
    return summary, details


def run_evaluation(
    split_path: Path,
    model_name: str = "ncf",
    lam: float = 0.25,
    limit: int | None = None,
    all_models: bool = False,
    confirm_test: bool = True,
) -> list[dict[str, Any]]:
    """指定分割を評価して表・CSVを保存する。

    Args:
        split_path: 評価対象の分割パス。
        model_name: 個別実行するモデル名。
        lam: 個別実行時の再ランキング重み。
        limit: 先頭N人だけを評価する上限。
        all_models: --all相当なら全ベースラインとNCFの全λを実行する。
        confirm_test: test評価前に確認を表示するか。

    Returns:
        表示・保存したサマリー辞書のリスト。test確認でnoの場合は空リスト。
    """
    resolved, split_name, is_test = _resolve_split_path(split_path)
    if is_test and not all_models:
        print("testは --all でだけ実行できます")
        return []
    if is_test and confirm_test:
        try:
            answer = input("testの評価は2回までです。続けますか? (yes/no) ")
        except EOFError:
            answer = "no"
        if answer.strip().lower() != "yes":
            return []

    split_name, split_data = load_split_data(resolved, limit)
    if not split_data:
        raise ValueError("評価対象の利用者がありません。")
    training_data = load_training_data()
    movie_to_index, _, _, _ = build_id_maps(training_data["ratings"])
    input_movies_removed = 0
    users_with_reduced_input = 0
    users_excluded_for_short_input = 0
    eligible_split_data: dict[int, dict[str, Any]] = {}
    for user_id, user_data in split_data.items():
        original_input = user_data["input"]
        used_input = [
            (movie_id, rating)
            for movie_id, rating in original_input
            if movie_id in movie_to_index
        ]
        removed_count = len(original_input) - len(used_input)
        input_movies_removed += removed_count
        if removed_count:
            users_with_reduced_input += 1
        if len(used_input) < 3:
            users_excluded_for_short_input += 1
            continue
        eligible_split_data[user_id] = {
            **user_data,
            "_used_input": used_input,
        }
    if not eligible_split_data:
        raise ValueError("対応表にある入力映画が3本以上の利用者がいません。")
    training_data["movie_to_index"] = movie_to_index
    training_data["input_movies_removed"] = input_movies_removed
    training_data["users_with_reduced_input"] = users_with_reduced_input
    training_data["users_excluded_for_short_input"] = users_excluded_for_short_input
    split_data = eligible_split_data

    if all_models:
        configurations = [
            (name, 0.0) for name in ("mean", "popular", "random", "mf")
        ] + [("ncf", value) for value in ALL_NCF_LAMBDAS]
    else:
        configurations = [(model_name, lam)]

    prepared: dict[str, tuple[dict[int, tuple[np.ndarray, np.ndarray]], LoadedModel | None]] = {}
    for current_model in dict.fromkeys(name for name, _ in configurations):
        prepared[current_model] = _model_predictions(
            current_model,
            split_data,
            training_data,
            seed=SEED,
        )

    summaries: list[dict[str, Any]] = []
    detail_outputs: list[tuple[str, float, list[dict[str, Any]]]] = []
    for current_model, current_lam in configurations:
        user_predictions, loaded_model = prepared[current_model]
        summary, details = _evaluate_configuration(
            current_model,
            current_lam,
            split_data,
            training_data,
            user_predictions,
            loaded_model,
        )
        summary["split"] = split_name
        summaries.append(summary)
        if split_name.lower().startswith("dev"):
            detail_outputs.append((current_model, current_lam, details))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    result_model = "all" if all_models else model_name
    result_lam = "all" if all_models else f"{lam:g}"
    summary_path = (
        RESULTS_DIR
        / f"{split_name}_{result_model}_lam_{result_lam}_{timestamp}.csv"
    )
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(summary_path, index=False, encoding="utf-8")

    for detail_model, detail_lam, detail_rows in detail_outputs:
        detail_path = (
            RESULTS_DIR
            / f"{split_name}_details_{detail_model}_lam_{detail_lam:g}_{timestamp}.csv"
        )
        pd.DataFrame(detail_rows).to_csv(detail_path, index=False, encoding="utf-8")

    print(summary_frame.to_string(index=False))
    print(f"CSV: {summary_path}")
    if is_test:
        model_lambdas = ", ".join(
            f"{name}:lam={value:g}" for name, value in configurations
        )
        log_path = RESULTS_DIR / "test_runs.log"
        run_time = datetime.now().astimezone().isoformat(timespec="seconds")
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(f"{run_time}\t{model_lambdas}\n")
    return summaries


def main(argv: Sequence[str] | None = None) -> None:
    """CLIから評価を実行する。

    Args:
        argv: CLI引数。Noneならsys.argvを使う。

    Returns:
        None。
    """
    args = parse_args(argv)
    _, _, is_test = _resolve_split_path(args.split)
    if is_test and not args.all:
        print("testは --all でだけ実行できます")
        return
    run_evaluation(
        args.split,
        model_name=args.model,
        lam=args.lam,
        limit=args.limit,
        all_models=args.all,
    )


if __name__ == "__main__":
    main()
