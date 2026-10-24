import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_utils import load_ratings


SEED = 42
INPUT_COUNT = 5
DATA_DIR = Path("data/ml-1m")
SPLITS_DIR = Path("splits")


def split_users(
    user_ids: list[int],
    seed: int = SEED,
) -> tuple[list[int], list[int], list[int]]:
    """利用者IDをseed付きでtrain、dev、testに分割する。

    Args:
        user_ids: 分割対象の利用者ID。
        seed: 利用者の並べ替えに使う乱数seed。

    Returns:
        train、dev、testの利用者IDリスト。各リスト内の順序は
        seedによる並べ替え順。
    """
    ordered_ids = np.asarray(sorted(user_ids), dtype=np.int64)
    shuffled_ids = np.random.RandomState(seed).permutation(ordered_ids).tolist()

    train_end = int(len(shuffled_ids) * 0.8)
    dev_end = int(len(shuffled_ids) * 0.9)
    return (
        shuffled_ids[:train_end],
        shuffled_ids[train_end:dev_end],
        shuffled_ids[dev_end:],
    )


def split_user_ratings(
    ratings: pd.DataFrame,
    user_ids: list[int],
    rng: np.random.RandomState,
) -> dict[int, dict[str, list[list[int]]]]:
    """利用者ごとの評価から入力用5件と評価用の残りを選ぶ。

    Args:
        ratings: user_id, movie_id, rating列を含む評価DataFrame。
        user_ids: 処理対象の利用者ID。昇順で処理する。
        rng: 入力用評価の選択に使う乱数生成器。

    Returns:
        利用者IDをキーとし、inputとevalを持つ辞書。各値は
        [movie_id, rating]形式の整数リスト。
    """
    split: dict[int, dict[str, list[list[int]]]] = {}
    ratings_by_user = ratings.groupby("user_id", sort=False)
    for user_id in sorted(user_ids):
        user_ratings = (
            ratings_by_user.get_group(user_id)[["movie_id", "rating"]]
            .sort_values("movie_id")
            .to_numpy(dtype=np.int64)
        )
        assert len(user_ratings) >= INPUT_COUNT, (
            f"user_id={user_id} has fewer than {INPUT_COUNT} ratings"
        )

        input_indices = np.sort(
            rng.choice(len(user_ratings), size=INPUT_COUNT, replace=False)
        )
        is_input = np.zeros(len(user_ratings), dtype=bool)
        is_input[input_indices] = True
        split[int(user_id)] = {
            "input": user_ratings[is_input].tolist(),
            "eval": user_ratings[~is_input].tolist(),
        }
    return split


def main() -> None:
    """利用者分割とdev/testの入力・評価データを作成し、件数を表示する。"""
    ratings = load_ratings(DATA_DIR / "ratings.dat")
    user_ids = sorted(int(user_id) for user_id in ratings["user_id"].unique())
    train_users, dev_users, test_users = split_users(user_ids)

    train_set = set(train_users)
    dev_set = set(dev_users)
    test_set = set(test_users)
    assert not train_set.intersection(dev_set)
    assert not train_set.intersection(test_set)
    assert not dev_set.intersection(test_set)

    assert (len(train_users), len(dev_users), len(test_users)) == (4832, 604, 604)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for name, users in (
        ("train", train_users),
        ("dev", dev_users),
        ("test", test_users),
    ):
        (SPLITS_DIR / f"{name}_users.txt").write_text(
            "".join(f"{user_id}\n" for user_id in users),
            encoding="utf-8",
        )

    rng = np.random.RandomState(SEED)
    all_eval_split = split_user_ratings(
        ratings,
        sorted(dev_users + test_users),
        rng,
    )
    dev_split = {
        user_id: all_eval_split[user_id]
        for user_id in sorted(dev_users)
    }
    test_split = {
        user_id: all_eval_split[user_id]
        for user_id in sorted(test_users)
    }
    for name, split in (("dev", dev_split), ("test", test_split)):
        with (SPLITS_DIR / f"{name}_inputs.json").open(
            "w", encoding="utf-8"
        ) as output:
            json.dump(split, output, ensure_ascii=False, indent=2)
            output.write("\n")

    train_rating_count = int(ratings["user_id"].isin(train_set).sum())
    dev_eval_count = sum(len(user_data["eval"]) for user_data in dev_split.values())
    test_eval_count = sum(len(user_data["eval"]) for user_data in test_split.values())
    print(f"train: {len(train_users)} users, {train_rating_count} ratings")
    print(f"dev: {len(dev_users)} users, {dev_eval_count} eval ratings")
    print(f"test: {len(test_users)} users, {test_eval_count} eval ratings")


if __name__ == "__main__":
    main()
