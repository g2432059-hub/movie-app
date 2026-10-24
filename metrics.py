from collections.abc import Iterable, Sequence

import numpy as np


def rmse_pooled(preds: Sequence[float], trues: Sequence[float]) -> float:
    """全評価をまとめてRMSEを計算する。

    Args:
        preds: 予測値の数値列。
        trues: 正解値の数値列。predsと同じ長さであること。

    Returns:
        全評価をまとめたRMSE（float）。

    Raises:
        ValueError: 入力が空、または予測値と正解値の長さが異なる場合。
    """
    if len(preds) == 0 or len(trues) == 0:
        raise ValueError("preds and trues must not be empty")
    if len(preds) != len(trues):
        raise ValueError("preds and trues must have the same length")

    errors = np.asarray(preds, dtype=float) - np.asarray(trues, dtype=float)
    return float(np.sqrt(np.mean(errors**2)))


def rmse_macro(
    list_of_pairs: Sequence[tuple[Sequence[float], Sequence[float]]],
) -> float:
    """利用者ごとのRMSEを計算し、その平均を返す。

    Args:
        list_of_pairs: 利用者ごとの(preds, trues)の列。

    Returns:
        利用者ごとのRMSEの算術平均（float）。

    Raises:
        ValueError: 入力が空、またはいずれかの利用者のデータが空の場合。
    """
    if len(list_of_pairs) == 0:
        raise ValueError("list_of_pairs must not be empty")
    return float(np.mean([rmse_pooled(preds, trues) for preds, trues in list_of_pairs]))


def high_threshold(
    eval_ratings: Sequence[float],
    mode: str,
) -> float | None:
    """評価値から高評価の閾値を求める。

    Args:
        eval_ratings: 利用者の評価用評価値の数値列。
        mode: "top25"（75パーセンタイル）または"fixed4"。

    Returns:
        top25では閾値（全評価が同じならNone）、fixed4では4（float）。

    Raises:
        ValueError: 評価値が空、またはmodeが未対応の場合。
    """
    if len(eval_ratings) == 0:
        raise ValueError("eval_ratings must not be empty")
    if mode == "fixed4":
        return 4.0
    if mode != "top25":
        raise ValueError(f"unsupported mode: {mode}")

    values = np.asarray(eval_ratings, dtype=float)
    if np.all(values == values[0]):
        return None
    return float(np.percentile(values, 75, method="lower"))


def precision_at_k(
    recommended_ids: Sequence[int],
    relevant_ids: Iterable[int],
    k: int = 10,
) -> float:
    """推薦上位k件に対する適合率を計算する。

    Args:
        recommended_ids: 順位順の推薦ID列。
        relevant_ids: 関連IDの集合または列。
        k: 評価する上位件数。分母にも使用する正の整数。

    Returns:
        上位k件に含まれる関連ID数をkで割った値（float）。
        relevant_idsが空の場合は0.0。

    Raises:
        ValueError: kが正の整数でない場合。
    """
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    relevant = set(relevant_ids)
    hits = sum(item in relevant for item in recommended_ids[:k])
    return hits / k


def genre_count(list_of_genre_lists: Iterable[Iterable[str]]) -> int:
    """ジャンルリスト全体に含まれる異なるジャンル数を数える。

    Args:
        list_of_genre_lists: 各映画のジャンル名リスト。

    Returns:
        重複を除いたジャンルの種類数（int）。入力が空なら0。
    """
    return len({genre for genres in list_of_genre_lists for genre in genres})
