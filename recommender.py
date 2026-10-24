from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn

from data_utils import load_movies
from models import MF, NCF


MODELS_DIR = Path("models")
MOVIES_PATH = Path("data/ml-1m/movies.dat")


@dataclass
class LoadedModel:
    """Checkpoint、モデル、映画カタログをまとめた推論用データ。"""

    name: str
    model: MF | NCF
    user_to_index: dict[int, int]
    index_to_user: dict[int, int]
    movie_to_index: dict[int, int]
    index_to_movie: dict[int, int]
    movie_catalog: dict[int, tuple[str, list[str]]]
    item_counts: np.ndarray
    train_user_vec_mean: torch.Tensor


@lru_cache(maxsize=2)
def load_model(name: str) -> LoadedModel:
    """models/からMFまたはNCFをCPUへ読み込む。checkpointごとに一度だけ実行する。

    Args:
        name: 読み込むモデル名。"mf"または"ncf"。

    Returns:
        モデル、ID対応表、映画カタログ、映画評価件数、fold-in初期値を
        持つLoadedModel。

    Raises:
        ValueError: nameがmf/ncfでない場合。
        FileNotFoundError: checkpointまたはmovies.datがない場合。
    """
    if name not in {"mf", "ncf"}:
        raise ValueError("モデル名は 'mf' または 'ncf' を指定してください。")

    checkpoint_path = MODELS_DIR / f"{name}.pt"
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    user_to_index = checkpoint["user_to_index"]
    index_to_user = checkpoint["index_to_user"]
    movie_to_index = checkpoint["movie_to_index"]
    index_to_movie = checkpoint["index_to_movie"]
    global_mean = float(checkpoint["global_mean"])

    if name == "mf":
        model: MF | NCF = MF(
            len(user_to_index),
            len(movie_to_index),
            int(checkpoint["dim"]),
            global_mean,
        )
    else:
        model = NCF(
            len(user_to_index),
            len(movie_to_index),
            int(checkpoint["dim"]),
            global_mean,
            float(checkpoint["dropout"]),
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.cpu()
    model.eval()

    movies = load_movies(MOVIES_PATH)
    movie_catalog = {
        int(row.movie_id): (str(row.title), list(row.genres))
        for row in movies.itertuples(index=False)
    }
    return LoadedModel(
        name=name,
        model=model,
        user_to_index=user_to_index,
        index_to_user=index_to_user,
        movie_to_index=movie_to_index,
        index_to_movie=index_to_movie,
        movie_catalog=movie_catalog,
        item_counts=checkpoint["item_counts"].cpu().numpy().copy(),
        train_user_vec_mean=checkpoint["train_user_vec_mean"].detach().cpu().clone(),
    )


def validate_inputs(
    items: Sequence[tuple[int, int]],
    model_name: str = "ncf",
    loaded_model: LoadedModel | None = None,
) -> None:
    """入力映画と評価を検証し、問題があれば日本語のValueErrorを送出する。

    Args:
        items: (movie_id, rating)形式の入力列。
        model_name: 映画ID対応表を確認するモデル名。
        loaded_model: すでに読み込んだモデル情報。省略時はmodel_nameで読む。

    Returns:
        入力がすべて有効ならNone。

    Raises:
        ValueError: 本数不足、重複、カタログ・対応表にない映画、
            または1〜5の整数でない評価がある場合。
    """
    if len(items) < 3:
        raise ValueError("映画を3本以上入力してください。")

    seen_movie_ids: set[int] = set()
    for item in items:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError("各入力は (映画ID, 評価) の形式で指定してください。")
        movie_id, rating = item
        if isinstance(movie_id, bool) or not isinstance(movie_id, Integral):
            raise ValueError("映画IDは整数で指定してください。")
        movie_id = int(movie_id)
        if movie_id in seen_movie_ids:
            raise ValueError("同じ映画を重複して入力できません。")
        seen_movie_ids.add(movie_id)

        if isinstance(rating, bool) or not isinstance(rating, Integral) or not 1 <= rating <= 5:
            raise ValueError("評価は1〜5の整数で指定してください。")

    model_data = loaded_model or load_model(model_name)
    for movie_id, _ in items:
        movie_id = int(movie_id)
        if movie_id not in model_data.movie_catalog:
            raise ValueError(f"映画ID {movie_id} は映画カタログにありません。")
        if movie_id not in model_data.movie_to_index:
            raise ValueError(
                f"映画ID {movie_id} は学習データの映画ID対応表にありません。"
            )


def to_movie_index(
    items: Sequence[tuple[int, int]],
    model: LoadedModel,
) -> tuple[np.ndarray, np.ndarray]:
    """movie_idを連番indexへ変換し、評価値とともに返す。

    Args:
        items: (movie_id, rating)形式の入力列。
        model: load_model()が返すモデル情報。

    Returns:
        映画indexのint64配列と評価値のfloat32配列。
    """
    item_idx = np.asarray(
        [model.movie_to_index[int(movie_id)] for movie_id, _ in items],
        dtype=np.int64,
    )
    ratings = np.asarray([rating for _, rating in items], dtype=np.float32)
    return item_idx, ratings


def fold_in_user(
    model: LoadedModel,
    item_idx: Sequence[int] | np.ndarray | torch.Tensor,
    ratings: Sequence[float] | np.ndarray | torch.Tensor,
    steps: int = 50,
    lr: float = 0.05,
    l2: float = 0.01,
) -> tuple[dict[str, torch.Tensor], float, float]:
    """モデル本体を固定し、入力評価に合わせた利用者パラメータを最適化する。

    Args:
        model: load_model()が返すMFまたはNCFモデル情報。
        item_idx: 入力映画の連番index列。
        ratings: 入力評価値列。
        steps: Adamの更新回数。
        lr: Adamの学習率。
        l2: 初期値からの二乗距離に掛ける正則化係数。

    Returns:
        (利用者パラメータ辞書, 初期値での損失, 最適化後の損失)。
        MF辞書にはuser_vecとuser_bias、NCF辞書にはuser_vecが含まれる。

    Raises:
        ValueError: 入力長が異なる、空、または最適化設定が不正な場合。
    """
    if steps <= 0 or lr <= 0 or l2 < 0:
        raise ValueError("stepsとlrは正、l2は0以上で指定してください。")
    item_idx_tensor = torch.as_tensor(item_idx, dtype=torch.long, device="cpu")
    rating_tensor = torch.as_tensor(ratings, dtype=torch.float32, device="cpu")
    if item_idx_tensor.ndim != 1 or rating_tensor.ndim != 1:
        raise ValueError("映画indexと評価は1次元で指定してください。")
    if len(item_idx_tensor) == 0 or len(item_idx_tensor) != len(rating_tensor):
        raise ValueError("映画indexと評価は同じ件数の空でない列にしてください。")

    network = model.model
    network.eval()
    for parameter in network.parameters():
        parameter.requires_grad_(False)

    user_vec_initial = model.train_user_vec_mean.detach().clone()
    user_vec = nn.Parameter(user_vec_initial.clone())
    optimized_parameters: list[nn.Parameter] = [user_vec]
    if model.name == "mf":
        user_bias_initial = torch.zeros((), dtype=torch.float32)
        user_bias = nn.Parameter(user_bias_initial.clone())
        optimized_parameters.append(user_bias)
    else:
        user_bias_initial = None
        user_bias = None

    def calculate_loss() -> torch.Tensor:
        if model.name == "mf":
            assert user_bias is not None and user_bias_initial is not None
            predictions = network.predict_with_user_params(
                user_vec,
                user_bias,
                item_idx_tensor,
            )
            distance = torch.sum((user_vec - user_vec_initial) ** 2)
            distance += torch.sum((user_bias - user_bias_initial) ** 2)
        else:
            predictions = network.predict_with_user_params(user_vec, item_idx_tensor)
            distance = torch.sum((user_vec - user_vec_initial) ** 2)
        return torch.mean((predictions - rating_tensor) ** 2) + l2 * distance

    initial_loss = float(calculate_loss().detach())
    optimizer = torch.optim.Adam(optimized_parameters, lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = calculate_loss()
        loss.backward()
        optimizer.step()
    final_loss = float(calculate_loss().detach())

    user_params = {"user_vec": user_vec.detach().clone()}
    if user_bias is not None:
        user_params["user_bias"] = user_bias.detach().clone()
    return user_params, initial_loss, final_loss


def predict_ratings(
    model: LoadedModel,
    user_params: dict[str, torch.Tensor],
    clip: bool = True,
) -> np.ndarray:
    """全映画の評価を予測し、必要なら1〜5にclipした配列を返す。

    Args:
        model: load_model()が返すモデル情報。
        user_params: fold_in_user()が返す利用者パラメータ辞書。
        clip: Trueなら予測値を1〜5にclipする。

    Returns:
        映画index順のfloat32予測評価配列。clip=Falseなら生の値を返す。
        推論時はmodelがeval()であること。
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
    if clip:
        predictions = torch.clamp(predictions, min=1.0, max=5.0)
    return predictions.cpu().numpy()


def rerank(
    candidates: Sequence[tuple[int, float, Sequence[str]]],
    lam: float,
    k: int = 10,
) -> list[tuple[int, float, Sequence[str]]]:
    """ジャンル新規性を加点し、上位k件を貪欲に再ランキングする。

    Args:
        candidates: (movie_id, pred, genres)形式の候補列。
        lam: 新規性に掛ける非負の重み。
        k: 返す上位件数。

    Returns:
        選択順の最大k候補。score同点時はmovie_id昇順。

    Raises:
        ValueError: lamが負、またはkが正の整数でない場合。
    """
    if lam < 0:
        raise ValueError("lamは0以上で指定してください。")
    if isinstance(k, bool) or not isinstance(k, Integral) or k <= 0:
        raise ValueError("kは正の整数で指定してください。")

    remaining = list(candidates)
    selected: list[tuple[int, float, Sequence[str]]] = []
    selected_genres: set[str] = set()
    for _ in range(min(int(k), len(remaining))):
        best_candidate: tuple[int, float, Sequence[str]] | None = None
        best_score = float("-inf")
        for candidate in remaining:
            movie_id, pred, genres = candidate
            genre_set = set(genres)
            novelty = (
                len(genre_set - selected_genres) / len(genre_set)
                if genre_set
                else 0.0
            )
            score = float(pred) + lam * novelty
            if (
                best_candidate is None
                or score > best_score
                or (score == best_score and movie_id < best_candidate[0])
            ):
                best_candidate = candidate
                best_score = score
        assert best_candidate is not None
        selected.append(best_candidate)
        remaining.remove(best_candidate)
        selected_genres.update(best_candidate[2])
    return selected


def _raw_predictions(
    model: LoadedModel,
    user_params: dict[str, torch.Tensor],
) -> np.ndarray:
    """clip前の全映画予測を映画index順に返す。

    Args:
        model: load_model()が返すモデル情報。
        user_params: fold_in_user()が返す利用者パラメータ辞書。

    Returns:
        映画index順のfloat32予測配列。
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


def recommend(
    items: Sequence[tuple[int, int]],
    lam: float = 0.25,
    model_name: str = "ncf",
    debug: bool = False,
) -> list[tuple[str, float, list[str]]]:
    """入力映画評価から、視聴済みを除いた上位映画を推薦する。

    Args:
        items: (movie_id, rating)形式の入力評価列。
        lam: ジャンル新規性に掛ける重み。
        model_name: "mf"または"ncf"。
        debug: 中間のindex、評価範囲、損失、上位生予測を表示するか。

    Returns:
        (title, clip済みpred, genres)形式の最大10件の推薦リスト。
    """
    model = load_model(model_name)
    validate_inputs(items, model_name=model_name, loaded_model=model)
    item_idx, ratings = to_movie_index(items, model)
    user_params, initial_loss, final_loss = fold_in_user(
        model,
        item_idx,
        ratings,
    )
    raw_predictions = predict_ratings(model, user_params, clip=False)
    predictions = np.clip(raw_predictions, 1.0, 5.0)

    if debug:
        print(
            f"item_idx: shape={item_idx.shape}, dtype={item_idx.dtype}; "
            f"ratings: shape={ratings.shape}, dtype={ratings.dtype}"
        )
        print(f"ratings min={ratings.min()}, max={ratings.max()}")
        print(f"fold-in loss: initial={initial_loss:.6f}, final={final_loss:.6f}")
        top_indices = sorted(
            range(len(raw_predictions)),
            key=lambda index: (
                -float(raw_predictions[index]),
                model.index_to_movie[index],
            ),
        )[:5]
        top_raw = [
            (model.index_to_movie[index], float(raw_predictions[index]))
            for index in top_indices
        ]
        print(f"raw prediction top 5 (movie_id, pred): {top_raw}")

    seen_movie_ids = {int(movie_id) for movie_id, _ in items}
    candidates = []
    for index, movie_id in model.index_to_movie.items():
        if movie_id in seen_movie_ids or model.item_counts[index] == 0:
            continue
        movie_data = model.movie_catalog.get(movie_id)
        if movie_data is None:
            continue
        title, genres = movie_data
        candidates.append((movie_id, float(raw_predictions[index]), genres))

    selected = rerank(candidates, lam, k=10)
    return [
        (
            model.movie_catalog[movie_id][0],
            float(predictions[model.movie_to_index[movie_id]]),
            list(genres),
        )
        for movie_id, _, genres in selected
    ]
