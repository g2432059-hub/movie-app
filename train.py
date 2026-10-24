import argparse
import copy
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data_utils import build_id_maps, load_ratings
from metrics import rmse_pooled
from models import MF, NCF


DATA_DIR = Path("data/ml-1m")
SPLITS_DIR = Path("splits")
MODELS_DIR = Path("models")
EARLY_STOPPING_PATIENCE = 3
VAL_FRACTION = 0.05


def load_train_ratings(
    ratings_path: str | Path,
    train_users_path: str | Path,
) -> pd.DataFrame:
    """ratings.datからtrain_users.txtに含まれる利用者の評価だけを返す。

    Args:
        ratings_path: ratings.datのファイルパス。
        train_users_path: train利用者IDを1行ずつ記したファイルパス。

    Returns:
        user_id, movie_id, rating, timestamp列を持つtrain評価DataFrame。
    """
    train_users = {
        int(line)
        for line in Path(train_users_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    ratings = load_ratings(ratings_path)
    return ratings.loc[ratings["user_id"].isin(train_users)].copy()


def split_train_val(
    ratings: pd.DataFrame,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """train利用者評価の5%をseed固定で検証用に分ける。

    Args:
        ratings: train利用者の評価DataFrame。
        seed: 検証行の選択に使う乱数seed。

    Returns:
        (学習用DataFrame, 検証用DataFrame)の組。
    """
    if len(ratings) < 2:
        raise ValueError("at least two ratings are required to create a validation set")

    rng = np.random.RandomState(seed)
    val_count = max(1, int(len(ratings) * VAL_FRACTION))
    val_count = min(val_count, len(ratings) - 1)
    val_indices = np.sort(rng.choice(len(ratings), size=val_count, replace=False))
    is_val = np.zeros(len(ratings), dtype=bool)
    is_val[val_indices] = True
    train = ratings.iloc[np.flatnonzero(~is_val)].reset_index(drop=True)
    val = ratings.iloc[val_indices].reset_index(drop=True)
    return train, val


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """コマンドライン引数を解析する。

    Args:
        argv: 引数列。Noneならsys.argvを使う。

    Returns:
        学習設定を格納したargparse.Namespace。
    """
    parser = argparse.ArgumentParser(
        description="MovieLensのmean、MF、NCFモデルを学習する"
    )
    parser.add_argument("--model", choices=("mean", "mf", "ncf"), default="mf")
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-5)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shrink", type=float, default=10)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args(argv)
    if args.dim <= 0 or args.epochs <= 0 or args.batch <= 0:
        parser.error("--dim, --epochs, and --batch must be positive")
    if args.lr <= 0 or args.wd < 0 or args.shrink < 0:
        parser.error("--lr must be positive and --wd/--shrink must be non-negative")
    if not 0.0 <= args.dropout < 1.0:
        parser.error("--dropout must be in [0, 1)")
    return args


def _encode_ratings(
    ratings: pd.DataFrame,
    user_to_index: dict[int, int],
    movie_to_index: dict[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """評価DataFrameをindexと評価値のTensorに変換する。

    Args:
        ratings: user_id, movie_id, rating列を含む評価DataFrame。
        user_to_index: user_idから連番indexへの辞書。
        movie_to_index: movie_idから連番indexへの辞書。

    Returns:
        user index、movie index、ratingの各1次元Tensor。
    """
    user_indices = ratings["user_id"].map(user_to_index).to_numpy(
        dtype=np.int64,
        copy=True,
    )
    movie_indices = ratings["movie_id"].map(movie_to_index).to_numpy(
        dtype=np.int64,
        copy=True,
    )
    values = ratings["rating"].to_numpy(dtype=np.float32, copy=True)
    return (
        torch.from_numpy(user_indices),
        torch.from_numpy(movie_indices),
        torch.from_numpy(values),
    )


def _validate_mappings(
    train_ratings: pd.DataFrame,
    val_ratings: pd.DataFrame,
    user_to_index: dict[int, int],
    movie_to_index: dict[int, int],
) -> None:
    """学習・検証評価の全IDが対応表に含まれることを検証する。

    Args:
        train_ratings: 学習評価DataFrame。
        val_ratings: 検証評価DataFrame。
        user_to_index: user_idからindexへの辞書。
        movie_to_index: movie_idからindexへの辞書。

    Returns:
        None。

    Raises:
        ValueError: user_idまたはmovie_idが対応表にない場合。
    """
    all_ratings = pd.concat([train_ratings, val_ratings], ignore_index=True)
    missing_users = set(all_ratings["user_id"]) - user_to_index.keys()
    missing_movies = set(all_ratings["movie_id"]) - movie_to_index.keys()
    if missing_users or missing_movies:
        raise ValueError(
            "ID maps must be built from all train-user ratings, including validation"
        )


def _make_checkpoint(
    model_name: str,
    model_state: dict[str, torch.Tensor] | None,
    user_to_index: dict[int, int],
    index_to_user: dict[int, int],
    movie_to_index: dict[int, int],
    index_to_movie: dict[int, int],
    global_mean: float,
    item_counts: np.ndarray,
    train_user_vec_mean: torch.Tensor | None,
    **model_data: object,
) -> dict[str, object]:
    """学習済みモデルと推論用メタデータを保存形式にまとめる。

    Args:
        model_name: "mean"または"mf"。
        model_state: MFのstate_dict。meanモデルではNone。
        user_to_index: 利用者IDからindexへの辞書。
        index_to_user: indexから利用者IDへの辞書。
        movie_to_index: 映画IDからindexへの辞書。
        index_to_movie: indexから映画IDへの辞書。
        global_mean: 学習評価の全体平均。
        item_counts: 学習用映画評価件数の配列。
        train_user_vec_mean: train利用者ベクトルの平均。meanではNone。
        **model_data: モデル固有の推論用値。

    Returns:
        torch.saveに渡すcheckpoint辞書。
    """
    return {
        "model": model_name,
        "model_state_dict": model_state,
        "user_to_index": user_to_index,
        "index_to_user": index_to_user,
        "movie_to_index": movie_to_index,
        "index_to_movie": index_to_movie,
        "global_mean": float(global_mean),
        "item_counts": torch.from_numpy(item_counts.astype(np.int64)),
        "train_user_vec_mean": train_user_vec_mean,
        **model_data,
    }


def _save_checkpoint(checkpoint: dict[str, object], model_name: str) -> Path:
    """モデルcheckpointをmodels/{model_name}.ptへ保存する。

    Args:
        checkpoint: モデル重みと推論用メタデータ。
        model_name: 保存ファイル名に使うモデル名。

    Returns:
        保存したcheckpointのPath。
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{model_name}.pt"
    torch.save(checkpoint, path)
    return path


def _run_mean(
    train_ratings: pd.DataFrame,
    val_ratings: pd.DataFrame,
    movie_to_index: dict[int, int],
    index_to_movie: dict[int, int],
    user_to_index: dict[int, int],
    index_to_user: dict[int, int],
    shrink: float,
) -> None:
    """縮約映画平均を計算し、検証値とcheckpointを出力する。

    Args:
        train_ratings: 学習に使う評価DataFrame。
        val_ratings: 検証評価DataFrame。
        movie_to_index: 映画IDからindexへの辞書。
        index_to_movie: indexから映画IDへの辞書。
        user_to_index: 利用者IDからindexへの辞書。
        index_to_user: indexから利用者IDへの辞書。
        shrink: 映画平均の縮約強度。

    Returns:
        None。
    """
    global_mean = float(train_ratings["rating"].mean())
    grouped = train_ratings.groupby("movie_id")["rating"].agg(["sum", "count"])
    item_counts = np.zeros(len(movie_to_index), dtype=np.int64)
    item_means = np.full(len(movie_to_index), global_mean, dtype=np.float32)
    for movie_id, row in grouped.iterrows():
        item_idx = movie_to_index[int(movie_id)]
        item_counts[item_idx] = int(row["count"])
        item_means[item_idx] = (
            float(row["sum"]) + shrink * global_mean
        ) / (int(row["count"]) + shrink)

    val_movie_indices = val_ratings["movie_id"].map(movie_to_index).to_numpy(dtype=np.int64)
    val_trues = val_ratings["rating"].to_numpy(dtype=float)
    val_preds = item_means[val_movie_indices].astype(float)
    unseen_val_count = int(np.count_nonzero(item_counts[val_movie_indices] == 0))
    train_movie_indices = train_ratings["movie_id"].map(movie_to_index).to_numpy(dtype=np.int64)
    train_trues = train_ratings["rating"].to_numpy(dtype=float)
    train_preds = item_means[train_movie_indices].astype(float)

    train_mse = float(np.mean((train_preds - train_trues) ** 2))
    val_rmse = rmse_pooled(val_preds, val_trues)
    print(
        f"epoch 1/1: train MSE={train_mse:.6f}, "
        f"val RMSE={val_rmse:.6f}, val movies without train ratings={unseen_val_count}"
    )
    print("best epoch: 1; stopped epoch: 1 (closed-form mean model)")

    checkpoint = _make_checkpoint(
        "mean",
        None,
        user_to_index,
        index_to_user,
        movie_to_index,
        index_to_movie,
        global_mean,
        item_counts,
        None,
        item_means=torch.from_numpy(item_means),
        shrink=float(shrink),
    )
    print(f"saved: {_save_checkpoint(checkpoint, 'mean')}")


def _run_mf(
    train_ratings: pd.DataFrame,
    val_ratings: pd.DataFrame,
    user_to_index: dict[int, int],
    index_to_user: dict[int, int],
    movie_to_index: dict[int, int],
    index_to_movie: dict[int, int],
    dim: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
) -> None:
    """MFをAdamで学習し、最良val RMSEのcheckpointを保存する。

    Args:
        train_ratings: 学習に使う評価DataFrame。
        val_ratings: 検証評価DataFrame。
        user_to_index: 利用者IDからindexへの辞書。
        index_to_user: indexから利用者IDへの辞書。
        movie_to_index: 映画IDからindexへの辞書。
        index_to_movie: indexから映画IDへの辞書。
        dim: 埋め込み次元。
        epochs: 最大epoch数。
        learning_rate: Adamの学習率。
        weight_decay: Adamのweight decay。
        batch_size: ミニバッチサイズ。

    Returns:
        None。
    """
    global_mean = float(train_ratings["rating"].mean())
    model = MF(len(user_to_index), len(movie_to_index), dim, global_mean)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    train_tensors = _encode_ratings(train_ratings, user_to_index, movie_to_index)
    val_tensors = _encode_ratings(val_ratings, user_to_index, movie_to_index)
    train_loader = DataLoader(
        TensorDataset(*train_tensors),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(torch.initial_seed()),
    )

    best_val_rmse = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    stopped_epoch = 0
    val_users, val_items, val_trues = val_tensors

    for epoch in range(1, epochs + 1):
        model.train()
        train_squared_error = 0.0
        train_count = 0
        for user_idx, item_idx, ratings in train_loader:
            optimizer.zero_grad()
            predictions = model(user_idx, item_idx)
            loss = loss_fn(predictions, ratings)
            loss.backward()
            optimizer.step()
            batch_count = len(ratings)
            train_squared_error += float(loss.detach()) * batch_count
            train_count += batch_count
        train_mse = train_squared_error / train_count

        model.eval()
        with torch.no_grad():
            val_preds = model(val_users, val_items)
            val_rmse = rmse_pooled(
                val_preds.cpu().numpy(),
                val_trues.cpu().numpy(),
            )

        print(
            f"epoch {epoch}/{epochs}: train MSE={train_mse:.6f}, "
            f"val RMSE={val_rmse:.6f}"
        )
        stopped_epoch = epoch
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                break

    if best_state is None:
        raise RuntimeError("MF training did not produce a best checkpoint")
    model.load_state_dict(best_state)
    train_user_vec_mean = model.user_embedding.weight.detach().mean(dim=0).cpu()
    train_counts = train_ratings["movie_id"].value_counts()
    item_counts = np.zeros(len(movie_to_index), dtype=np.int64)
    for movie_id, count in train_counts.items():
        item_counts[movie_to_index[int(movie_id)]] = int(count)

    print(f"best epoch: {best_epoch}; stopped epoch: {stopped_epoch}")
    checkpoint = _make_checkpoint(
        "mf",
        best_state,
        user_to_index,
        index_to_user,
        movie_to_index,
        index_to_movie,
        global_mean,
        item_counts,
        train_user_vec_mean,
        dim=dim,
    )
    print(f"saved: {_save_checkpoint(checkpoint, 'mf')}")


def _run_ncf(
    train_ratings: pd.DataFrame,
    val_ratings: pd.DataFrame,
    user_to_index: dict[int, int],
    index_to_user: dict[int, int],
    movie_to_index: dict[int, int],
    index_to_movie: dict[int, int],
    dim: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    dropout: float,
) -> None:
    """NCFをAdamで学習し、最良val RMSEのcheckpointを保存する。

    Args:
        train_ratings: 学習に使う評価DataFrame。
        val_ratings: 検証評価DataFrame。
        user_to_index: 利用者IDからindexへの辞書。
        index_to_user: indexから利用者IDへの辞書。
        movie_to_index: 映画IDからindexへの辞書。
        index_to_movie: indexから映画IDへの辞書。
        dim: 埋め込み次元。
        epochs: 最大epoch数。
        learning_rate: Adamの学習率。
        weight_decay: Adamのweight decay。
        batch_size: ミニバッチサイズ。
        dropout: MLP隠れ層のDropout率。

    Returns:
        None。
    """
    global_mean = float(train_ratings["rating"].mean())
    model = NCF(
        len(user_to_index),
        len(movie_to_index),
        dim,
        global_mean,
        dropout,
    )
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    train_tensors = _encode_ratings(train_ratings, user_to_index, movie_to_index)
    val_tensors = _encode_ratings(val_ratings, user_to_index, movie_to_index)
    train_loader = DataLoader(
        TensorDataset(*train_tensors),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(torch.initial_seed()),
    )

    best_val_rmse = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    stopped_epoch = 0
    val_users, val_items, val_trues = val_tensors

    for epoch in range(1, epochs + 1):
        model.train()
        train_squared_error = 0.0
        train_count = 0
        for user_idx, item_idx, ratings in train_loader:
            optimizer.zero_grad()
            predictions = model(user_idx, item_idx)
            loss = loss_fn(predictions, ratings)
            loss.backward()
            optimizer.step()
            batch_count = len(ratings)
            train_squared_error += float(loss.detach()) * batch_count
            train_count += batch_count
        train_mse = train_squared_error / train_count

        model.eval()
        with torch.no_grad():
            val_preds = model(val_users, val_items)
            val_rmse = rmse_pooled(
                val_preds.cpu().numpy(),
                val_trues.cpu().numpy(),
            )

        print(
            f"epoch {epoch}/{epochs}: train MSE={train_mse:.6f}, "
            f"val RMSE={val_rmse:.6f}"
        )
        stopped_epoch = epoch
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                break

    if best_state is None:
        raise RuntimeError("NCF training did not produce a best checkpoint")
    model.load_state_dict(best_state)
    train_user_vec_mean = model.user_embedding.weight.detach().mean(dim=0).cpu()
    train_counts = train_ratings["movie_id"].value_counts()
    item_counts = np.zeros(len(movie_to_index), dtype=np.int64)
    for movie_id, count in train_counts.items():
        item_counts[movie_to_index[int(movie_id)]] = int(count)

    print(f"best epoch: {best_epoch}; stopped epoch: {stopped_epoch}")
    checkpoint = _make_checkpoint(
        "ncf",
        best_state,
        user_to_index,
        index_to_user,
        movie_to_index,
        index_to_movie,
        global_mean,
        item_counts,
        train_user_vec_mean,
        dim=dim,
        dropout=dropout,
    )
    print(f"saved: {_save_checkpoint(checkpoint, 'ncf')}")


def main(argv: Sequence[str] | None = None) -> None:
    """train利用者データでmean、MFまたはNCFを学習して保存する。

    Args:
        argv: コマンドライン引数。Noneならsys.argvを使う。

    Returns:
        None。
    """
    args = parse_args(argv)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ratings = load_train_ratings(
        DATA_DIR / "ratings.dat",
        SPLITS_DIR / "train_users.txt",
    )
    if ratings.empty:
        raise ValueError("no ratings found for train users")

    train_ratings, val_ratings = split_train_val(ratings, seed=args.seed)
    (
        movie_to_index,
        index_to_movie,
        user_to_index,
        index_to_user,
    ) = build_id_maps(ratings)
    _validate_mappings(
        train_ratings,
        val_ratings,
        user_to_index,
        movie_to_index,
    )

    print(
        f"train users: {len(user_to_index)}, train ratings: {len(train_ratings)}, "
        f"val ratings: {len(val_ratings)}"
    )
    if args.model == "mean":
        _run_mean(
            train_ratings,
            val_ratings,
            movie_to_index,
            index_to_movie,
            user_to_index,
            index_to_user,
            args.shrink,
        )
    elif args.model == "mf":
        _run_mf(
            train_ratings,
            val_ratings,
            user_to_index,
            index_to_user,
            movie_to_index,
            index_to_movie,
            args.dim,
            args.epochs,
            args.lr,
            args.wd,
            args.batch,
        )
    else:
        _run_ncf(
            train_ratings,
            val_ratings,
            user_to_index,
            index_to_user,
            movie_to_index,
            index_to_movie,
            args.dim,
            args.epochs,
            args.lr,
            args.wd,
            args.batch,
            args.dropout,
        )


if __name__ == "__main__":
    main()
