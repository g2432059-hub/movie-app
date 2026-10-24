from pathlib import Path

import pandas as pd


def load_ratings(path: str | Path) -> pd.DataFrame:
    """ratings.datを読み込む。

    Args:
        path: ratings.datのファイルパス。

    Returns:
        user_id, movie_id, rating, timestamp列を持つDataFrame。
        4列はいずれも整数型。
    """
    return pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
        dtype={
            "user_id": "int64",
            "movie_id": "int64",
            "rating": "int64",
            "timestamp": "int64",
        },
        encoding="ascii",
    )


def load_movies(path: str | Path) -> pd.DataFrame:
    """movies.datを読み込み、ジャンルを文字列のリストに変換する。

    Args:
        path: movies.datのファイルパス。

    Returns:
        movie_id, title, genres列を持つDataFrame。movie_idは整数型、
        titleは文字列型、genresはジャンル名のlist[str]。
    """
    movies = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["movie_id", "title", "genres"],
        dtype={"movie_id": "int64", "title": str, "genres": str},
        encoding="iso-8859-1",
    )
    movies["genres"] = movies["genres"].str.split("|")
    return movies


def build_id_maps(
    ratings_train: pd.DataFrame,
) -> tuple[
    dict[int, int],
    dict[int, int],
    dict[int, int],
    dict[int, int],
]:
    """train評価に含まれる映画・利用者IDの相互変換辞書を作る。

    Args:
        ratings_train: user_idとmovie_id列を含むtrain評価のDataFrame。

    Returns:
        movie_id→index、index→movie_id、user_id→index、
        index→user_idの順の4辞書。IDとindexはいずれも整数型で、
        indexは各ID集合を昇順に並べた0始まりの連番。
    """
    movie_ids = sorted(ratings_train["movie_id"].unique())
    user_ids = sorted(ratings_train["user_id"].unique())

    movie_to_index = {int(movie_id): index for index, movie_id in enumerate(movie_ids)}
    index_to_movie = {index: movie_id for movie_id, index in movie_to_index.items()}
    user_to_index = {int(user_id): index for index, user_id in enumerate(user_ids)}
    index_to_user = {index: user_id for user_id, index in user_to_index.items()}

    return movie_to_index, index_to_movie, user_to_index, index_to_user


if __name__ == "__main__":
    ratings = load_ratings("data/ml-1m/ratings.dat")
    movies = load_movies("data/ml-1m/movies.dat")

    print(f"ratings: {len(ratings)} rows")
    print(ratings.head())
    print(f"movies: {len(movies)} rows")
    print(movies.head())
