from pathlib import Path

import pandas as pd
import streamlit as st

from recommender import LoadedModel, load_model, recommend, validate_inputs


@st.cache_resource
def get_cached_model() -> LoadedModel:
    """NCFモデルを一度だけ読み込んで返す。

    Returns:
        推論用のLoadedModel。
    """
    return load_model("ncf")


def main() -> None:
    """映画と評価を受け取り、推薦結果をStreamlit画面に表示する。"""
    st.title("映画のおすすめ")

    if not Path("data").is_dir():
        st.error("データ用の data/ ディレクトリがありません。MovieLens 1Mのデータを配置してください。")
        return
    if not Path("models").is_dir():
        st.error("学習済みモデル用の models/ ディレクトリがありません。先にモデルを学習してください。")
        return
    if not Path("data/ml-1m/movies.dat").is_file():
        st.error("映画データ data/ml-1m/movies.dat がありません。")
        return
    if not Path("models/ncf.pt").is_file():
        st.error("学習済みNCFモデル models/ncf.pt がありません。先にモデルを学習してください。")
        return

    try:
        model = get_cached_model()
    except Exception as exc:
        st.error(f"モデルまたは映画データを読み込めませんでした: {exc}")
        return

    available_movie_ids = sorted(
        movie_id
        for movie_id in model.movie_to_index
        if movie_id in model.movie_catalog
    )
    if not available_movie_ids:
        st.error("学習データに対応する映画がありません。モデルと映画データを確認してください。")
        return

    selected_movie_ids = st.multiselect(
        "見た映画を選択",
        options=available_movie_ids,
        format_func=lambda movie_id: model.movie_catalog[movie_id][0],
    )
    items = [
        (
            int(movie_id),
            int(
                st.number_input(
                    f"{model.movie_catalog[movie_id][0]} の評価",
                    min_value=1,
                    max_value=5,
                    value=5,
                    step=1,
                    key=f"rating_{movie_id}",
                )
            ),
        )
        for movie_id in selected_movie_ids
    ]
    lam = st.slider(
        "ジャンル多様性の重み λ",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
    )
    st.caption(
        "0 に近いほど、予測評価の高い順に並べます。大きくするほど、おすすめの中にいろいろなジャンルを混ぜます(そのぶん、予測評価が少し低い映画も入ります)。"
    )

    if len(items) < 3:
        st.info(f"3本以上選んでください(現在 {len(items)} 本)")

    if st.button("おすすめを表示", disabled=len(items) < 3):
        try:
            validate_inputs(items, model_name="ncf", loaded_model=model)
            recommendations = recommend(items, lam=lam, model_name="ncf")
        except Exception as exc:
            st.error(f"入力の確認または推薦中にエラーが発生しました: {exc}")
            return

        rows = [
            {
                "順位": rank,
                "タイトル": title,
                "予測評価": f"{prediction:.2f}",
                "ジャンル": ", ".join(genres),
            }
            for rank, (title, prediction, genres) in enumerate(
                recommendations[:10],
                start=1,
            )
        ]
        st.table(pd.DataFrame(rows, columns=["順位", "タイトル", "予測評価", "ジャンル"]))
        st.caption(
            "並び順は『予測評価 + λ × ジャンルの新しさ』で決まります。表示の予測評価には加点を含まないため、並びと一致しない場合があります。"
        )
        genre_total = len(
            {genre for _, _, genres in recommendations[:10] for genre in genres}
        )
        st.write(f"10件に含まれるジャンル数: {genre_total}")

    st.caption("出典: MovieLens 1M を使った授業課題のデモ")


if __name__ == "__main__":
    main()
