import torch
from torch import nn


class MF(nn.Module):
    """利用者・映画の埋め込みによる行列分解モデル。"""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        dim: int,
        global_mean: float,
    ) -> None:
        """モデルの埋め込み、偏り、全体平均を初期化する。

        Args:
            n_users: 利用者数。
            n_items: 映画数。
            dim: 利用者・映画ベクトルの次元数。
            global_mean: 学習評価から計算した全体平均。

        Returns:
            None。
        """
        super().__init__()
        if n_users <= 0 or n_items <= 0 or dim <= 0:
            raise ValueError("n_users, n_items, and dim must be positive")

        self.user_embedding = nn.Embedding(n_users, dim)
        self.item_embedding = nn.Embedding(n_items, dim)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.register_buffer("global_mean", torch.tensor(float(global_mean)))

        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=0.05)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.05)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(
        self,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
    ) -> torch.Tensor:
        """指定された利用者・映画indexの評価を予測する。

        Args:
            user_idx: 利用者indexの整数Tensor。
            item_idx: 映画indexの整数Tensor。user_idxと同じ形状。

        Returns:
            global_mean + user_bias + item_bias + 内積の予測Tensor。
        """
        user_vec = self.user_embedding(user_idx)
        item_vec = self.item_embedding(item_idx)
        user_bias = self.user_bias(user_idx).squeeze(-1)
        item_bias = self.item_bias(item_idx).squeeze(-1)
        interaction = (user_vec * item_vec).sum(dim=-1)
        return self.global_mean + user_bias + item_bias + interaction

    def predict_with_user_params(
        self,
        user_vec: torch.Tensor,
        user_bias: torch.Tensor | float,
        item_idx: torch.Tensor,
    ) -> torch.Tensor:
        """外部の利用者ベクトル・偏りを使って映画評価を予測する。

        Args:
            user_vec: fold-inで得た利用者ベクトル。形状は(dim,)または
                item_idxに対応する(..., dim)。
            user_bias: fold-inで得た利用者偏り。スカラーまたは各映画分。
            item_idx: 予測対象映画の整数index Tensor。

        Returns:
            global_mean + user_bias + 映画偏り + 内積の予測Tensor。
        """
        item_vec = self.item_embedding(item_idx)
        item_bias = self.item_bias(item_idx).squeeze(-1)
        interaction = (user_vec * item_vec).sum(dim=-1)
        return self.global_mean + user_bias + item_bias + interaction
