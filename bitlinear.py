import torch
import torch.nn as nn
from torch import Tensor

class BitRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps):
        """
        BitRMSNorm is equivalent to LlamaRMSNorm and T5LayerNorm
        refers: https://github.com/huggingface/transformers/blob/c5f0288bc7d76f65996586f79f69fba8867a0e67/src/transformers/models/llama/modeling_llama.py#L76C1-L90C59

        Args:
            hidden_size (_type_): _description_
            eps (_type_, optional): _description_. Defaults to 1e-6.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)  # mean(x**2)
        # x / sqrt(mean(x**2) + epsilon)
        hidden_states = hidden_states * \
            torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class BitLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, rms_norm_eps=1e-6, bits=8, flg_before_linear=True):
        super(BitLinear, self).__init__(in_features, out_features, bias)
        self.layer_norm = BitRMSNorm(in_features, rms_norm_eps)
        self.bits = bits
        self.Qb = 2 ** (bits - 1)
        self.flg_before_linear = flg_before_linear
        self.epsilon = 1e-6

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the BitLinear layer.

        Args:
            x (Tensor): _description_

        Returns:
            Tensor: _description_
        """

        # 1. レイヤー正規化 (in: x, out: x_norm)
        # レイヤー正規化：https://i-main.net/emmanote-ai-normalization/#toc3
        # 今回はRMSNormを使用
        x_norm = self.layer_norm(x)

        # 2. Absmax 量子化 (in: x_norm, out: x_q, gamma)
        x_q, gamma = self.absmax_quantize(x_norm)

        # 3. 1-bit Weight化 (in: - , out: w_q, beta)
        w_q, beta = self.quantize_weights()

        # 4. テンソル積 (in: x_q, gamma, out: x_matmul)
        x_matmul = torch.nn.functional.linear(x_q, w_q, self.bias)

        # 5. 逆量子化 (in: x_matmul, beta, gamma, out: x_out)
        output = x_matmul * (beta * gamma / self.Qb)

        return output

    def absmax_quantize(self, x):
        if self.flg_before_linear:
            # パターン①：　通常は[-Qb, Qb]にスケール: 式(4), (5)を適用
            gamma = torch.abs(x).max().clamp(min=self.epsilon)
            x_scaled = x * self.Qb / gamma
            x_q = torch.round(x_scaled).clamp(-self.Qb, self.Qb - 1)
        else:
            # パターン②：　Reluなどの非線形関数前の場合は[0, Qb]にスケール：　式(6)を適用
            # 論文中には記載はないですが、スケールが異なるためスケーリングの基準として使っているgammaもetaを反映した値にすべきだと考えます。
            eta = x.min()
            gamma = torch.abs(x - eta).max().clamp(min=self.epsilon)
            x_scaled = (x - eta) * self.Qb / gamma
            x_q = torch.round(x_scaled).clamp(0, self.Qb - 1)
        # STE
        x_q = (x_q - x_scaled).detach() + x_scaled
        return x_q, gamma

    def custom_sign(self, x):
        return (x > 0).to(torch.int8) * 2 - 1

    def quantize_weights(self):
        # 式(3): alphaの計算
        alpha = self.weight.mean()

        # 式(1),(2): 重みの中心化とバイナリ化
        weight_centered = self.weight - alpha
        weight_binarized = self.custom_sign(weight_centered)

        # 式(12): betaの計算
        beta = self.weight.abs().mean()

        # STE (weight_binarizedとスケールを合わせるためweight_centeredをweight_scaledにスケールしています。)
        weight_scaled = weight_centered / (weight_centered.abs().max() + self.epsilon)
        weight_binarized = (weight_binarized - weight_scaled).detach() + weight_scaled

        return weight_binarized, beta
