import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ─────────────────────────────────────────────────────────────────────────────
# 基础工具函数
# ─────────────────────────────────────────────────────────────────────────────

def evidence(x):
    """将原始 logit 约束为正值。"""
    return F.softplus(x)


def get_nig_params(logit):
    """
    将 [B, 4] 的原始 logit 分拆并激活为合法 NIG 参数 (u, la, alpha, beta)。

    Args:
        logit : [B, 4]  原始网络输出，通道顺序 [u, logla, logalpha, logbeta]
    Returns:
        u, la, alpha, beta  各 [B, 1]
    """
    u_raw, logla, logalpha, logbeta = torch.split(logit, 1, dim=1)
    u     = u_raw
    la    = evidence(logla)
    alpha = evidence(logalpha) + 1.0
    beta  = evidence(logbeta)
    return u, la, alpha, beta


def moe_nig(u1, la1, alpha1, beta1, u2, la2, alpha2, beta2):
    """
    两个 NIG 分布的贝叶斯混合（Mixture-of-Experts）。

    λ    = λ1 + λ2
    u    = (λ1·u1 + λ2·u2) / λ
    α    = α1 + α2 + 0.5
    β    = β1 + β2 + 0.5·(λ1·(u1−u)² + λ2·(u2−u)²)
    """
    la    = la1 + la2
    u     = (la1 * u1 + la2 * u2) / la
    alpha = alpha1 + alpha2 + 0.5
    beta  = beta1 + beta2 + 0.5 * (la1 * (u1 - u) ** 2 + la2 * (u2 - u) ** 2)
    return u, la, alpha, beta


def combine_uncertainty(ests):
    """
    对多个 NIG 估计迭代调用 moe_nig。

    Args:
        ests : List of [u, la, alpha, beta]，每个元素各为 [B, 1]
    Returns:
        (u, la, alpha, beta)  各 [B, 1]
    """
    u, la, alpha, beta = ests[0]
    for i in range(1, len(ests)):
        u1, la1, alpha1, beta1 = ests[i]
        u, la, alpha, beta = moe_nig(u, la, alpha, beta, u1, la1, alpha1, beta1)
    return u, la, alpha, beta


# ─────────────────────────────────────────────────────────────────────────────
# NIG Head：将特征映射到 NIG 参数
# ─────────────────────────────────────────────────────────────────────────────

class NIGHead(nn.Module):
    """
    将特征向量映射为 NIG 参数 (u, la, alpha, beta) 的轻量头部。

    输入：
        [B, C]   — 已池化的特征向量
        [B, N, C] — 序列特征（自动对 token 维度做均值池化）
    输出：
        u, la, alpha, beta  各 [B, 1]
    """

    def __init__(self, in_channels: int, hidden_dim: int = 64):
        super(NIGHead, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x: torch.Tensor):
        if x.dim() == 3:
            x = x.mean(dim=1)
        logit = self.head(x)
        return get_nig_params(logit)


class TokenLevelNIGHead(nn.Module):
    """
    Token-level NIG Head for fine-grained uncertainty estimation.
    
    Produces NIG parameters for each token, then aggregates to global uncertainty.
    This enables the model to capture local uncertainty variations before fusion.
    
    输入：
        [B, N, C] — token 序列特征
    输出：
        u, la, alpha, beta  各 [B, 1] (聚合后)
        token_level_params  — 各 [B, N, 1] (token 级别原始参数)
    """

    def __init__(self, in_channels: int, hidden_dim: int = 64):
        super(TokenLevelNIGHead, self).__init__()
        self.token_head = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4)
        )
        
        # Global aggregation head (after token-level processing)
        self.global_head = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, N, C] token sequence features
        Returns:
            u, la, alpha, beta: 各 [B, 1] (globally aggregated)
            token_params: dict of token-level [B, N, 1] parameters
        """
        # Token-level NIG parameters
        token_logits = self.token_head(x)  # [B, N, 4]
        token_u, token_la, token_alpha, token_beta = torch.split(token_logits, 1, dim=2)
        
        # Activate token-level parameters
        token_u = token_u
        token_la = evidence(token_la)
        token_alpha = evidence(token_alpha) + 1.0
        token_beta = evidence(token_beta)
        
        # Global pooling for final NIG parameters
        x_pooled = x.mean(dim=1)  # [B, C]
        global_logit = self.global_head(x_pooled)  # [B, 4]
        u, la, alpha, beta = get_nig_params(global_logit)
        
        token_params = {
            'u': token_u,
            'la': token_la,
            'alpha': token_alpha,
            'beta': token_beta
        }
        
        return u, la, alpha, beta, token_params


# ─────────────────────────────────────────────────────────────────────────────
# Token-level 多模态不确定性模块
# ─────────────────────────────────────────────────────────────────────────────

class TokenLevelUncertaintyModule(nn.Module):
    """
    Token-level multi-modal uncertainty estimation module.
    
    Key improvements over MultiModalUncertaintyModule:
    1. Token-level NIG heads for fine-grained uncertainty
    2. Cross-modal token fusion before uncertainty estimation
    3. Hierarchical aggregation from token to global uncertainty
    
    Architecture:
    - Text NIG head: processes [B, L, 768] → token-level + global
    - Image NIG heads (4 scales): process [B, H*W, C] → token-level + global
    - Fusion: combine token-level uncertainties, then aggregate
    """

    def __init__(
        self,
        feature_dims: list,
        text_dim: int = 768,
        hidden_dim: int = 64,
    ):
        super(TokenLevelUncertaintyModule, self).__init__()

        # ── Token-level 文本 NIG head ────────────────────────────────────
        self.text_nig_head = TokenLevelNIGHead(text_dim, hidden_dim)

        # ── Token-level 高频图像 NIG heads（4 个尺度）────────────────────
        self.high_nig_heads = nn.ModuleList([
            TokenLevelNIGHead(dim, hidden_dim) for dim in feature_dims
        ])

        # ── Token-level 低频图像 NIG heads（4 个尺度）───────────────────
        self.low_nig_heads = nn.ModuleList([
            TokenLevelNIGHead(dim, hidden_dim) for dim in feature_dims
        ])
        
        # ── Cross-modal token fusion for uncertainty ────────────────────
        # Projects text features to each image scale for token-level fusion
        self.text_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(text_dim, dim),
                nn.LayerNorm(dim),
                nn.GELU()
            ) for dim in feature_dims
        ])

    def forward(
        self,
        high_features: list,
        low_features: list,
        text_tokens: torch.Tensor,
    ):
        """
        Args:
            high_features : 4 个张量的列表，每个 [B, H*W, C]  (高频 encoder)
            low_features  : 4 个张量的列表，每个 [B, H*W, C]  (低频 encoder2)
            text_tokens   : [B, L, 768]  BERT 最后一层所有 token 的隐状态

        Returns:
            final_nig : (u, la, alpha, beta)  各 [B, 1]，最终融合结果
        """
        # ── Step 0: 文本 Token-level NIG ─────────────────────────────────
        text_u, text_la, text_alpha, text_beta, text_token_params = \
            self.text_nig_head(text_tokens)
        text_nig = (text_u, text_la, text_alpha, text_beta)

        # ── Step 1: 高频 4 尺度 Token-level NIG ──────────────────────────
        high_nigs = []
        for i, feat in enumerate(high_features):
            u, la, alpha, beta, _ = self.high_nig_heads[i](feat)
            high_nigs.append((u, la, alpha, beta))

        # ── Step 2: 低频 4 尺度 Token-level NIG ──────────────────────────
        low_nigs = []
        for i, feat in enumerate(low_features):
            u, la, alpha, beta, _ = self.low_nig_heads[i](feat)
            low_nigs.append((u, la, alpha, beta))

        # ── Step 3: 高频 4 尺度 → 1（解析贝叶斯融合）─────────────────────
        fused_high = combine_uncertainty(high_nigs)

        # ── Step 4: 低频 4 尺度 → 1（解析贝叶斯融合）─────────────────────
        fused_low = combine_uncertainty(low_nigs)

        # ── Step 5: NIG1 = 高频×低频图像融合 ────────────────────────────
        nig1 = moe_nig(*fused_high, *fused_low)

        # ── Step 6: NIG2 = 图像×文本融合 ────────────────────────────────
        final_nig = moe_nig(*nig1, *text_nig)

        # ── Step 7: 最终融合 ────────────────────────────────────────────
        # final_nig = moe_nig(*nig1, *nig2)

        return final_nig


# ─────────────────────────────────────────────────────────────────────────────
# NIG 损失函数
# ─────────────────────────────────────────────────────────────────────────────

def nig_nll_loss(y, u, la, alpha, beta, reduction='mean'):
    """
    Normal-Inverse-Gamma 分布的负对数似然（NLL）。

    L_NLL = 0.5·log(π/λ) − α·log(Ω) + (α+0.5)·log(λ·(y−u)²+Ω)
            + lgamma(α) − lgamma(α+0.5)
    其中 Ω = 2β(1+λ)
    """
    omega = 2.0 * beta * (1.0 + la)
    nll = (
          0.5 * torch.log(torch.pi / la)
        - alpha * torch.log(omega)
        + (alpha + 0.5) * torch.log(la * (y - u) ** 2 + omega)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )
    return nll.mean() if reduction == 'mean' else nll


def nig_reg_loss(y, u, la, alpha, reduction='mean'):
    """
    证据感知正则化：当预测有误时惩罚过高的证据。

    L_Reg = |y − u| · (2λ + α)
    """
    error = torch.abs(y - u)
    evidence_sum = 2.0 * la + alpha
    reg = error * evidence_sum
    return reg.mean() if reduction == 'mean' else reg


def nig_loss(y, nig_params, lam=1e-2, reduction='mean'):
    """
    组合 NIG 损失：L_NIG = L_NLL + λ · L_Reg

    Args:
        y          : 回归目标 [B, 1]
        nig_params : (u, la, alpha, beta)  各 [B, 1]
        lam        : 正则化权重
    Returns:
        (total, nll, reg)
    """
    u, la, alpha, beta = nig_params
    nll   = nig_nll_loss(y, u, la, alpha, beta, reduction=reduction)
    reg   = nig_reg_loss(y, u, la, alpha, reduction=reduction)
    total = nll + lam * reg
    return total, nll, reg


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数：从 NIG 参数中提取可解释不确定性
# ─────────────────────────────────────────────────────────────────────────────

def get_uncertainty(nig_params):
    """
    从 NIG 参数中分解偶然不确定性与认知不确定性。

    偶然不确定性 (数据噪声)   : U_a = β / (α − 1)
    认知不确定性 (模型不确定) : U_e = β / (λ·(α − 1))

    Args:
        nig_params : (u, la, alpha, beta)  各 [B, 1]
    Returns:
        aleatoric  : [B, 1]
        epistemic  : [B, 1]
    """
    _, la, alpha, beta = nig_params
    denom = (alpha - 1.0).clamp(min=1e-6)
    aleatoric = beta / denom
    epistemic = beta / (la * denom)
    return aleatoric, epistemic
