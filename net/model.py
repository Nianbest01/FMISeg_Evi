import torch
import torch.nn as nn
from einops import rearrange, repeat
from net.decoder import Decoder
from net.uncertainty import TokenLevelUncertaintyModule
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.upsample import SubpixelUpsample
from transformers import AutoTokenizer, AutoModel


class BERTModel(nn.Module):
    """Enhanced BERT encoder with token-level output for fine-grained fusion."""

    def __init__(self, bert_type, project_dim):
        super(BERTModel, self).__init__()
        self.model = AutoModel.from_pretrained(
            bert_type, output_hidden_states=True, trust_remote_code=True)
        self.project_head = nn.Sequential(
            nn.Linear(768, project_dim),
            nn.LayerNorm(project_dim),
            nn.GELU(),
            nn.Linear(project_dim, project_dim)
        )
        # Freeze BERT parameters for stability
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        output = self.model(
            input_ids=input_ids, attention_mask=attention_mask,
            output_hidden_states=True, return_dict=True)
        # Stack layers 1, 2, and last for multi-level representation
        last_hidden_states = torch.stack([
            output['hidden_states'][1],
            output['hidden_states'][2],
            output['hidden_states'][-1]
        ])  # [3, B, L, 768]
        # Global pooled embedding for classification tasks
        embed = last_hidden_states.permute(1, 0, 2, 3).mean(2).mean(1)
        embed = self.project_head(embed)
        # Return both full hidden states and projected embedding
        return {'feature': output['hidden_states'], 'project': embed}


class VisionModel(nn.Module):
    """Vision encoder extracting multi-scale features."""

    def __init__(self, vision_type, project_dim):
        super(VisionModel, self).__init__()
        self.model = AutoModel.from_pretrained(
            vision_type, output_hidden_states=True)
        self.project_head = nn.Linear(768, project_dim)
        self.spatial_dim = 768

    def forward(self, x):
        output = self.model(x, output_hidden_states=True)
        embeds  = output['pooler_output'].squeeze()
        project = self.project_head(embeds)
        return {"feature": output['hidden_states'], "project": project}


class FFBI(nn.Module):
    """Dual-Frequency Feature Bidirectional Interaction module."""

    def __init__(self, dim, num, batchf):
        super(FFBI, self).__init__()
        self.cross_attnl = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num, batch_first=batchf)
        self.cross_attnh = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num, batch_first=batchf)

    def forward(self, x, y):
        # Low-frequency branch attends to high-frequency
        x1, _ = self.cross_attnl(query=x, key=y, value=y)
        x2 = x1 + x
        # High-frequency branch attends to low-frequency
        y1, _ = self.cross_attnh(query=y, key=x, value=x)
        y2 = y1 + y
        return x2, y2


class TokenLevelCrossModalFusion(nn.Module):
    """
    Token-level cross-modal fusion module.

    Enables fine-grained interaction between text tokens and image patches
    through cross-attention mechanisms at each spatial scale.
    Includes projection to align text/image dimensions when they differ.
    """

    def __init__(self, img_dim, txt_dim=768, num_heads=8, batch_first=True):
        super(TokenLevelCrossModalFusion, self).__init__()
        self.img_dim = img_dim
        self.txt_dim = txt_dim

        # Project text features to match image dimension if needed
        self.text_project = nn.Linear(txt_dim, img_dim) if txt_dim != img_dim else nn.Identity()

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=img_dim, num_heads=num_heads, batch_first=batch_first)
        self.norm1 = nn.LayerNorm(img_dim)
        self.norm2 = nn.LayerNorm(img_dim)
        self.ffn = nn.Sequential(
            nn.Linear(img_dim, img_dim * 2),
            nn.GELU(),
            nn.Linear(img_dim * 2, img_dim),
        )

    def forward(self, image_tokens, text_tokens, attention_mask=None):
        """
        Args:
            image_tokens: [B, N_img, C_img] image patch tokens
            text_tokens:  [B, N_txt, C_txt] text tokens
            attention_mask: optional attention mask for text tokens
        Returns:
            fused_tokens: [B, N_img, C_img] image tokens fused with text information
            text_fused:   [B, N_txt, C_img] text tokens fused with image information (projected)
        """
        # Project text to image dimension
        text_proj = self.text_project(text_tokens)  # [B, N_txt, img_dim]

        # Cross-attention: image queries, text keys/values
        img_attn, _ = self.cross_attn(
            query=image_tokens,
            key=text_proj,
            value=text_proj,
            key_padding_mask=attention_mask
        )
        img_fused = self.norm1(image_tokens + img_attn)

        # Cross-attention: text queries, image keys/values
        txt_attn, _ = self.cross_attn(
            query=text_proj,
            key=image_tokens,
            value=image_tokens
        )
        txt_fused = self.norm2(text_proj + txt_attn)

        # Feed-forward with residual connection
        img_fused = img_fused + self.ffn(img_fused)
        txt_fused = txt_fused + self.ffn(txt_fused)

        return img_fused, txt_fused


class SegModel(nn.Module):
    """
    Segmentation model with token-level text-image fusion.

    Key improvements over model4.py:
    1. Token-level cross-modal fusion at multiple scales
    2. Bidirectional text-image interaction
    3. Enhanced uncertainty estimation with token-level features
    """

    def __init__(self, bert_type, vision_type, project_dim=512):
        super(SegModel, self).__init__()
        self.encoder  = VisionModel(vision_type, project_dim)
        self.encoder2 = VisionModel(vision_type, project_dim)
        self.text_encoder = BERTModel(bert_type, project_dim)
        self.spatial_dim = [7, 14, 28, 56]
        feature_dim  = [768, 384, 192, 96]

        # Token-level fusion modules at each scale (with dimension projection)
        self.token_fusion3 = TokenLevelCrossModalFusion(feature_dim[0], txt_dim=768, num_heads=8)
        self.token_fusion2 = TokenLevelCrossModalFusion(feature_dim[1], txt_dim=768, num_heads=8)
        self.token_fusion1 = TokenLevelCrossModalFusion(feature_dim[2], txt_dim=768, num_heads=8)
        self.token_fusion0 = TokenLevelCrossModalFusion(feature_dim[3], txt_dim=768, num_heads=8)

        # ── 高频分支 decoder ──────────────────────────────────────────────
        self.decoder16 = Decoder(feature_dim[0], feature_dim[1], self.spatial_dim[0], 24)
        self.decoder8  = Decoder(feature_dim[1], feature_dim[2], self.spatial_dim[1], 12)
        self.decoder4  = Decoder(feature_dim[2], feature_dim[3], self.spatial_dim[2],  9)
        self.decoder1  = SubpixelUpsample(2, feature_dim[3], 24, 4)
        self.out       = UnetOutBlock(2, in_channels=24, out_channels=1)

        # ── 低频分支 decoder ───────────────────────────────────────────────
        self.decoder16_2 = Decoder(feature_dim[0], feature_dim[1], self.spatial_dim[0], 24)
        self.decoder8_2  = Decoder(feature_dim[1], feature_dim[2], self.spatial_dim[1], 12)
        self.decoder4_2  = Decoder(feature_dim[2], feature_dim[3], self.spatial_dim[2],  9)
        self.decoder1_2  = SubpixelUpsample(2, feature_dim[3], 24, 4)
        self.out_2       = UnetOutBlock(2, in_channels=24, out_channels=1)

        self.ffbi = FFBI(feature_dim[0], 4, True)

        # ── Token-level 多模态多尺度 NIG 不确定性模块 ─────────────────────
        self.uncertainty = TokenLevelUncertaintyModule(
            feature_dims=[96, 192, 384, 768],
            text_dim=768,
            hidden_dim=64,
        )

    def forward(self, data):
        image2, image, text = data

        if image.shape[1] == 1:
            image  = repeat(image,  'b 1 h w -> b c h w', c=3)
            image2 = repeat(image2, 'b 1 h w -> b c h w', c=3)

        # ── 特征提取 ──────────────────────────────────────────────────────
        image_output  = self.encoder(image)
        image_output2 = self.encoder2(image2)
        image_features,  _ = image_output['feature'],  image_output['project']
        image_features2, _ = image_output2['feature'], image_output2['project']
        text_output = self.text_encoder(text['input_ids'], text['attention_mask'])
        text_embeds, _ = text_output['feature'], text_output['project']

        # Process image features: remove layer 0, reshape to token format
        if len(image_features[0].shape) == 4:
            image_features  = image_features[1:]  # 4 layers
            image_features  = [rearrange(item, 'b c h w -> b (h w) c')
                                for item in image_features]
            image_features2 = image_features2[1:]
            image_features2 = [rearrange(item, 'b c h w -> b (h w) c')
                                for item in image_features2]

        # Get text tokens (use last hidden layer: [B, L, 768])
        text_tokens = text_embeds[-1]  # [B, L, 768]

        # Create attention mask for text tokens (mask padding tokens)
        text_attention_mask = None
        if 'attention_mask' in text:
            # Invert: True for padding (to be masked)
            text_attention_mask = ~text['attention_mask'].bool()  # [B, L]

        # ── Token-level 跨模态融合 ────────────────────────────────────────
        # Apply token-level fusion at each scale
        fused_img_features = []
        fused_img_features2 = []

        fusion_modules = [self.token_fusion0, self.token_fusion1,
                          self.token_fusion2, self.token_fusion3]

        for i in range(len(image_features)):
            # High-frequency branch fusion
            fused_img, fused_txt = fusion_modules[i](
                image_features[i], text_tokens, text_attention_mask
            )
            fused_img_features.append(fused_img)

            # Low-frequency branch fusion
            fused_img2, _ = fusion_modules[i](
                image_features2[i], text_tokens, text_attention_mask
            )
            fused_img_features2.append(fused_img2)

        # ── FFBI 双频尺度交互 ──────────────────────────────────────────
        os32   = fused_img_features[3]
        os32_2 = fused_img_features2[3]
        fu32, fu32_2 = self.ffbi(os32, os32_2)

        # ── 高频分支逐级解码 ───────────────────────────────────────────────
        os16   = self.decoder16(fu32,   fused_img_features[2],  text_tokens)
        os8    = self.decoder8(os16,    fused_img_features[1],  text_tokens)
        os4    = self.decoder4(os8,     fused_img_features[0],  text_tokens)
        os4    = rearrange(os4,  'B (H W) C -> B C H W',
                           H=self.spatial_dim[-1], W=self.spatial_dim[-1])
        os1    = self.decoder1(os4)
        out    = self.out(os1).sigmoid()

        # ── 低频分支逐级解码 ───────────────────────────────────────────────
        os16_2 = self.decoder16_2(fu32_2, fused_img_features2[2], text_tokens)
        os8_2  = self.decoder8_2(os16_2,  fused_img_features2[1], text_tokens)
        os4_2  = self.decoder4_2(os8_2,   fused_img_features2[0], text_tokens)
        os4_2  = rearrange(os4_2, 'B (H W) C -> B C H W',
                           H=self.spatial_dim[-1], W=self.spatial_dim[-1])
        os1_2  = self.decoder1_2(os4_2)
        out_2  = self.out_2(os1_2).sigmoid()

        # ── Token-level 多模态多尺度 NIG 不确定性估计 ─────────────────────
        final_nig = self.uncertainty(
            high_features=fused_img_features,    # list of 4 × [B, H*W, C]
            low_features=fused_img_features2,    # list of 4 × [B, H*W, C]
            text_tokens=text_tokens,             # [B, L, 768]
        )

        return out, out_2, final_nig