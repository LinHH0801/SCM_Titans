from functools import partial

import torch
import torch.nn as nn
from timm.models.layers import DropPath

from models.vtitans.titans_pytorch import NeuralMemory
from models.vtitans.titans_pytorch.mac_transformer import GEGLU, SegmentedAttention, flex_attention, create_mac_block_mask
from models.utils.func import exists, default


class TitansBlock(nn.Module):
    def __init__(self, dim, layer_id=0, neural_memory_layers=None, dim_head=64, heads=8, segment_len=32,
                 chunk_size=64, neural_memory_batch_size=1024, num_longterm_mem_tokens=0,
                 num_persist_mem_tokens=4, ff_mult=4, drop_path=0., init_values=None, use_flex_attn=False,
                 neural_mem_weight_residual=True, sliding_window_attn=True, gate_attn_output=False):
        super().__init__()
        self.segment_len = segment_len
        self.num_longterm_mem_tokens = num_longterm_mem_tokens
        self.num_persist_mem_tokens = num_persist_mem_tokens
        self.attn_window_size = segment_len + num_longterm_mem_tokens
        self.use_flex_attn = use_flex_attn
        self.sliding_window_attn = sliding_window_attn

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale = (init_values is not None)
        if self.layer_scale:
            self.gamma_att = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
            self.gamma_ffn = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
        neural_memory_layers = default(neural_memory_layers, [])
        if layer_id in neural_memory_layers:
            self.mem = NeuralMemory(
                dim=dim,
                chunk_size=chunk_size,
                batch_size=neural_memory_batch_size,
                qkv_receives_diff_views=False,
                accept_weight_residual=neural_mem_weight_residual and not (layer_id == neural_memory_layers[0]),
                attn_pool_chunks=True,
                qk_rmsnorm=True,
                momentum=True,
                momentum_order=1,
                default_step_transform_max_lr=1e-1,
                per_parameter_lr_modulation=True
            )
            self.gate_attn_output = gate_attn_output
            if not gate_attn_output:
                if self.layer_scale:
                    self.gamma_mem = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
            self.neural_mem_weight_residual = neural_mem_weight_residual
        else:
            self.mem = None
        self.att = SegmentedAttention(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            segment_len=segment_len,
            use_flex_attn=use_flex_attn,
            accept_value_residual=not (layer_id == 0),
            num_longterm_mem_tokens=num_longterm_mem_tokens,
            num_persist_mem_tokens=num_persist_mem_tokens,
            sliding=sliding_window_attn
        )

        dim_inner = int(dim * ff_mult * 2 / 3)
        self.ffn = nn.Sequential(
            nn.RMSNorm(dim),
            nn.Linear(dim, dim_inner * 2),
            GEGLU(),
            nn.Linear(dim_inner, dim)
        )

    def forward(self, x, mem_weight_residual=None, value_residual=None, disable_flex_attn=False):
        if x.is_cuda and self.use_flex_attn and not disable_flex_attn:
            seq_len = x.shape[1]
            seq_len_with_mem = self.seq_len_with_longterm_mem(seq_len)
            block_mask = create_mac_block_mask(seq_len_with_mem, self.attn_window_size, self.num_persist_mem_tokens,
                                               self.sliding_window_attn)
            flex_attn_fn = partial(flex_attention, block_mask=block_mask)
        else:
            flex_attn_fn = None

        if exists(self.mem):
            retrieved, next_neural_mem_cache = self.mem(x, prev_weights=mem_weight_residual)
            if self.neural_mem_weight_residual:
                mem_weight_residual = next_neural_mem_cache.updates
            if self.gate_attn_output:
                attn_out_gates = retrieved.sigmoid()
            else:
                if self.layer_scale:
                    x = x + self.drop_path(self.gamma_mem * retrieved)
                else:
                    x = x + self.drop_path(retrieved)
                attn_out_gates = None

        else:
            attn_out_gates = None

        x_att, (values, _) = self.att(
            x,
            value_residual=value_residual,
            disable_flex_attn=disable_flex_attn,
            flex_attn_fn=flex_attn_fn,
            output_gating=attn_out_gates,
        )
        if self.layer_scale:
            x = x + self.drop_path(self.gamma_att * x_att)
            x = x + self.drop_path(self.gamma_ffn * self.ffn(x))
        else:
            x = x + self.drop_path(x_att)
            x = x + self.drop_path(self.ffn(x))
        value_residual = default(value_residual, values)
        return x, mem_weight_residual, value_residual

    def seq_len_with_longterm_mem(self, seq_len):
        assert seq_len > 0

        segment_len, num_mem = self.segment_len, self.num_longterm_mem_tokens
        return ((seq_len - 1) // segment_len) * num_mem + seq_len
if __name__ == '__main__':
    # 构造符合 TitansBlock 输入要求的序列张量
    x_img = torch.randn(1, 512, 32, 32).cuda()  # [B, C, H, W]
    print(x_img)
    B, C, H, W = x_img.shape
    x_seq = x_img.flatten(2).transpose(1, 2)  # [B, H*W, C] = [1, 1024, 512]
    print("Input sequence shape:", x_seq.shape)  # torch.Size([1, 1024, 512])

    # 初始化 TitansBlock（需指定 dim=512）
    model = TitansBlock(
        dim=512,
        layer_id=0,
        neural_memory_layers=[0],  # 启用 memory
        segment_len=32,
        chunk_size=64,
        num_longterm_mem_tokens=8,
        num_persist_mem_tokens=4,
        use_flex_attn=False,  # 简化测试，关闭 flex_attn
        sliding_window_attn=True
    ).cuda()

    # 前向传播
    x_out, mem_weight_residual, value_residual = model(x_seq)
    print(x_out)

    print("Output shape:", x_out.shape)  # 应为 [1, 1024, 512]
    print("mem_weight_residual shape (if exists):",
          mem_weight_residual.shape if mem_weight_residual is not None else None)
    print("value_residual shape:", value_residual.shape)  # 应与 x_out 的 value 部分一致