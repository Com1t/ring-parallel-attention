import torch
import torch.nn.functional as F
from xformers.ops.fmha import memory_efficient_attention_partial, merge_attentions
import torch
import torch.distributed as dist
import os
import nvtx

"""
Ring Attention just different communication from Blockwise attention,
only make sense for distributed message passing.

To verify that each hosts received all KV blocks, check out ring-attention-string.py

```bash
torchrun --nproc_per_node=5 ring-attention-sdpa.py
```

Output,

```
0 torch.Size([1, 16, 20, 128])
4 torch.Size([1, 16, 20, 128])
1 torch.Size([1, 16, 20, 128])
2 torch.Size([1, 16, 20, 128])
3 torch.Size([1, 16, 20, 128])
```

To verify the integrity of Blockwise attention, check out prefill-sdpa.ipynb
"""


def verify_attn_output(
    q: torch.Tensor, k_blocks: list, v_blocks: list, attn_output: torch.Tensor
):
    k_blocks = torch.cat(k_blocks, dim=1)
    v_blocks = torch.cat(v_blocks, dim=1)
    golden_output = F.scaled_dot_product_attention(
        query=q.transpose(1, 2),
        key=k_blocks.transpose(1, 2),
        value=v_blocks.transpose(1, 2),
        is_causal=False,
    )
    torch.testing.assert_close(golden_output, attn_output, atol=1e-3, rtol=1e-3)


def torch_seq_parallel(
    world_size: int,
    batch_size: int,
    head_num: int,
    dim: int,
    seq_len: int,
    verify: bool = False,
    nvtx_enable: bool = False,
):
    chunk_len = seq_len // world_size
    max_lse = None
    new_denominator = None
    attn_output = None
    new_lse_full = None

    if verify:
        k_blocks = []
        v_blocks = []

    q = torch.ones(batch_size, chunk_len, head_num, dim)

    for step in range(world_size):
        k = torch.ones(batch_size, chunk_len, head_num, dim) * step
        v = torch.ones(batch_size, chunk_len, head_num, dim) * step

        if verify:
            k_blocks.append(k)
            v_blocks.append(v)

        out_, lse_ = memory_efficient_attention_partial(q, k, v)

        if nvtx_enable:
            merge_rng = nvtx.start_range(
                message="torch_merge_attentions", color="yellow"
            )
        lse_ = lse_.transpose(1, 2)
        if max_lse is None:
            max_lse = lse_
            adjust_factors = torch.ones_like(lse_).unsqueeze(-1)
            new_denominator = adjust_factors
            attn_output = out_ * adjust_factors
            new_lse_full = lse_
        else:
            new_max_lse = torch.maximum(max_lse, lse_)

            old_adjust_factors = torch.exp(max_lse - new_max_lse).unsqueeze(-1)

            new_adjust_factors = torch.exp(lse_ - new_max_lse).unsqueeze(-1)

            new_denominator = old_adjust_factors * new_denominator + new_adjust_factors
            attn_output = old_adjust_factors * attn_output + new_adjust_factors * out_
            new_lse_full = new_max_lse + torch.log(
                torch.exp(new_lse_full - new_max_lse) + torch.exp(lse_ - new_max_lse)
            )

            max_lse = new_max_lse
        print(f"torch {torch.cuda.max_memory_allocated() / 1024**2} MB")

        if nvtx_enable:
            nvtx.end_range(merge_rng)

    attn_output = attn_output / new_denominator

    if verify:
        attn_output = attn_output.transpose(1, 2).to(torch.float16)
        verify_attn_output(q, k_blocks, v_blocks, attn_output)


def xformers_seq_parallel(
    world_size: int,
    batch_size: int,
    head_num: int,
    dim: int,
    seq_len: int,
    verify: bool = False,
    nvtx_enable: bool = False,
):
    chunk_len = seq_len // world_size

    if verify:
        k_blocks = []
        v_blocks = []
    o_blocks = []
    lse_values = []

    q = torch.ones(batch_size, chunk_len, head_num, dim)

    for step in range(world_size):
        k = torch.ones(batch_size, chunk_len, head_num, dim) * step
        v = torch.ones(batch_size, chunk_len, head_num, dim) * step

        if verify:
            k_blocks.append(k)
            v_blocks.append(v)

        out_, lse_ = memory_efficient_attention_partial(q, k, v)

        # attn_out is in the shape of [B, M, num of heads, head_dim]
        o_blocks.append(out_)

        # LSE is in the shape of [B, num of heads, M]
        lse_values.append(lse_)
        print(f"once {torch.cuda.max_memory_allocated() / 1024**2} MB")

    if nvtx_enable:
        merge_rng = nvtx.start_range(message="xformers_merge_attentions", color="blue")

    attn_output, _ = merge_attentions(o_blocks, lse_values, write_lse=False)

    if nvtx_enable:
        nvtx.end_range(merge_rng)

    if verify:
        verify_attn_output(q, k_blocks, v_blocks, attn_output.transpose(1, 2))


def xformers_seq_parallel_merge_in_middle(
    world_size: int,
    batch_size: int,
    head_num: int,
    dim: int,
    seq_len: int,
    verify: bool = False,
    nvtx_enable: bool = False,
):
    chunk_len = seq_len // world_size

    if verify:
        k_blocks = []
        v_blocks = []
    prev_o = None
    prev_lse = None
    attn_output = None

    q = torch.ones(batch_size, chunk_len, head_num, dim)

    for step in range(world_size):
        k = torch.ones(batch_size, chunk_len, head_num, dim) * step
        v = torch.ones(batch_size, chunk_len, head_num, dim) * step

        if verify:
            k_blocks.append(k)
            v_blocks.append(v)

        out_, lse_ = memory_efficient_attention_partial(q, k, v)

        if nvtx_enable:
            merge_rng = nvtx.start_range(
                message="xformers_merge_attentions_mid", color="green"
            )

        if attn_output is None:
            prev_o = out_
            attn_output = prev_o
            prev_lse = lse_
        else:
            prev_o, prev_lse = merge_attentions(
                [attn_output, out_], [prev_lse, lse_], write_lse=True
            )
            attn_output = prev_o
        print(f"in the middle {torch.cuda.max_memory_allocated() / 1024**2} MB")

        if nvtx_enable:
            nvtx.end_range(merge_rng)

    if verify:
        verify_attn_output(q, k_blocks, v_blocks, attn_output.transpose(1, 2))


if __name__ == "__main__":
    world_size = 8
    batch_size = 1
    head_num = 32
    seq_len = 4096
    dim = 128
    verify = False

    assert seq_len % world_size == 0, "seq_len must be divisible by world_size"

    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.float16)

    # warm up
    torch_seq_parallel(
        world_size=world_size,
        batch_size=batch_size,
        head_num=head_num,
        dim=dim,
        seq_len=seq_len,
        verify=verify,
    )

    xformers_seq_parallel(
        world_size=world_size,
        batch_size=batch_size,
        head_num=head_num,
        dim=dim,
        seq_len=seq_len,
        verify=verify,
    )

    xformers_seq_parallel_merge_in_middle(
        world_size=world_size,
        batch_size=batch_size,
        head_num=head_num,
        dim=dim,
        seq_len=seq_len,
        verify=verify,
    )

    torch.cuda.reset_peak_memory_stats()
    # benchmark
    torch_seq_parallel(
        world_size=world_size,
        batch_size=batch_size,
        head_num=head_num,
        dim=dim,
        seq_len=seq_len,
        verify=verify,
        nvtx_enable=True,
    )

    torch.cuda.reset_peak_memory_stats()

    xformers_seq_parallel(
        world_size=world_size,
        batch_size=batch_size,
        head_num=head_num,
        dim=dim,
        seq_len=seq_len,
        verify=verify,
        nvtx_enable=True,
    )

    torch.cuda.reset_peak_memory_stats()

    xformers_seq_parallel_merge_in_middle(
        world_size=world_size,
        batch_size=batch_size,
        head_num=head_num,
        dim=dim,
        seq_len=seq_len,
        verify=verify,
        nvtx_enable=True,
    )
