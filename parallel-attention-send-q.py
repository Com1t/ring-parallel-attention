import torch
import torch.nn.functional as F
import torch.distributed as dist
from xformers.ops.fmha import memory_efficient_attention_partial, merge_attentions
import math
import nvtx
from utils import RingComm
import os
import time

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


# Returns the result of running `fn()` and the time it took for `fn()` to run,
# in milliseconds. We use CUDA events and synchronization for the most accurate
# measurements.
def timed(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end)


def gather_attn_output(attn_output):
    if dist.get_rank() == 0:
        gathered_values = [
            torch.empty_like(attn_output) for _ in range(dist.get_world_size())
        ]
    else:
        gathered_values = None
    dist.gather(attn_output, gathered_values, dst=0)

    return gathered_values


def verify_attn_output(
    q_blocks: torch.Tensor, k_blocks: list, v_blocks: list, attn_output: torch.Tensor
):
    q_blocks = torch.cat(q_blocks, dim=1)
    k_blocks = torch.cat(k_blocks, dim=1)
    v_blocks = torch.cat(v_blocks, dim=1)
    golden_output = F.scaled_dot_product_attention(
        query=q.transpose(1, 2),
        key=k_blocks.transpose(1, 2),
        value=v_blocks.transpose(1, 2),
        is_causal=False,
    )
    torch.testing.assert_close(golden_output, attn_output, atol=1e-3, rtol=1e-3)


@torch.compile
def torch_attention_partial(query, key, value, attn_mask=None):
    """Perform partial sdpa attention with given inputs"""
    head_dim = query.size(-1)

    # Scaled Dot-Product Attention
    attn_weights = torch.matmul(
        query.transpose(1, 2), key.permute(0, 2, 3, 1)
    ) / math.sqrt(head_dim)

    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask

    # Softmax with numerical stability
    max_vals = torch.max(attn_weights, dim=-1, keepdim=True).values
    exp_weights = torch.exp(attn_weights - max_vals)
    sum_exp_weights = torch.sum(exp_weights, dim=-1, keepdim=True)
    attn_weights = exp_weights / sum_exp_weights
    lse = torch.log(sum_exp_weights) + max_vals

    output = torch.matmul(attn_weights, value.transpose(1, 2))

    return output.transpose(1, 2).contiguous(), lse.squeeze(-1).to(
        torch.float32
    ).contiguous()


def seq_parallel_send_q(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
):
    comm = RingComm(process_group)

    next_q = None

    o_blocks = []
    lse_values = []

    for step in range(world_size):
        if step + 1 != comm.world_size:
            next_q: torch.Tensor = comm.send_recv(q)
            comm.commit()

        out_, lse_ = torch_attention_partial(q, k, v)

        # attn_out is in the shape of [B, M, num of heads, head_dim]
        o_blocks.append(out_)

        # LSE is in the shape of [B, num of heads, M]
        lse_values.append(lse_)
        # print(f"once {torch.cuda.max_memory_allocated() / 1024**2} MB")

        if step + 1 != comm.world_size:
            comm.wait()
            q = next_q

    recv_buf = [torch.zeros_like(temp) for temp in o_blocks]

    dist.all_to_all(recv_buf, o_blocks)

    o_blocks = recv_buf

    recv_buf = [torch.zeros_like(temp) for temp in lse_values]

    dist.all_to_all(recv_buf, lse_values)

    lse_values = recv_buf

    with torch.cuda.device(q.device.index):
        attn_output, _ = merge_attentions(o_blocks, lse_values, write_lse=False)

    return attn_output


if __name__ == "__main__":
    batch_size = 1
    head_num = 32
    seq_len = 1024
    dim = 128
    verify = True

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["LOCAL_RANK"])

    if not dist.is_initialized():
        dist.init_process_group("nccl")
    default_process_group = dist.group.WORLD

    device = torch.device(f"cuda:{rank}")
    torch.set_default_device(device)
    torch.set_default_dtype(torch.float16)

    assert seq_len % world_size == 0, "seq_len must be divisible by world_size"

    chunk_len = seq_len // world_size
    q = torch.ones(batch_size, chunk_len, head_num, dim) * (rank + 1)
    k = torch.ones(batch_size, chunk_len, head_num, dim) * (rank + 1)
    v = torch.ones(batch_size, chunk_len, head_num, dim) * (rank + 1)

    attn_output = seq_parallel_send_q(
        process_group=default_process_group,
        q=q,
        k=k,
        v=v,
    )

    # torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start_time = time.time()
    attn_output = seq_parallel_send_q(
        process_group=default_process_group,
        q=q,
        k=k,
        v=v,
    )
    torch.cuda.synchronize()
    end_time = time.time()

    print(f"RANK {rank}: Time taken: {(end_time - start_time) * 1e3} ms")

    gather_attn_output(attn_output)

    q_blocks = []
    k_blocks = []
    v_blocks = []
    if rank == 0:
        torch.cat([attn_output], dim=1)
        for rank in range(world_size):
            q = torch.ones(batch_size, chunk_len, head_num, dim) * (rank + 1)
            k = torch.ones(batch_size, chunk_len, head_num, dim) * (rank + 1)
            v = torch.ones(batch_size, chunk_len, head_num, dim) * (rank + 1)
            q_blocks.append(q)
            k_blocks.append(k)
            v_blocks.append(v)

        verify_attn_output(q_blocks, k_blocks, v_blocks, attn_output.transpose(1, 2))

    dist.barrier()
    dist.destroy_process_group()
