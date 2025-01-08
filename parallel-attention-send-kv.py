import torch
import torch.nn.functional as F
import torch.distributed as dist
from xformers.ops.fmha import memory_efficient_attention_partial, merge_attentions
from xformers.ops import LowerTriangularMask
import math
import nvtx
from utils import RingComm
import os
import time


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
        query=q_blocks.transpose(1, 2),
        key=k_blocks.transpose(1, 2),
        value=v_blocks.transpose(1, 2),
        is_causal=True,
    )
    torch.testing.assert_close(golden_output[:, :], attn_output[:, :], atol=1e-3, rtol=1e-3)


def seq_parallel_send_kv(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
):
    comm = RingComm(process_group)

    next_k, next_v = None, None

    o_blocks = []
    lse_values = []

    for step in range(world_size):
        if step + 1 != comm.world_size:
            next_k: torch.Tensor = comm.send_recv(k)
            next_v: torch.Tensor = comm.send_recv(v)
            comm.commit()

        kv_origin = (rank - step) % world_size

        if kv_origin == rank:
            out_, lse_ = memory_efficient_attention_partial(q, k, v, attn_bias=LowerTriangularMask())
        elif kv_origin < rank:
            out_, lse_ = memory_efficient_attention_partial(q, k, v)

        if kv_origin <= rank:
            # attn_out is in the shape of [B, M, num of heads, head_dim]
            o_blocks.append(out_)

            # LSE is in the shape of [B, num of heads, M]
            lse_values.append(lse_)

        if step + 1 != comm.world_size:
            comm.wait()
            k = next_k
            v = next_v

    if rank > 0:
        with torch.cuda.device(q.device.index):
            attn_output, _ = merge_attentions(o_blocks, lse_values, write_lse=False)
    else:
        attn_output = o_blocks[0]

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
    q = torch.rand(batch_size, chunk_len, head_num, dim)
    k = torch.rand(batch_size, chunk_len, head_num, dim)
    v = torch.rand(batch_size, chunk_len, head_num, dim)

    # warm up
    attn_output = seq_parallel_send_kv(
        process_group=default_process_group,
        q=q,
        k=k,
        v=v,
    )

    dist.barrier()

    # torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start_time = time.time()
    attn_output = seq_parallel_send_kv(
        process_group=default_process_group,
        q=q,
        k=k,
        v=v,
    )

    torch.cuda.synchronize()
    end_time = time.time()

    print(f"RANK {rank}: Time taken: {(end_time - start_time) * 1e3} ms")

    attn_output = gather_attn_output(attn_output)

    q_blocks = gather_attn_output(q)
    print("q colleted")
    k_blocks = gather_attn_output(k)
    print("k colleted")
    v_blocks = gather_attn_output(v)
    print("v colleted")
    if rank == 0:
        attn_output = torch.cat(attn_output, dim=1).transpose(1, 2)
        verify_attn_output(q_blocks, k_blocks, v_blocks, attn_output)

    dist.barrier()
    dist.destroy_process_group()
