# optimized_triton_hash_search.py

import torch
import triton
import triton.language as tl
import numpy as np
import time
import random

# -----------------------------------------------------------------------------
# 1) CPU 参考哈希函数 (与设备端完全一致的 15 次迭代，32-bit wrap 每步)
# -----------------------------------------------------------------------------
def custom_hash_cpu(val: int) -> int:
    """CPU 版哈希函数，用于计算目标哈希值，比对时验证正确性。"""
    h = np.uint32(val)
    for _ in range(15):
        h = (h ^ np.uint32(61)) ^ (h >> np.uint32(16))
        h = (h + (h << np.uint32(3))) & np.uint32(0xFFFFFFFF)
        h = (h ^ (h >> np.uint32(4))) & np.uint32(0xFFFFFFFF)
        h = np.uint32(h * np.uint32(0x27d4eb2d)) & np.uint32(0xFFFFFFFF)
        h = (h ^ (h >> np.uint32(15))) & np.uint32(0xFFFFFFFF)
    return int(h)


# -----------------------------------------------------------------------------
# 2) GPU 端 Inline Hash: 15 次迭代完全展开，避免 for-loop 开销
# -----------------------------------------------------------------------------
@triton.jit
def inline_hash15(x: tl.uint32) -> tl.uint32:
    # --- Iteration 1 ---
    x = (x ^ 61) ^ (x >> 16)
    x = x + (x << 3)
    x = x ^ (x >> 4)
    x = x * 0x27d4eb2d
    x = x ^ (x >> 15)
    # --- Iteration 2 ---
    x = (x ^ 61) ^ (x >> 16)
    x = x + (x << 3)
    x = x ^ (x >> 4)
    x = x * 0x27d4eb2d
    x = x ^ (x >> 15)
    # --- Iteration 3 ---
    x = (x ^ 61) ^ (x >> 16)
    x = x + (x << 3)
    x = x ^ (x >> 4)
    x = x * 0x27d4eb2d
    x = x ^ (x >> 15)
    # --- Iteration 4 ---
    x = (x ^ 61) ^ (x >> 16)
    x = x + (x << 3)
    x = x ^ (x >> 4)
    x = x * 0x27d4eb2d
    x = x ^ (x >> 15)
    # --- Iteration 5 ---
    x = (x ^ 61) ^ (x >> 16)
    x = x + (x << 3)
    x = x ^ (x >> 4)
    x = x * 0x27d4eb2d
    x = x ^ (x >> 15)
    # --- Iteration 6 ---
    x = (x ^ 61) ^ (x >> 16)
    x = x + (x << 3)
    x = x ^ (x >> 4)
    x = x * 0x27d4eb2d
    x = x ^ (x >> 15)
    # --- Iteration 7 ---
    x = (x ^ 61) ^ (x >> 16)
    x = x + (x << 3)
    x = x ^ (x >> 4)
    x = x * 0x27d4eb2d
    x = x ^ (x >> 15)
    # --- Iteration 8 ---
    x = (x ^ 61) ^ (x >> 16)
    x = x + (x << 3)
    x = x ^ (x >> 4)
    x = x * 0x27d4eb2d
    x = x ^ (x >> 15)
    # --- Iteration 9 ---
    x = (x ^ 61) ^ (x >> 16)
    x = x + (x << 3)
    x = x ^ (x >> 4)
    x = x * 0x27d4eb2d
    x = x ^ (x >> 15)
    # --- Iteration 10 ---
    x = (x ^ 61) ^ (x >> 16)
    x = x + (x << 3)
    x = x ^ (x >> 4)
    x = x * 0x27d4eb2d
    x = x ^ (x >> 15)
    # --- Iteration 11 ---
    x = (x ^ 61) ^ (x >> 16)
    x = x + (x << 3)
    x = x ^ (x >> 4)
    x = x * 0x27d4eb2d
    x = x ^ (x >> 15)
    # --- Iteration 12 ---
    x = (x ^ 61) ^ (x >> 16)
    x = x + (x << 3)
    x = x ^ (x >> 4)
    x = x * 0x27d4eb2d
    x = x ^ (x >> 15)
    # --- Iteration 13 ---
    x = (x ^ 61) ^ (x >> 16)
    x = x + (x << 3)
    x = x ^ (x >> 4)
    x = x * 0x27d4eb2d
    x = x ^ (x >> 15)
    # --- Iteration 14 ---
    x = (x ^ 61) ^ (x >> 16)
    x = x + (x << 3)
    x = x ^ (x >> 4)
    x = x * 0x27d4eb2d
    x = x ^ (x >> 15)
    # --- Iteration 15 ---
    x = (x ^ 61) ^ (x >> 16)
    x = x + (x << 3)
    x = x ^ (x >> 4)
    x = x * 0x27d4eb2d
    x = x ^ (x >> 15)

    return x


# -----------------------------------------------------------------------------
# 3) 精简版 Triton Kernel：用 vectorized reduction 找到“第一个匹配”，避免 Python loop
# -----------------------------------------------------------------------------
@triton.jit
def gpu_bruteforce_no_reduce(
    target_hash: tl.uint32,     # scalar, 要匹配的目标哈希
    max_val: tl.uint32,         # scalar, 搜索上限
    result_ptr: tl.tensor,      # device ptr, 存放找到的结果 idx
    found_flag_ptr: tl.tensor,  # device ptr, 标志位，0=未找到，1=已找到
    start_idx: tl.uint32,       # scalar, 搜索起始偏移（通常传 0）
    BLOCK_SIZE: tl.constexpr,   # 编译期常量，block 大小（如 512）
):
    pid = tl.program_id(0)
    base = start_idx + pid * BLOCK_SIZE
    offsets = base + tl.arange(0, BLOCK_SIZE)                     # shape = (BLOCK_SIZE,)
    in_bounds_mask = offsets <= max_val                             # 布尔向量

    # 计算哈希
    hash_vec = inline_hash15(tl.cast(offsets, tl.uint32))          # shape = (BLOCK_SIZE,)

    # matches[i] = True 当 offsets[i]≤max_val 且 hash_vec[i]==target_hash
    matches = in_bounds_mask & (hash_vec == target_hash)            # 布尔向量

    # 如果这一 block 没有任何匹配，就直接 return
    cnt = tl.sum(tl.cast(matches, tl.int32), axis=0)               # 标量
    if cnt == 0:
        return

    # 至此，block 内至少有一个匹配。我们要找“第一个”匹配 lane。
    # 先构造一个向量：matched_offsets[i] = offsets[i] if matches[i] else (最大 uint32)
    big = 0xFFFFFFFF                                 
    matched_offsets = tl.where(matches, offsets, big)              # shape = (BLOCK_SIZE,)

    # min 会拿到最小那个匹配的 offsets
    first_match = tl.min(matched_offsets, axis=0)                  # 标量

    # 原子写回：只有第一个线程成功
    if tl.atomic_cas(found_flag_ptr, 0, 1) == 0:
        tl.store(result_ptr, first_match)


# -----------------------------------------------------------------------------
# 4) 主机端 Launch 函数：固定 BLOCK_SIZE=512，不做 autotune；使用 CUDA Event 精确计时
# -----------------------------------------------------------------------------
def triton_bruteforce_no_reduce(target_hash: int, max_val: int, start_idx: int = 0):
    """
    在 GPU 上做暴力搜索：
      - target_hash: 要匹配的哈希值
      - max_val: 搜索范围 [0..max_val]
      - start_idx: 可以指定从非 0 开始（默认为 0）
    返回： (found_idx, elapsed_ms)
        found_idx = 匹配到的最小 idx，如果没找到返回 -1
        elapsed_ms = GPU kernel 执行时间（毫秒）
    """
    # 4.1) 在 GPU 上开辟输出缓冲
    device_res  = torch.full((1,), 0xFFFFFFFF, dtype=torch.uint32, device='cuda')
    device_flag = torch.zeros((1,),    dtype=torch.int32,  device='cuda')

    # 4.2) 计算搜索空间大小、BLOCK_SIZE 和所需 blocks 数
    search_space = max_val - start_idx + 1
    BLOCK_SIZE   = 512  # 经验值：大多数 GPU 下 512-lane 能兼顾寄存器占用和吞吐
    num_blocks   = (search_space + BLOCK_SIZE - 1) // BLOCK_SIZE

    # 4.3) 设定 CUDA Event 来精确测时
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt   = torch.cuda.Event(enable_timing=True)

    # 4.4) Launch kernel 并测时
    start_evt.record()
    gpu_bruteforce_no_reduce[(num_blocks,)](
        target_hash,
        max_val,
        device_res,
        device_flag,
        start_idx,
        BLOCK_SIZE=BLOCK_SIZE
    )
    end_evt.record()
    torch.cuda.synchronize()
    elapsed_ms = start_evt.elapsed_time(end_evt)

    # 4.5) 拷回结果
    found = int(device_res.item())
    if found == 0xFFFFFFFF:
        return -1, elapsed_ms
    return found, elapsed_ms


# -----------------------------------------------------------------------------
# 5) 主程序：示例如何调用，并验证结果
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # --- 5.1) 配置搜索范围，随意调整以便测试性能 ---
    MAX_CHECK = 2000_000_000   # 1e8 范围以内测试可在几百 ms 完成；可改成更大值
    SOURCE    = 142385552
    target    = custom_hash_cpu(SOURCE)

    print(f"目标哈希: 0x{target:08x}")
    print(f"要查找的源值: {SOURCE}")
    print(f"搜索范围: [0 .. {MAX_CHECK}]\n")

    # --- 5.2) GPU (Triton) 暴力搜索 ---
    print("▶️  开始 GPU（Triton）暴力搜索 ...")
    found, gpu_time = triton_bruteforce_no_reduce(target, MAX_CHECK, 0)
    if found != -1:
        print(f"✅ GPU 找到源值: {found}  (哈希: 0x{custom_hash_cpu(found):08x})")
    else:
        print("❌ GPU: 范围内未找到源值")
    print(f"GPU 搜索耗时: {gpu_time:.3f} ms\n")

    # --- 5.3) 对于中小范围，可选择运行 CPU 暴力搜索验证 ---
    if MAX_CHECK <= 20_000_000:
        print("▶️  开始 CPU 暴力搜索 ...")
        t0 = time.time()
        cpu_found = -1
        for i in range(MAX_CHECK + 1):
            if custom_hash_cpu(i) == target:
                cpu_found = i
                break
        cpu_time = (time.time() - t0) * 1000
        if cpu_found != -1:
            print(f"✅ CPU 找到源值: {cpu_found}")
        else:
            print("❌ CPU: 范围内未找到源值")
        print(f"CPU 搜索耗时: {cpu_time:.3f} ms\n")
    else:
        print("⚠️ 跳过 CPU 搜索 (范围过大)\n")

    # --- 5.4) 验证正确性 ---
    if found != -1:
        assert custom_hash_cpu(found) == target, "错误：GPU 找到的值哈希不匹配！"
        if found == SOURCE:
            print("🔍 验证通过：GPU 找到的源值与原始 SOURCE 完全一致。")
        else:
            print("⚠️ GPU 找到的源值与原始不一致，可能存在哈希碰撞。")
    print("\nDone.")
