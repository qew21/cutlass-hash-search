import os
import cupy as cp
import numpy as np

# ----------------------------------------
# 1) 确定 CUTLASS 路径
# ----------------------------------------
cutlass_path = "/root/cutlass/include"
if cutlass_path is None or not os.path.isdir(cutlass_path):
    raise RuntimeError("请先 clone CUTLASS 并设置环境变量 CUTLASS_PATH 指向它的根目录")

# ----------------------------------------
# 2) CUDA kernel 源码（与之前一致）
# ----------------------------------------
kernel_code = r'''
#include <cutlass/cutlass.h>               // CUTLASS 核心
#include <cutlass/platform/platform.h>

extern "C" __global__ void brute_cutlass(
    unsigned int target_hash,
    unsigned int max_val,
    unsigned int* found_src,
    int* found_flag
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= max_val) {
        unsigned int h = idx;
        for (int i = 0; i < 15; ++i) {
            h = ((h ^ 61) ^ (h >> 16)) & 0xFFFFFFFF;
            h = (h + (h << 3)) & 0xFFFFFFFF;
            h = (h ^ (h >> 4)) & 0xFFFFFFFF;
            h = (h * 0x27d4eb2d) & 0xFFFFFFFF;
            h = (h ^ (h >> 15)) & 0xFFFFFFFF;
        }
        if (h == target_hash) {
            if (atomicCAS(found_flag, 0, 1) == 0) {
                *found_src = idx;
            }
        }
    }
}
'''

# ----------------------------------------
# 3) 构建 RawModule，传入正确的 include 路径
# ----------------------------------------
module = cp.RawModule(
    code=kernel_code,
    backend='nvcc',
    options=(
        f'-I/root/cutlass/include',
        '-std=c++17',
    ),
)
kernel = module.get_function('brute_cutlass')

# ----------------------------------------
# 4) 运行函数
# ----------------------------------------
def run_with_cutlass(target_hash, max_val, threads_per_block=256):
    found_src  = cp.array([-1], dtype=cp.uint32)
    found_flag = cp.array([0],  dtype=cp.int32)

    blocks = (int(max_val) + threads_per_block) // threads_per_block

    start = cp.cuda.Event(); end = cp.cuda.Event()
    start.record()
    kernel(
        (blocks, 1, 1), (threads_per_block, 1, 1),
        (np.uint32(target_hash), np.uint32(max_val),
         found_src, found_flag)
    )
    end.record(); end.synchronize()
    # 计算 elapsed ms
    elapsed_ms = cp.cuda.get_elapsed_time(start, end)
    return int(found_flag.get()), int(found_src.get()), elapsed_ms


# ----------------------------------------
# 5) 测试主流程
# ----------------------------------------
if __name__ == "__main__":
    def custom_hash_python(val):
        h = val & 0xFFFFFFFF
        for _ in range(15):
            h = ((h ^ 61) ^ (h >> 16)) & 0xFFFFFFFF
            h = (h + (h << 3)) & 0xFFFFFFFF
            h = (h ^ (h >> 4)) & 0xFFFFFFFF
            h = (h * 0x27d4eb2d) & 0xFFFFFFFF
            h = (h ^ (h >> 15)) & 0xFFFFFFFF
        return h

    SOURCE    = 142385552
    MAX_CHECK = 2000_000_000
    target    = custom_hash_python(SOURCE)

    print(f"Target hash: 0x{target:08x}, searching for {SOURCE}")
    flag, src, ms = run_with_cutlass(target, MAX_CHECK)
    if flag:
        print(f"✅ Found source = {src}")
    else:
        print("❌ Not found in range")
    print(f"CUTLASS-style kernel time: {ms:.3f} ms")
