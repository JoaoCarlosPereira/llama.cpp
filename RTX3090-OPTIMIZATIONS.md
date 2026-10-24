# RTX 3090 / Ampere (SM 8.6) Performance Optimizations for llama.cpp

This branch (`rtx3090`) contains dedicated kernel and runtime engine optimizations tailored specifically for NVIDIA GeForce RTX 3090 (Ampere GA102, Compute Capability 8.6, 24 GB GDDR6X) and similar modern NVIDIA GPUs, targeting maximum decode and prefill throughput on large hybrid architectures (such as **Qwen 3.8 27B**) under **massive context windows (204.8k tokens)** with **Multi-Token Prediction (MTP) Speculative Decoding**.

---

## 1. Key Architectural Enhancements

### 1. Ampere 8-Warp MMVQ Fastpath (`ggml/src/ggml-cuda/mmvq.cu`)
- **Problem:** Upstream NVIDIA commit `#26843` introduced `MMVQ_PARAMETERS_GB10` (8 warps = 256 threads per thread block for `ncols_dst == 1`, halving K-loop iterations for quantized matrix-vector multiplication), but restricted it exclusively to `__CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK`. Ampere (`sm_86`) cards fell back to `MMVQ_PARAMETERS_GENERIC` (4 warps = 128 threads), underutilizing the 82 Streaming Multiprocessors of the RTX 3090.
- **Solution:** Routed `__CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE` directly to `MMVQ_PARAMETERS_GB10` in both host and device dispatch tables.
- **Impact:** Immediate +3.8% dense token generation speedup and higher SM occupancy.

### 2. Dedicated Speculative Decoding Graph Scheduler Slots (`src/llama-context.cpp`, `src/llama-context.h`)
- **Problem:** Speculative decoding with Multi-Token Prediction (MTP) constantly alternates between draft token generation (`batch = 1`, 6x steps) and verify passes (`batch = n_draft + 1 = 7`, 1x step). Upstream `llama.cpp` maintains only a single `ggml_backend_sched` compute graph allocation; each change in batch size invalidates the cached graph, forcing the host CPU to rebuild the graph, re-run memory layout algorithms, re-reserve buffers, and re-capture CUDA graphs on every single token.
- **Solution:** Integrated Patch 22 (`decode-sched-slots`) updated for the modern `copy_tensor_async_rows` and `sched_active()` APIs. A dedicated scheduler slot caches the allocation for `n_tokens == 1`.
- **Memory Safety & Sizing:** Controlled via environment variables:
  - `LLAMA_DEC_SLOTS=1` (allocates 1 dedicated decode slot)
  - `LLAMA_DEC_MAX_TOK=1` (routes `n_tokens == 1` draft steps into the dedicated slot)
  This adds only **7.8 MiB** of VRAM overhead on a 24 GB GPU, giving 100% graph reuse during the 6 draft steps per cycle with zero OOM risk across long contexts up to 204.8k tokens.

### 3. DeltaNet / GDN Operator Fusion (Qwen 3.8 Hybrid Architecture)
- Includes the full suite of DeltaNet operator fusions:
  - `fuse-gdn-beta-sigmoid`
  - `fuse-gdn-state-gather`
  - `gdn-gather-single-snapshot`
  - `fuse-concat-gather`
  - `fuse-add-unary-mul`
  - `concat-non-cont-flat`
- Fuses multiple element-wise memory passes into single register-resident CUDA operations.

### 4. Reduced MTP Draft Vocabulary (`LLAMA_MTP_DRAFT_VOCAB`)
- Reduces LM head projection gather during MTP draft steps from 546 MB to ~83 MB using a pre-filtered high-frequency vocabulary file (`qwen_draft_vocab.txt`).

---

## 2. Recommended Compilation Flags

```bash
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="86" \
    -DGGML_CUDA_COMPRESSION_MODE=speed \
    -DGGML_CUDA_FA_ALL_QUANTS=ON \
    -DGGML_CUDA_GRAPHS=ON \
    -DCMAKE_CUDA_FLAGS="-O3 --use_fast_math -Xptxas -O3" \
    -DCMAKE_CXX_FLAGS="-O3 -march=native -mtune=native"

cmake --build build --config Release -j$(nproc)
```

---

## 3. Verified Benchmark Results (RTX 3090, 24 GB)

Workload: **Qwen 3.8 27B UD-Q4_K_M** + **MTP Draft Q4_0**  
Context Window: **204,800 tokens** (`--cache-type-k q4_0 --cache-type-v q4_0 --flash-attn on`)

| Context Depth | Prompt Tokens | Upstream Stock (`b10353`) | Optimized RTX 3090 Build | Improvement | Draft Acceptance | Sustained Prefill |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Short (~1.5k)** | 2,019 tok | 58.0 tok/s | **68.3 tok/s** | **+17.8%** | 95.5 % | 1,071.7 tok/s |
| **Medium (~7k)** | 9,020 tok | 52.0 tok/s | **59.4 tok/s** | **+14.2%** | 69.2 % | 1,211.6 tok/s |
| **Long (~28k)** | 28,070 tok | 47.0 tok/s | **63.2 tok/s** | **+34.5%** | 90.9 % | 1,136.1 tok/s |
| **Ultra-Long (~49k)**| 49,020 tok | 42.0 tok/s | **54.0 tok/s** | **+28.6%** | 90.5 % | 1,026.2 tok/s |
