# Cutlass Hash Search Example Project

This project demonstrates a brute-force hash search using CUDA on NVIDIA GPUs. It implements a custom 32-bit hashing algorithm that is consistent across multiple implementations (CUTLASS, Triton, PyCUDA), and performs a parallel search over a large range of integers to find a match for a given target hash.

While CUTLASS itself is not directly used in the hash logic in this example, it is included as a dependency to showcase integration with other high-performance GPU libraries. The focus is on comparing execution times and understanding GPU compute capabilities through CUDA.

---

## 📋 System Requirements

- **Operating System**: Linux (Ubuntu 20.04 or newer recommended)
- **CUDA Toolkit**: 12.x
- **GPU**: CUDA-capable NVIDIA GPU (e.g., Tesla T4, RTX 30/40 series)
- **Python Version**: 3.8+
- **Dependencies**:
  - `torch`
  - `triton`
  - `pycuda`
  - `cupy-cuda12x`

---

## 🧰 Software Dependencies

Install the required Python packages:

```bash
pip install torch triton pycuda cupy-cuda12x
```

---

## 🚀 Project Structure

```
cutlass-hash-search/
│
├── cutlass_hash_search.cu     # CUTLASS implementation of hash search
├── cutlass_hash_search.py     # Python wrapper interface
├── triton_hash_search.py      # Triton implementation of hash search
├── triton_hash_search_optimized.py  # Optimized version of Triton implementation
├── pycuda_hash_search.py      # PyCUDA implementation of hash search
├── cpu_hash_search.py         # CPU-only implementation (very slow)
├── README.md                  # This file
```

---

## 🛠️ Build & Run

### 1. Install CUTLASS

Clone the official [CUTLASS GitHub repository](https://github.com/NVIDIA/cutlass):

```bash
git clone https://github.com/NVIDIA/cutlass.git
```

> **Note:** Make sure to place the `cutlass` directory in a suitable location and set the `-I` include path accordingly when compiling.

---

### 2. Compile the CUTLASS Hash Search Program

Modify the following command to match your local `cutlass` path:

```bash
nvcc -std=c++17 -I/path/to/cutlass/include -lcuda -lcudart -o cutlass_hash_search cutlass_hash_search.cu
```

Then run the compiled binary:

```bash
./cutlass_hash_search
```

---

### 3. Run Python Scripts

#### Using Triton

```bash
python triton_hash_search.py
# For optimized version:
python triton_hash_search_optimized.py
```

#### Using PyCUDA

```bash
python pycuda_hash_search.py
```

#### Using CPU (slow)

```bash
python cpu_hash_search.py
```

---

## 📊 Performance Comparison

You can use this project to compare execution times across different implementations (CUTLASS, Triton, PyCUDA, CPU). This is useful for benchmarking and understanding performance trade-offs on various hardware setups.

| Implementation   | Language | Time（ms） |
| ---------------- | -------- | -------- |
| CPU              | Python   | 10471763 |
| CPU              | C++      | 24482    |
| Triton           | Python   | 3699     |
| Triton_optimized | Python   | 1831     |
| PyCuda           | Python   | 101      |
| CUDA             | C++      | 78       |
| Cutlass          | Python   | 76       |
| Cutlass          | C++      | 72       |