# Python/Rust/CUDA ML Framework
A custom machine learning framework that is implemented in Rust and CUDA for optimized and parallel processing. This framework uses Python as an interface to the functions in Rust for convenience and flexibility in creating custom Python classes that utilizes such functions on the Rust/CUDA backend.

List of Contents:
1. [Dependencies](#dependencies)
2. [Project Building Steps](#project-building-steps)
3. [Testing](#testing)
4. [Acknowledgements](#acknowledgements)

## Dependencies
***NOTE: This project is tested and currently only compatible with Windows 11.***

### Python 3.11.0
Third party Python libraries required:
- numpy==1.24.3

Third party Python libraries can be installed by running:  
`pip install -r requirements.txt`

### Rust 1.90.0
Rust crates required (crates are already included in Cargo.toml):
- serde = { version = "1.0", features = ["derive"] }
- serde_json = "1.0"
- rand = "0.8.5"

Rust and its dependencies can be installed for either Windows or Linux by following the instructions from: https://rust-lang.org/tools/install/.  
After Rust is installed, version 1.90.0 can be installed by running the following commands
in terminal:
```
# Install Rust version 1.90.0.
rustup install 1.90.0

# Set Rust version to 1.90.0.
rustup override set 1.90.0
```

### NVIDIA CUDA
Machine learning models rely on custom CUDA kernels to make use of parallel processing to improve performance and inference. Versions required for NVIDIA CUDA Toolkit, GPU and Driver is provided below.
- Operating System: Windows 11
- NVIDIA GPU Compute Capability: 8.9 (Tested on RTX 4050.)
- CUDA Toolkit version: v12.5
- NVIDIA Driver version: 581.57
- *CUDA Toolkit v12.5 is not included in this project and can be installed from the link below:*  
https://developer.nvidia.com/cuda-12-5-0-download-archive

## Project Building Steps
1. Create the directory `./test/nnpackage/backend` which is required to store all dynamic libraries for this project.
2. In the project's root directory `./` run `buildcuda.bat` and `buildrust.bat` to create the `cuda_backend.dll` and `rust_backend.dll` dynamic libraries respectively. These libraries contain functions that handle the main computation of machine learning models and can be interfaced via Python.
3.  - Locate the directory where CUDA v12.5 is installed (e.g. `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5`), search for the `bin\` subdirectory and copy the following binaries to `./test/nnpackage/backend`:  
        - cublas64_12.dll
        - cublasLt64_12.dll  
    - These binaries provide dependencies for the `cuda_backend.dll` and `rust_backend.dll` libraries and need to be in the same directory so such dependencies can be located.

## Testing

- To test machine learning models a default `test.py` Python script is included in the `./test` directory.  
- This script trains a small feedforward neural network on randomized data to test forward/backward propagation and ensure data flows correctly. 
- This script can be executed using the following command in the `./test` directory:
```
python test.py
```

## Acknowledgements

Python libraries used:
- [NumPy](https://numpy.org/) — Python library for numerical computing, handling of arrays and linear algebra.
    - License type: BSD 3-Clause
    - Link: https://github.com/numpy/numpy/blob/main/LICENSE.txt

Rust crates used:
- [Rand](https://github.com/rust-random/rand) — Required for generating random parameter values for neuron edges between the low and high bounds specified in JSON hyperparameter files.
    - License type: MIT
    - Link: https://github.com/rust-random/rand/blob/master/LICENSE-MIT

- [Serde](https://serde.rs/) — Required for Serde JSON to save and load neural network parameters which is necessary for pausing/resuming training progress and inference.
    - License type: MIT
    - Link: https://github.com/serde-rs/serde/blob/master/LICENSE-MIT

- [Serde JSON](https://github.com/serde-rs/json) — Depends on Serde to save and load neural network parameters to and from JSON files.
    - License type: MIT
    - Link: https://github.com/serde-rs/json/blob/master/LICENSE-MIT