# Python/Rust/CUDA ML Framework
A custom machine learning framework that is implemented in Rust and CUDA C++ for optimized and parallel processing. This framework uses Python as an interface to the functions in Rust for convenience and flexibility in creating custom Python classes that utilizes such functions on the Rust/CUDA C++ backend.

List of Contents:
1. [Dependencies](#dependencies)
2. [Project Building Steps](#project-building-steps)
3. [Testing](#training-and-testing)
4. [Acknowledgements](#acknowledgements)

## Dependencies
### Compatible Operating System
- ***This project is currently only compatible with Windows 11.***

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

### NVIDIA CUDA C++
- NVIDIA CUDA Version required: **12.5**
- CUDA 12.5 is not included in this project and can be installed from the link below:  
https://developer.nvidia.com/cuda-12-5-0-download-archive
