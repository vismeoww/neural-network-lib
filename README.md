# HelloNN
My experiments on building a neural network library from scratch


## Build
### CXX project
```bash
cmake -B build -S . \
    -DLLVM_DIR=/path/to/llvm-project/build/lib/cmake/llvm \
    -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir \
cmake --build .
```
### Python project
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run
### CXX project
```bash
./app/hello_nn
```
### python project
```bash
python pywrap/example/ex2.py
```
