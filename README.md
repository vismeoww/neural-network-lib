# HelloNN

A simple C++ project using CMake.

## Structure

*   `app/`: Contains the executable code.
*   `lib/`: Contains the library code.
*   `include/`: Contains the library header files.

## Build
### CXX project
```bash
cmake -B build -S .
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
./pywrap/example.py
```
