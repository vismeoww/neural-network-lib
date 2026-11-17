cmake_minimum_required(VERSION 3.10)
project(HelloNN CXX)

# set create compile commands database
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# set the root directory so that it can be used from other CMakeLists.txt files
set(PROJECT_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# setup mlir 
# 1. Find LLVM and MLIR packages
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)

# 2. Include directories
include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS} )
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_LIBRARY_DIRS})



add_subdirectory(lib)
add_subdirectory(app)
add_subdirectory(pywrap)
