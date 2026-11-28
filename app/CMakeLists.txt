add_executable(autograd autograd.cpp)
add_executable(mlire mlir.cpp)

if(NOT LLVM_ENABLE_RTTI)
  target_compile_options(mlire PRIVATE -fno-rtti)
endif()

target_link_libraries(autograd NNLib)
target_link_libraries(mlire NNLib)

