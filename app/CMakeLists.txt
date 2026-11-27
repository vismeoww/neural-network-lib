add_executable(autograd autograd.cpp)
add_executable(mlire mlir.cpp)

target_link_libraries(autograd NNLib)
target_link_libraries(mlire NNLib)

