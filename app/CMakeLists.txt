add_executable(hello_nn main.cpp)
add_executable(autograd autograd.cpp)

target_link_libraries(hello_nn NNLib)
target_link_libraries(autograd NNLib)
