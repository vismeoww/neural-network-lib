add_library(NNLib NeuralNet.cpp)

# target_include_directories(my_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_include_directories(NNLib PUBLIC ${PROJECT_ROOT_DIR}/include)
target_include_directories(NNLib PUBLIC ${PROJECT_ROOT_DIR}/eigen)

