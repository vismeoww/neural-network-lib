
add_library(NNLib 
    Mat.cpp 
    Node.cpp
    NNet.cpp
)
set_property(TARGET NNLib PROPERTY POSITION_INDEPENDENT_CODE ON)

# target_include_directories(my_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_include_directories(NNLib PUBLIC ${PROJECT_ROOT_DIR}/include)

