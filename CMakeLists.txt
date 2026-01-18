cmake_minimum_required(VERSION 3.10)
project(HelloNN CXX)

# set create compile commands database
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# set the root directory so that it can be used from other CMakeLists.txt files
set(PROJECT_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})


add_subdirectory(lib)
add_subdirectory(app)
# we will deal with this later
# add_subdirectory(pywrap)
