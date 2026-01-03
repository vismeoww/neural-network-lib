# pywrap/CMakeLists.txt

# 1. Force CMake to prefer the active virtual environment
set(Python3_FIND_VIRTUALENV FIRST)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# 2. Fetch pybind11 locally (no install needed)
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.13.6
)
FetchContent_MakeAvailable(pybind11)

# 3. Create the module
pybind11_add_module(pynn wrapper.cpp)

# 4. Link to your core C++ library from the /src directory
target_link_libraries(pynn PRIVATE NNLib)

# 5. Move the finished .so/.pyd file into pywrap for easy access
set_target_properties(pynn PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
set_target_properties(pynn PROPERTIES PREFIX "")
