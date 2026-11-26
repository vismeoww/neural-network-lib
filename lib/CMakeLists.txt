
add_library(NNLib 
    Tensor.cpp
    CPU/CPUGraph.cpp
    MLIR/MLIRGraph.cpp
    MLIR/MLIROp.cpp
)
set_property(TARGET NNLib PROPERTY POSITION_INDEPENDENT_CODE ON)

# target_include_directories(my_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_include_directories(NNLib PUBLIC ${PROJECT_ROOT_DIR}/include)

target_link_libraries(NNLib PRIVATE 
  LLVMSupport
  MLIRIR
  MLIRSupport
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRTensorDialect
  MLIRArithDialect
  MLIRRegisterAllDialects
)
