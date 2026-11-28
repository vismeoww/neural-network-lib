
add_library(NNLib 
    Tensor.cpp
    CPU/CPUGraph.cpp
    MLIR/MLIRGraph.cpp
    MLIR/MLIROp.cpp
)
set_property(TARGET NNLib PROPERTY POSITION_INDEPENDENT_CODE ON)

if(NOT LLVM_ENABLE_RTTI)
  target_compile_options(NNLib PRIVATE -fno-rtti)
endif()

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
  MLIRBuiltinToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
  MLIRExecutionEngine
)
