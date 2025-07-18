cmake_minimum_required(VERSION 3.10)
project(tvm_hpc)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_C_COMPILER "clang")
set(CMAKE_CXX_COMPILER "clang++")

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -gdwarf-4")

#omp option
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
# LLVM 및 Clang 경로 설정
find_package(LLVM REQUIRED CONFIG)
find_package(Clang REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)


include_directories(${Python3_INCLUDE_DIRS})
include_directories(/usr/local/lib/clang/19/include)
include_directories(${LLVM_INCLUDE_DIRS} ${CLANG_INCLUDE_DIRS})



link_directories(${LLVM_LIBRARY_DIRS})
add_definitions(${LLVM_DEFINITIONS})



# 실행 파일 생성
add_executable(tvm_hpc tvm_hpc.cpp)
add_executable(call_tvm_module call_tvm_module.cpp)

set(TVM_HOME /root/tvm) # TVM이 설치된 경로로 변경하세요
include_directories(${TVM_HOME}/include)
include_directories(${TVM_HOME}/3rdparty/dmlc-core/include)
include_directories(${TVM_HOME}/3rdparty/dlpack/include)
include_directories(${TVM_HOME}/3rdparty/rang/include)
include_directories(${TVM_HOME}/src/target/spirv)
link_directories(${TVM_HOME}/build)


# 라이브러리 링크
target_link_libraries(tvm_hpc
    clangTooling
    clangFrontend
    clangAST
    clangASTMatchers
    clangSerialization
    clangRewrite
    clangBasic
    clangCodeGen 
    LLVMCore
    LLVMSupport
    LLVMFrontendOpenMP   # OpenMP 라이브러리 명시
    clangLex
    ${TVM_HOME}/build/libtvm.so 
)

target_link_libraries(call_tvm_module
    ${TVM_HOME}/build/libtvm.so 
)