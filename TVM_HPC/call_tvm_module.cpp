#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <dlfcn.h>
#include <iostream>
#include <omp.h>
#include <chrono> 
#include <random>
#include <ctime>
#include <dmlc/memory_io.h>
int main() {
    alignas(64) const int N = 1000; // 원하는 배열 크기
    alignas(64) int A_data[N][N];
    // int* A_data;
    // size_t alignment = 64;  // 64바이트 정렬
    // int ret = posix_memalign((void**)&A_data, alignment, N * N * sizeof(int));
    // if (ret != 0) {
    //     std::cerr << "Memory alignment failed" << std::endl;
    //     return -1;
    // }
    int x_data[N];
    alignas(64) int y_data[N];
    int y_result[N];

    // 난수 생성기를 위한 준비
    std::random_device rd;  // 하드웨어 난수
    std::mt19937 gen(rd()); // Mersenne Twister 엔진
    std::uniform_int_distribution<int> distA(0, 100); // A_data용 난수 범위
    std::uniform_int_distribution<int> distX(0, N-1); // x_data용 난수 범위
    
    // A_data, x_data, y_data, y_result 초기화
    for (int i = 0; i < N; ++i) {
        std::cout << "i: " << i << std::endl;
        // A_data를 난수로 채움
        for (int j = 0; j < N; ++j) {
            // A_data[i * N + j] = distA(gen);
            A_data[i][j] = distA(gen);
        }

        // x_data는 0 ~ N-1 범위의 난수
        x_data[i] = distX(gen);

        // y_data와 y_result는 초기값 1로 설정
        y_data[i] = 1;
        y_result[i] = 1;
    }

    // OpenMP 디렉티브와 연산 수행
    // TVM 모듈 호출 코드
std::cout << "A_data: " << std::endl;
auto start = std::chrono::high_resolution_clock::now();
tvm::runtime::Module mod = tvm::runtime::Module::LoadFromFile("/root/TVM_HPC/build/../test_input.cpp_tvm_prim_func.so");
if (!mod.defined()) {
    std::cerr << "Error: Could not load module " << "/root/TVM_HPC/build/../test_input.cpp_tvm_prim_func.so" << std::endl;
    return -1;
}

tvm::runtime::PackedFunc f = mod.GetFunction("main");
DLDevice dev{kDLCPU, 0};

DLTensor dltensor;
dltensor.data = A_data;
dltensor.device = {kDLCPU, 0};
dltensor.ndim = 2;
dltensor.dtype = DLDataType{kDLInt, 32, 1};
int64_t shape[] = {1000, 1000};
dltensor.shape = shape;
dltensor.strides = nullptr;  // 연속적인 메모리일 경우
dltensor.byte_offset = 0;

tvm::runtime::NDArray A_data_nd = tvm::runtime::NDArray::FromExternalDLTensor(dltensor);
// tvm::runtime::NDArray A_data_nd = tvm::runtime::NDArray::Empty({1000, 1000}, {kDLInt,32,1}, dev);
// std::memcpy(A_data_nd->data, A_data, sizeof(A_data));

// tvm::runtime::NDArray N_nd = tvm::runtime::NDArray::Empty({1}, {kDLInt,32,1}, dev);
// std::memcpy(N_nd->data, &N, sizeof(N));

DLTensor N_dltensor;
N_dltensor.data = reinterpret_cast<void*>(const_cast<int32_t*>(&N)); // N의 주소를 void*로 변환
N_dltensor.device = {kDLCPU, 0};
N_dltensor.ndim = 1;
N_dltensor.dtype = DLDataType{kDLInt, 32, 1};
int64_t N_shape[] = {1};
N_dltensor.shape = N_shape;
N_dltensor.strides = nullptr;
N_dltensor.byte_offset = 0;
tvm::runtime::NDArray N_nd = tvm::runtime::NDArray::FromExternalDLTensor(N_dltensor);

tvm::runtime::NDArray x_data_nd = tvm::runtime::NDArray::Empty({1000}, {kDLInt,32,1}, dev);
std::memcpy(x_data_nd->data, x_data, sizeof(x_data));

// tvm::runtime::NDArray y_data_nd = tvm::runtime::NDArray::Empty({1000}, {kDLInt,32,1}, dev);
// std::memcpy(y_data_nd->data, y_data, sizeof(y_data));
DLTensor y_data_dltensor;
y_data_dltensor.data = y_data;
y_data_dltensor.device = {kDLCPU, 0};
y_data_dltensor.ndim = 1;
y_data_dltensor.dtype = DLDataType{kDLInt, 32, 1};
int64_t y_data_shape[] = {1000};
y_data_dltensor.shape = y_data_shape;
y_data_dltensor.strides = nullptr;
y_data_dltensor.byte_offset = 0;
tvm::runtime::NDArray y_data_nd = tvm::runtime::NDArray::FromExternalDLTensor(y_data_dltensor);


// Function call
f(A_data_nd, N_nd, x_data_nd, y_data_nd);
// std::memcpy(A_data, A_data_nd->data, sizeof(A_data));
// std::memcpy(x_data, x_data_nd->data, sizeof(x_data));
// std::memcpy(y_data, y_data_nd->data, sizeof(y_data));

auto end = std::chrono::high_resolution_clock::now();

std::chrono::duration<double> elapsed = end - start;
    
// 결과 출력
std::cout << "execution time(tvm): " << elapsed.count() << "sec" << std::endl;

    auto start_cpu = std::chrono::high_resolution_clock::now();
    // 기대 결과 계산
    for (int i = 0; i < N; i++) {
        int sum = 0;
        for (int j = 0; j < N; j++) {
            // sum +=  A_data[i * N + j] * x_data[j];  
            sum += A_data[i][j] * x_data[j];
        }
        y_result[i] = sum;
    }

    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_cpu = end_cpu - start_cpu;
    std::cout << "execution time(cpu): " << elapsed_cpu.count() << "sec" << std::endl;

    // 결과 출력 및 검증
    std::cout << "y_data 계산 결과:\n";
    for (int i = 0; i < N; i++) {
        std::cout << y_data[i] << " ";
    }
    std::cout << "\n\n기대값 (Expected):\n";
    for (int i = 0; i < N; i++) {
        std::cout << y_result[i] << " ";
    }
    std::cout << "\n";

    // 결과 비교
    bool success = true;
    for (int i = 0; i < N; i++) {
        if (y_data[i] != y_result[i]) {
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "\n연산 결과: 정확합니다.\n";
    } else {
        std::cout << "\n연산 결과: 오류가 있습니다.\n";
    }

    return 0;
}