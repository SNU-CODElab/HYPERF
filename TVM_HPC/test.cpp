#include <iostream>
#include <omp.h>
#include <random>
#include <ctime>
#include <chrono> 
 #include <iomanip>
int main() {
    const int N = 1024; // 원하는 배열 크기
    float A_data[N][N];
    float x_data[N];
    float y_data[N];
    float y_result[N];

    // 난수 생성기를 위한 준비
    std::random_device rd;                  // 하드웨어 난수
    std::mt19937 gen(rd());                 // Mersenne Twister 엔진
    std::uniform_real_distribution<float> distA(0.0f, 10.0f); // A_data용 float 범위
    std::uniform_real_distribution<float> distX(0.0f, static_cast<float>(N-1)); // x_data용 float 범위

    // A_data, x_data, y_data, y_result 초기화
    for (int i = 0; i < N; ++i) {
        // A_data를 난수로 채움
        for (int j = 0; j < N; ++j) {
            A_data[i][j] = distA(gen);
        }

        // x_data는 0.0 ~ N-1 범위의 난수
        x_data[i] = distX(gen);

        // y_data와 y_result는 초기값 1.0으로 설정
        y_data[i] = 1.0f;
        y_result[i] = 1.0f;
    }
    // OpenMP 디렉티브와 연산 수행
    // auto start = std::chrono::high_resolution_clock::now();
    #pragma omp tvm tvm_arr_size(A_data[0:1024][0:1024],x_data[0:1024],y_data[0:1024])
    for (int i = 0; i < 1024; i++) {
        for (int j = 0; j < 1024; j++) {
            //A*x gemv 연산
            y_data[i] += A_data[i][j] * x_data[j];
        }
    }
    // auto end = std::chrono::high_resolution_clock::now();

    // std::chrono::duration<double> elapsed = end - start;
    

    // std::cout << std::fixed << std::setprecision(12);
    // std::cout << "execution time: " << elapsed.count() << "sec" << std::endl;

    std::cout << std::fixed << std::setprecision(12);

    // Warm-up phase
    std::cout << "Warming up...\n";
    for (int w = 0; w < 100; w++) { // 웜업 10회 반복
        for (int i = 0; i < N; i++) {
            float sum = 0;
            for (int j = 0; j < N; j++) {
                sum += A_data[i][j] * x_data[j];
            }
            y_result[i] = sum;
        }
    }

    // Timing measurement phase
    int measure_iters_cpu = 1000; // 측정 반복 횟수
    std::chrono::duration<double> total_elapsed_cpu(0); // 누적 시간 초기화

    auto start_cpu = std::chrono::high_resolution_clock::now();
    for (int m = 0; m < measure_iters_cpu; m++) {
        for (int i = 0; i < N; i++) {
            float sum = 0;
            for (int j = 0; j < N; j++) {
                sum += A_data[i][j] * x_data[j];
            }
            y_result[i] = sum;
        }
    }
    auto end_cpu = std::chrono::high_resolution_clock::now();

    // Calculate and display average time
    std::chrono::duration<double> average_time_cpu = (end_cpu-start_cpu) / measure_iters_cpu;
    std::cout << "Average execution time (clang): " << average_time_cpu.count() << " sec" << std::endl;
    // 결과 출력 및 검증
    // std::cout << "y_data 계산 결과:\n";
    // for (int i = 0; i < N; i++) {
    //     std::cout << y_data[i] << " ";
    // }
    // std::cout << "\n\n기대값 (Expected):\n";
    // for (int i = 0; i < N; i++) {
    //     std::cout << y_result[i] << " ";
    // }
    // std::cout << "\n";

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