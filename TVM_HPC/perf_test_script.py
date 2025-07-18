# import subprocess
# import re
# import time

# def measure_execution_time(executable, num_runs=1000, label=""):
#     times = []
    
#     for i in range(num_runs):
#         # 실행 파일 호출
#         result = subprocess.run([executable], capture_output=True, text=True)
#         output = result.stdout
        
#         # 실행 시간 추출
#         match = re.search(r"execution time:\s*([\d.]+)sec", output)
#         if match:
#             exec_time = float(match.group(1))
#         else:
#             print(f"Error: Execution time not found in output: {output}")
#             continue
        
#         times.append(exec_time)
#         print(f"Run {i+1}: {label} = {exec_time:.8f} sec")
#         time.sleep(0.01)  # 약간의 딜레이 (optional)
    
#     # 평균 계산
#     avg_time = sum(times) / len(times) if times else 0.0
#     print(f"\n=== {label} Execution Time Summary ===")
#     print(f"Average execution time ({label}): {avg_time:.8f} sec\n")
#     return avg_time

# if __name__ == "__main__":
#     # 실행 파일 경로
#     tvm_executable = "./build/transformed_output"  # TVM 사용
#     clang_executable = "./test_input_clang"       # Clang 사용
    
#     # 각각의 실행 시간 측정
#     print("Measuring TVM Execution Time...")
#     avg_tvm_time = measure_execution_time(tvm_executable, num_runs=1000, label="TVM")

#     print("Measuring Clang Execution Time...")
#     avg_clang_time = measure_execution_time(clang_executable, num_runs=1000, label="Clang")
    
#     # 최종 결과 출력
#     print("\n=== Final Summary ===")
#     print(f"Average execution time (TVM): {avg_tvm_time:.8f} sec")
#     print(f"Average execution time (Clang): {avg_clang_time:.8f} sec")


import subprocess
import re
import time

def measure_execution_time(executable, num_runs=1000):
    tvm_times = []
    cpu_times = []
    
    for i in range(num_runs):
        # 실행 파일 호출
        result = subprocess.run([executable], capture_output=True, text=True)
        output = result.stdout

        # 실행 시간 추출
        tvm_time = float(re.search(r"execution time: (\d+\.\d+)sec", output).group(1))
        cpu_time = float(re.search(r"execution time\(clang\): (\d+\.\d+)sec", output).group(1))
        
        tvm_times.append(tvm_time)
        cpu_times.append(cpu_time)
        
        print(f"Run {i+1}: tvm={tvm_time:.8f} sec, cpu={cpu_time:.8f} sec")
        time.sleep(0.01)  # 약간의 딜레이 (optional)
    
    # 평균 계산
    avg_tvm_time = sum(tvm_times) / num_runs
    avg_cpu_time = sum(cpu_times) / num_runs
    
    print("\n=== Execution Time Summary ===")
    print(f"Average execution time (tvm): {avg_tvm_time:.8f} sec")
    print(f"Average execution time (cpu): {avg_cpu_time:.8f} sec")

if __name__ == "__main__":
    executable_path = "./build/transformed_output"  # 실행 파일 경로
    measure_execution_time(executable_path)