#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "사용법: $0 <file_path> <file_name> <compiler_options>"
    exit 1
fi

FILE_PATH=$1
FILE_NAME=$2
COMPILER_OPTIONS=$3


FILE_NAME_BASE=$(basename "$FILE_NAME" .c)
FILE_NAME_HYPERF="${FILE_NAME_BASE}_hyperf.c"
REPLACED_FILE="${FILE_NAME_HYPERF}_replaced.cpp"

# 1. 변환 실행
echo "Running tvm_hpc on ${FILE_NAME_HYPERF}..."
cd "$FILE_PATH" || exit 1
/root/TVM_HPC/build/tvm_hpc "$FILE_NAME_HYPERF" "$COMPILER_OPTIONS"

# # 2. 원본 코드 빌드
# -march=sapphirerapids
echo "Compiling original ${FILE_NAME} ..."
clang -O3 -march=sapphirerapids -fopenmp -DPOLYBENCH_TIME "$COMPILER_OPTIONS"\
  -I/root/HYPERF/polybench/OpenMP/utilities \
  "$FILE_NAME" \
  /root/HYPERF/polybench/OpenMP/utilities/polybench.c \
  -o openmp_exec -lstdc++

# 3. 변환된 코드 빌드
echo "Compiling transformed ${REPLACED_FILE}..."
clang -O3 -march=sapphirerapids -DPOLYBENCH_TIME "$COMPILER_OPTIONS"\
  -I/root/tvm/include \
  -I/root/tvm/3rdparty/dmlc-core/include \
  -I/root/tvm/3rdparty/dlpack/include \
  -I/root/tvm/3rdparty/rang/include \
  -I/root/tvm/src/target/spirv \
  -L/root/tvm/build \
  -ltvm_runtime -ldl \
  -I/root/HYPERF/polybench/OpenMP/utilities \
  "$REPLACED_FILE" \
  /root/test/gemm/PolyBench-ACC/OpenMP/utilities/polybench.c \
  -o hyperf_exec -lstdc++

echo "Running performance tests..."

LOG_DIR="${FILE_NAME_BASE}_hyperf_log"
mkdir -p "$LOG_DIR"

measure_exec() {
  local exec_name=$1
  local log_file="${LOG_DIR}/${exec_name}_${COMPILER_OPTIONS}.log"
  local times=()

  echo "Running ${exec_name} 11 times..."
  for i in {1..11}; do
    echo "[$exec_name] Run #$i" >> "$log_file"
    output=$(./"$exec_name" 2>&1)
    echo "$output" >> "$log_file"

    # 소수점으로 시작하는 숫자 줄 찾기 (예: 0.005526)
    time=$(echo "$output" | grep -E '^[0-9]+\.[0-9]+' | head -n 1)

    if [[ -n "$time" ]]; then
      times+=("$time")
    fi
    sleep 2
  done

  if [ "${#times[@]}" -eq 0 ]; then
    echo "No valid times collected for $exec_name"
    echo "Median execution time: N/A" >> "$log_file"
    return
  fi

  # 정렬 및 중간값 계산
  IFS=$'\n' sorted=($(sort -n <<<"${times[*]}"))
  unset IFS
  len=${#sorted[@]}

  if (( len % 2 == 1 )); then
    median=${sorted[$((len / 2))]}
  else
    mid=$((len / 2))
    median=$(echo "scale=6; (${sorted[$mid]} + ${sorted[$((mid - 1))]}) / 2" | bc)
  fi

  echo "Median execution time for ${exec_name}: ${median} seconds"
  echo "Median execution time: ${median} seconds" >> "$log_file"
}
sleep 1
measure_exec "openmp_exec"
sleep 1
measure_exec "hyperf_exec"
sleep 1
measure_exec "openmp_exec"
sleep 1
measure_exec "hyperf_exec"
# sleep 1
measure_exec "openmp_exec"
sleep 1
measure_exec "hyperf_exec"

echo "All done! Logs are saved in $LOG_DIR/"


echo "Done!"