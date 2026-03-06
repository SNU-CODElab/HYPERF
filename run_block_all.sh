#!/bin/bash

BUILD_SCRIPT="./build_script.sh"
BENCH_ROOT="/root/HYPERF/polybench/OpenMP"

declare -a FILES=(
  # "blocked_workload/batched_gemm/small/batched_gemm_small.c"
  # "blocked_workload/conv_im2col/small/conv_im2col_small.c"
  # "blocked_workload/geqrf/small/geqrf_small.c"
  # "blocked_workload/syrk/small/syrk_small.c"
  # "blocked_workload/trmm_block/small/trmm_blocked.c"

  # "blocked_workload/batched_gemm/standard/batched_gemm_standard.c"
  # "blocked_workload/conv_im2col/standard/conv_im2col_standard.c"
  # "blocked_workload/geqrf/standard/geqrf_standard.c"
  # "blocked_workload/syrk/standard/syrk_standard.c"
  # "blocked_workload/trmm_block/standard/trmm_blocked.c"

  # "blocked_workload/batched_gemm/large/batched_gemm_large.c"
  "blocked_workload/conv_im2col/large/conv_im2col_large.c"
  # "blocked_workload/geqrf/large/geqrf_large.c"
  # "blocked_workload/syrk/large/syrk_large.c"
  # "blocked_workload/trmm_block/large/trmm_blocked.c"
)

for FILE in "${FILES[@]}"; do
  DIR=$(dirname "$FILE")
  BASE=$(basename "$FILE")

  # 기본은 dataset 없음
  DATASET=""

  # trmm_block일 때만 dataset 설정
  if [[ "$FILE" == *"trmm_block"* ]]; then
    if [[ "$FILE" == *"/small/"* ]]; then
      DATASET="-DSMALL_DATASET"
    elif [[ "$FILE" == *"/standard/"* ]]; then
      DATASET="-DSTANDARD_DATASET"
    elif [[ "$FILE" == *"/large/"* ]]; then
      DATASET="-DLARGE_DATASET"
    fi
  fi

  echo "=============================="
  echo "▶ 실행 중: $FILE ($DATASET)"
  echo "=============================="

  if [[ -z "$DATASET" ]]; then
    "$BUILD_SCRIPT" "${BENCH_ROOT}/${DIR}" "$BASE" ""
  else
    "$BUILD_SCRIPT" "${BENCH_ROOT}/${DIR}" "$BASE" "$DATASET"
  fi

  echo ""

done