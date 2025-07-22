import tvm
from tvm.script import tir as T
import numpy as np
import ctypes
import subprocess
import textwrap
import argparse
import tempfile
import logging
import tvm.testing
from tvm import tir, IRModule
from tvm.tir.stmt_functor import post_order_visit
from tvm.ir.transform import module_pass
from tvm.tir import transform
from tvm.ir import transform
from tvm.script import from_source
from tvm import meta_schedule as ms
from tvm.target import Target
from collections import defaultdict

import os
import re
import time

parser = argparse.ArgumentParser()
parser.add_argument(
    "--library_path", type=str, required=True, help="Path to the shared library"
)
parser.add_argument(
    "--tir_script_path", type=str, required=True, help="TIR script Path to be compiled"
)
parser.add_argument(
    "--output_so_path", type=str, required=True, help="Output shared library path"
)
parser.add_argument(
    "--params_path", type=str, required=False, help="Path to the input parameters"
)
args = parser.parse_args()


logging.basicConfig()
logging.getLogger("tvm.meta_schedule").setLevel(logging.DEBUG)

# print path
print(args.library_path)
print(args.tir_script_path)
print(args.output_so_path)


def parse_input_file(file_path):
    function_map = defaultdict(list)

    # 정규 표현식 패턴 (Function Name, Variable Name, Shape, Data Type 추출)
    pattern = re.compile(
        r"Fuction Name:\s*(\w+),\s*"
        r"Variable Name:\s*(\w+),\s*"
        r"Shape:\s*\{([\d,\s]+)\},\s*"
        r"Data Type:\s*(\w+)"
    )

    with open(file_path, "r", encoding="utf-8") as file:  # utf-8 인코딩 적용
        for line in file:
            match = pattern.search(line)
            if match:
                function_name = match.group(1)
                variable_name = match.group(2)
                shape = tuple(
                    map(int, match.group(3).replace(" ", "").split(","))
                )  # 쉼표로 나누어 튜플 변환
                data_type = match.group(4)

                print(
                    f"Function: {function_name}, Variable: {variable_name}, Shape: {shape}, Data Type: {data_type}"
                )
                function_map[function_name].append((variable_name, shape, data_type))

    return function_map


def create_task_inputs(function_map, function_name):
    task_inputs = []

    if function_name not in function_map:
        print(f"Warning: Function '{function_name}' not found in function_map.")
        return task_inputs

    for variable_name, shape, data_type in function_map[function_name]:

        if data_type == "kDLFloat32":
            np_dtype = "float32"
            array = np.random.uniform(low=0.0, high=1.0, size=shape).astype(
                np_dtype
            )  # 랜덤 float 생성
        elif data_type == "kDLInt32":
            np_dtype = "int32"
            array = np.random.randint(
                low=0, high=10, size=shape, dtype=np_dtype
            )  # 랜덤 int 생성
        elif data_type == "kDLInt64":
            np_dtype = "int64"
            array = np.random.randint(low=0, high=10, size=shape, dtype=np_dtype)
        elif data_type == "kDLFloat64":
            np_dtype = "float64"
            array = np.random.uniform(low=0.0, high=1.0, size=shape).astype(np_dtype)
        elif data_type == "kDLInt8":
            np_dtype = "int8"
            array = np.random.randint(
                low=0, high=10, size=shape, dtype=np_dtype
            )  # 랜덤 int8 생성
        elif data_type == "kDLUInt8":
            np_dtype = "uint8"
            array = np.random.randint(low=0, high=10, size=shape, dtype=np_dtype)
        elif data_type == "kDLUInt1":
            np_dtype = "bool"
            array = np.random.randint(
                low=0, high=2, size=shape, dtype=np_dtype
            )  # 랜덤 bool 생성
        else:
            print(
                f"Warning: Unsupported data type '{data_type}' for variable '{variable_name}'. Defaulting to float32."
            )
            np_dtype = "float32"
            array = np.random.uniform(low=0.0, high=1.0, size=shape).astype(np_dtype)

        # Shape이 (1,)이면 1로 초기화
        if shape == (1,):
            array = np.array([1], dtype=np_dtype)

        # print(f"Variable: {variable_name}, Shape: {shape}, Data Type: {np_dtype}, First Value: {array.flatten()[0]}")  # 디버깅용
        task_inputs.append(tvm.nd.array(array))

    return task_inputs


def get_mangled_names(library_path, function_names):
    result = subprocess.run(["nm", "-D", library_path], capture_output=True, text=True)
    mangled_names = {}

    # Parse `nm` output and map mangled names to their original function names
    for line in result.stdout.split("\n"):
        parts = line.split()
        if len(parts) > 1:  # Ensure valid symbol information
            symbol = parts[-1]
            for name in function_names:
                if name in symbol and name not in mangled_names:
                    mangled_names[name] = symbol
    # Construct the final list of mangled names or original names if not found
    return [mangled_names.get(name, name) for name in function_names]


def extract_call_extern_names(mod):
    """Extract all function names used in T.call_extern calls within a TIR module."""
    extern_names = []

    def visitor(stmt):
        if isinstance(stmt, tir.Call) and stmt.op.name == "tir.call_extern":
            func_name = stmt.args[0]
            if isinstance(func_name, tir.StringImm):
                extern_names.append(func_name.value)

    # Traverse each PrimFunc in the module
    for func_name, func in mod.functions.items():
        if isinstance(func, tir.PrimFunc):
            post_order_visit(func.body, visitor)

    return extern_names


@module_pass(opt_level=1)
class ReplaceCallExternNamesPass:
    def __init__(self, old_names, new_names):
        assert len(old_names) == len(
            new_names
        ), "old_names and new_names must have the same length"
        self.name_map = dict(zip(old_names, new_names))  # Map old names to new names

    def transform_module(self, mod, ctx):
        # Define a visitor function
        def visitor(stmt):
            if isinstance(stmt, tir.Call) and stmt.op.name == "tir.call_extern":
                func_name = stmt.args[0]
                if (
                    isinstance(func_name, tir.StringImm)
                    and func_name.value in self.name_map
                ):
                    # Replace the function name with the mapped new name
                    new_func_name = tir.StringImm(self.name_map[func_name.value])
                    return tir.Call(
                        stmt.dtype, stmt.op, [new_func_name] + stmt.args[1:]
                    )
            return stmt

        # Apply transformation to each PrimFunc in the module
        new_functions = {}
        for func_name, func in mod.functions.items():
            if isinstance(func, tir.PrimFunc):
                print(f"Transforming function: {func_name}")
                new_body = tir.stmt_functor.ir_transform(
                    func.body, None, visitor, ["tir.Call"]
                )
                new_func = tir.PrimFunc(
                    func.params, new_body, func.ret_type, func.buffer_map, func.attrs
                )
                new_functions[func_name] = new_func

        # Return a new IRModule with transformed functions
        return IRModule(new_functions)


@tvm.testing.requires_llvm
def tune_ir_module(mod, target, work_dir, function_dict):
    results = {}
    for func_name, func in mod.functions.items():
        if not isinstance(func, tvm.tir.PrimFunc):
            print(f"Skipping non-PrimFunc: {func_name}")
            continue

        print(f"Tuning function: {func_name}")
        single_func_mod = tvm.IRModule({"main": func})

        task_inputs = create_task_inputs(function_dict, func_name.name_hint)
        # space = ms.space_generator.PostOrderApply(
        #     # filter: which blocks to schedule (e.g. all of them)
        #     lambda block: True,
        #     # your schedule rules:
        #     sch_rules=[
        #         ms.schedule_rule.ParallelizeVectorizeUnroll(
        #             max_vectorize_extent=64,
        #         ),
        #     ],
        #     postprocs=[
        #         ms.postproc.RewriteParallelVectorizeUnroll(),
        #     ],
        # )
        # TIR 튜닝
        database = ms.tir_integration.tune_tir(
            mod=single_func_mod,
            target=target,
            work_dir="./",
            max_trials_global=256,
            num_trials_per_iter=64,
            task_inputs=task_inputs,
        )

        # 튜닝된 스케줄 컴파일
        print(f"Compiling the tuned schedule for {func_name}...")
        sch = ms.tir_integration.compile_tir(database, single_func_mod, target)
        if sch is None:
            print(f"No valid schedule found for function {func_name}!")
        else:
            results[func_name] = sch
            print(f"Tuned schedule for {func_name}:")
            sch.mod.show()
            sch.trace.show()

    return results


TIR_SCRIPT_PATH = args.tir_script_path
with open(TIR_SCRIPT_PATH, "r") as f:
    prim_func_text = f.read()

tir_script = f"""
@tvm.script.ir_module
class MyModule:
{textwrap.indent(prim_func_text, "    ")}
"""
print("Loaded TIR script:")


# TIR 출력
ir_module = from_source(tir_script)

print(ir_module)

passes = transform.Sequential(
    [
        tvm.tir.transform.ExpandReductionBuffers(),
        tvm.tir.transform.AddSpatialAxisToLoopsPass(),  # 어떤거는 이게 앞에 필요하고
        tvm.tir.transform.SimplifyLoopBounds(),
        tvm.tir.transform.PerfectlyNestedLoops(),
        tvm.tir.transform.AddSpatialAxisToLoopsPass(),  # 어떤거는 뒤에 필요함
        #
        # tvm.tir.transform.ResolveWAR(),
        tvm.tir.transform.Simplify(),
    ]
)
ir_module = passes(ir_module)

print("Transformed IRModule:")

# @tvm.script.ir_module
# class MyModule1:
#     # from tvm.script import tir as T
#     @T.prim_func
#     def main0(ny: T.Buffer((1,), "int32"), y: T.Buffer((4000,), "float32")):
#         T.func_attr({"tir.noalias": T.bool(True)})
#         # with T.block("root"):
#         for i in range(ny[0]):
#             with T.block("block1"):
#                 y[i] = T.float32(0)
#     # from tvm.script import tir as T
#     @T.prim_func
#     def main1(A: T.Buffer((4000, 4000), "float32"), tmp: T.Buffer((4000,), "float32"), x: T.Buffer((4000,), "float32"), y: T.Buffer((4000,), "float32")):
#         T.func_attr({"tir.noalias": T.bool(True)})
#         # with T.block("root"):
#         for i, j in T.grid(4000, 4000):
#             with T.block("reduction_block: tmp"):
#                 vi, vj = T.axis.remap("SR", [i, j])
#                 with T.init():
#                     tmp[vi] = T.float32(0)
#                 tmp[vi] += A[vi, vj] * x[vj]

#         for i, j in T.grid(4000, 4000):
#             with T.block("reduction_block: y"):
#                 vj, vi = T.axis.remap("SR", [j, i])
#                 with T.init():
#                     T.evaluate(0)
#                 y[vj] += A[vi, vj] * tmp[vi]

# ir_module = MyModule1

# library_path 가 있는지 확인
library_path = args.library_path


extern_names = extract_call_extern_names(ir_module)
mangled_names = get_mangled_names(library_path, extern_names)


print(ir_module.script())
trans_func_name = ReplaceCallExternNamesPass(extern_names, mangled_names)
transformed_ir_module = trans_func_name(ir_module)
print(transformed_ir_module.script())

# build the optimized TIR
# with tvm.transform.PassContext(opt_level=4):  # TVM 최적화 레벨 4
#     mod = tvm.build(transformed_ir_module, target="llvm -opt-level=3 -mcpu=native", name="main")
#     output_so_path = args.output_so_path
#     mod.export_library(output_so_path)
#     llvm_ir = mod.get_source("ll")
#     #store llvm ir
#     with open("output.ll", "w") as f:
#         f.write(llvm_ir)

#     print(f"Exported shared library to {output_so_path}")
# exit()

# os.environ["TVM_NUM_THREADS"] = "64"  # TVM이 64개의 스레드를 사용하도록 강제
# os.environ["OMP_NUM_THREADS"] = "64"

function_dict = parse_input_file(args.params_path)
march = os.environ.get("MARCH", "native")


with tempfile.TemporaryDirectory() as work_dir:
    # Step 1: Auto-tune the IRModule
    print(f"Library path: {library_path}")
    # lib = ctypes.CDLL(library_path, mode=ctypes.RTLD_GLOBAL)
    # ctypes.CDLL(library_path, mode=ctypes.RTLD_GLOBAL)
    # os.environ["LD_PRELOAD"] = library_path

    # -mcpu=sapphirerapids

    # ctypes.CDLL("/root/TVM_HPC/struct_function.so", mode=ctypes.RTLD_GLOBAL)
    # os.environ["LD_PRELOAD"] = "/root/TVM_HPC/struct_function.so"

    target = Target(f"llvm --num-cores=64 -mcpu={march} -opt-level=3")
    # target = Target("llvm --num-cores=64 -opt-level=3")
    start_time = time.time()
    print(f"Tuning start time: {start_time}")
    tuned_schedules = tune_ir_module(
        transformed_ir_module, target, work_dir, function_dict
    )
    end_time = time.time()
    print(f"Tuning completed in {end_time - start_time:.2f} seconds")
    final_ir_module = tvm.IRModule()

    for func_name, func in ir_module.functions.items():
        if func_name in tuned_schedules:
            sch = tuned_schedules[func_name]
            if sch is not None:
                # 튜닝된 함수 이름 확인
                tuned_func_name = next(iter(sch.mod.get_global_vars())).name_hint
                print(f"Tuned function name: {tuned_func_name}")

                # 튜닝된 함수 추가
                print(f"Adding tuned function: {func_name}")
                final_ir_module[func_name] = sch.mod[
                    next(iter(sch.mod.get_global_vars()))
                ]
            else:
                # 튜닝 실패한 경우 원래 함수 유지
                print(f"Tuning failed for {func_name}. Adding original function.")
                final_ir_module[func_name] = func
        else:
            # 튜닝 결과가 없는 경우 원래 함수 유지
            print(f"No schedule found for {func_name}. Adding original function.")
            final_ir_module[func_name] = func
    # target = tvm.target.Target("llvm -opt-level=3")
    # print final_ir_module
    print("Final IRModule after merging tuned and original functions:")
    print(final_ir_module.script())
    with tvm.transform.PassContext(opt_level=3):
        mod = tvm.build(
            final_ir_module,
            target=f"llvm --num-cores=64 -opt-level=3 -mcpu={march}",
            name="main",
        )

        # **Corrected line: Load the shared library with RTLD_GLOBAL**

        output_so_path = args.output_so_path
        mod.export_library(output_so_path)
        llvm_ir = mod.get_source("ll")
        # store llvm ir
        with open("output.ll", "w") as f:
            f.write(llvm_ir)

        print(f"Exported shared library to {output_so_path}")


# with tempfile.TemporaryDirectory() as work_dir:
#     print(f"Library path: {library_path}")
#     target = Target(f"llvm --num-cores=64 -mcpu={march} -opt-level=3")

#     # 튜닝 없이 원래 모듈 그대로 사용
#     final_ir_module = ir_module  # 또는 ir_module

#     print("Final IRModule (no tuning):")
#     print(final_ir_module.script())

#     with tvm.transform.PassContext(opt_level=3):
#         mod = tvm.build(final_ir_module, target=target, name="main")

#         output_so_path = args.output_so_path
#         mod.export_library(output_so_path)

#         llvm_ir = mod.get_source("ll")
#         with open("output.ll", "w") as f:
#             f.write(llvm_ir)

#         print(f"Exported shared library to {output_so_path}")
