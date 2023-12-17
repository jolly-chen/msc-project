import ROOT
import subprocess
from pathlib import Path

import os
import sys
from io import StringIO
import time


def run_nsys_benchmark(
    f, n, gpus, environs, bulksizes, nbins, input_files, output_file
):
    os.makedirs(f"{output_file}", exist_ok=True)

    for iter in range(n):
        for gi, gpu in enumerate(gpus):
            for ei, e in enumerate(environs):
                for bi, b in enumerate(bulksizes):
                    for nbi, nb in enumerate(nbins):
                        for nif, ipf in enumerate(input_files):
                            for edges in ["", "-e"]:
                                input_file = f"{input_folder}/{ipf}"
                                stem = Path(ipf).stem
                                arg = f"-b{b} -h{nb} -f{input_file} {edges}"
                                cmd = f"prun -v -np 1 -native '-C gpunode,{gpu} --gres=gpu:1'"
                                nsys = "/cm/shared/apps/cuda12.3/toolkit/12.3/nsight-systems-2023.4.1/bin/nsys"
                                print(f"{cmd} {nsys} profile -otemp {f} {arg}")

                                # Profile code with nsys
                                profile = subprocess.run(
                                    f"{cmd} {nsys} profile -otemp --force-overwrite=true {f} {arg}",
                                    env={**os.environ, e: "1"},
                                    stdout=subprocess.PIPE,
                                    check=True,
                                    shell=True,
                                )

                                # Gather statistics
                                result = subprocess.run(
                                    f"{cmd} {nsys} stats temp.nsys-rep --force-export=true --format=csv",
                                    stdout=subprocess.PIPE,
                                    check=True,
                                    shell=True,
                                )
                                subprocess.run(["rm", "temp.*"])
                                output = result.stdout.decode("utf-8").split("\n\n")

                                if not Path(f"{output_file}/api").exists():
                                    with open(f"{output_file}/api", "w") as file_handler:
                                        header = output[2].split("\n")[1]
                                        file_handler.write(f"iter,env,gpu,nbins,bulksize,input,edges,{header}\n")

                                if not Path(f"{output_file}/kernel").exists():
                                    with open(f"{output_file}/kernel", "w") as file_handler:
                                        header = output[3].split("\n")[1]
                                        file_handler.write(f"iter,env,gpu,nbins,bulksize,input,edges,{header}\n")

                                if not Path(f"{output_file}/memop").exists():
                                    with open(f"{output_file}/memop","w") as file_handler:
                                        header = output[4].split("\n")[1]
                                        file_handler.write(f"iter,env,gpu,nbins,bulksize,input,edges,{header}\n")

                                with open(
                                    f"{output_file}/api",
                                    "a",
                                ) as file_handler:
                                    file_handler.write(
                                        "\n".join([f"{iter},{e},{gpu},{nb},{b},{input_file},{'True' if edges != '' else 'False'},{s}" for s in output[2].split("\n")[2:]])
                                    )
                                    file_handler.write("\n")

                                with open(
                                    f"{output_file}/kernel",
                                    "a",
                                ) as file_handler:
                                    file_handler.write(
                                        "\n".join([f"{iter},{e},{gpu},{nb},{b},{input_file},{'True' if edges != '' else 'False'},{s}" for s in output[3].split("\n")[2:]])
                                    )
                                    file_handler.write("\n")

                                with open(
                                    f"{output_file}/memop",
                                    "a",
                                ) as file_handler:
                                    file_handler.write(
                                        "\n".join([f"{iter},{e},{gpu},{nb},{b},{input_file},{'True' if edges != '' else 'False'},{s}" for s in output[4].split("\n")[2:]])
                                    )
                                    file_handler.write("\n")

def run_benchmark(f, n, gpus, environs, bulksizes, nbins, input_files, output_file=""):
    if not Path(output_file).exists():
        with open(output_file, "w") as file_handler:
            file_handler.write(
                "iter,env,gpu,nbins,bulksize,input,edges,tfindbin,tfill,tstats,ttotal\n"
            )

    with open(output_file, "a") as file_handler, open(
        f"{output_file}-diff", "w"
    ) as diff_result:
        for iter in range(n):
            for gpu in gpus:
                for ei, e in enumerate(environs):
                    for bi, b in enumerate(bulksizes):
                        for nbi, nb in enumerate(nbins):
                            for nif, ipf in enumerate(input_files):
                                for edges in ["", "-e"]:
                                    input_file = f"{input_folder}/{ipf}"
                                    stem = Path(ipf).stem
                                    arg = f"-b{b} -h{nb} -f{input_file} {edges}"
                                    print(arg)
                                    cmd = f"prun -v -np 1 -native '-C gpunode,{gpu} --gres=gpu:1'"

                                    r = subprocess.run(
                                        f"{cmd} {f} {arg}",
                                        env={**os.environ, e: "1"},
                                        check=True,
                                        stdout=subprocess.PIPE,
                                        shell=True,
                                    )

                                    output = (
                                        r.stdout.decode("utf-8").strip().split("\n")
                                    )
                                    times = [o.split(":")[1] for o in output]
                                    file_handler.write(
                                        f"{iter},{e},{gpu},{nb},{b},{input_file},{'True' if edges != '' else 'False'},{','.join(times)}\n"
                                    )
                                    file_handler.flush()

                                    diff_result.write(f"{ipf}\n")
                                    subprocess.run(
                                        [
                                            "diff",
                                            f"{stem}_h{nb}_e{'1' if edges != '' else '0'}.out",
                                            f"{input_folder}/expected/{stem}_h{nb}_e{'1' if edges != '' else '0'}",
                                        ],
                                        check=False,
                                        stdout=diff_result,
                                        stderr=diff_result,
                                    )
                                    diff_result.flush()


if __name__ == "__main__":
    environs = [
#         "CUDA_HIST",
        "SYCL_HIST",
    ]

    n = 5
    nbins = [
        1,
        # 2,
        # 5,
       10,
        # 20,
        # 50,
       100,
        # 500,
        1000,
        # 5000,
        # 10000,
        # 50000,
    ]

    bulksizes = [
        # 1,
        # 2,
        # 4,
        # 8,
        # 16,
        # 32,
        # 64,
        # 128,
        # 256,
        # 512,
        # 1024,
        # 2048,
        # 4096,
        # 8192,
        # 16384,
        32768,
        # 65536,
        # 131072,
        # 262144,
    ]

    input_files = [
#        "doubles_uniform_50000000.root",  # 50M
#        "doubles_uniform_100000000.root",  # 100M
        "doubles_uniform_500000000.root",  # 500M
        "doubles_uniform_1000000000.root",  # 1B
    ]

    gpus = [
        "A4000",
#        "A2",
#        "A6000",
#        "A100 -p fatq",
    ]

    input_folder = "/var/scratch/jchen/input"
    f = "/home/jchen/msc-project/benchmarks/histond_benchmark"
    if len(sys.argv) > 1:
        output_file = sys.argv[1]
    else:
        output_file = f"{time.strftime('%Y%m%d-%H%M%S')}"

    output_folder = "/home/jchen/msc-project/gpu-synchronous/das6-gpu"
    os.makedirs(output_folder, exist_ok=True)
    print(f"writing results to {output_folder}/{output_file}...")

   run_benchmark(
      f,
      n,
      gpus,
      environs,
      bulksizes,
      nbins,
      input_files,
      f"{output_folder}/{output_file}",
  )

    run_nsys_benchmark(
        f,
        n,
        gpus,
        environs,
        bulksizes,
        nbins,
        input_files,
        f"{output_folder}/nsys-{output_file}",
    )
