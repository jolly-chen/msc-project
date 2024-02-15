import ROOT
import subprocess
from pathlib import Path

import os
import sys
import shlex
from io import StringIO
import time

base_header = "iter,env,gpu,cc,nbins,bulksize,input,edges,blocksize"
edges_list = ["", "-e"]
blocksize=256
cc = "86"
# reduction = 2
# types="USM"

def run_nsys_benchmark(
    f, n, gpus, environs, bulksizes, nbins, input_files, output_file
):
    os.makedirs(f"{output_file}", exist_ok=True)

    for gi, gpu in enumerate(gpus):
        for iter in range(n):
            for ei, e in enumerate(environs):
                for bi, b in enumerate(bulksizes):
                    for nbi, nb in enumerate(nbins):
                        for edges in edges_list:
                            for nif, ipf in enumerate(input_files):
                                input_file = f"{input_folder}/{ipf}"
                                stem = Path(ipf).stem
                                arg = f"-b{b} -h{nb} -f{input_file} {edges}"
                                cmd = f""
                                nsys = "/usr/local/cuda-12.3/nsight-systems-2023.3.3/target-linux-x64/nsys"
                                print(f"{cmd} {nsys} profile -otemp {f} {arg}")

                                # Profile code with nsys
                                p = subprocess.run(
                                    [*[s.strip("'") for s in shlex.split(f"{cmd} {nsys} profile -otemp-{nif} --force-overwrite=true {f} {arg}",
                                                 posix=False)]],
                                    env={**os.environ, e: "1"},
                                    stdout=subprocess.PIPE,
                                    check=True,
                                    # shell=True,
                                )

                                # Gather statistics
                                result = subprocess.run(
                                    f"{cmd} {nsys} stats temp-{nif}.nsys-rep --force-export=true --format=csv",
                                    stdout=subprocess.PIPE,
                                    check=True,
                                    shell=True,
                                )
                                subprocess.run(["rm", "temp.*"])
                                output = result.stdout.decode("utf-8").split("\n\n")

                                if not Path(f"{output_file}/api").exists():
                                    with open(f"{output_file}/api", "w") as file_handler:
                                        header = output[2].split("\n")[1]
                                        file_handler.write(f"{base_header},{header}\n")

                                if not Path(f"{output_file}/kernel").exists():
                                    with open(f"{output_file}/kernel", "w") as file_handler:
                                        header = output[3].split("\n")[1]
                                        file_handler.write(f"{base_header},{header}\n")

                                if not Path(f"{output_file}/memop").exists():
                                    with open(f"{output_file}/memop","w") as file_handler:
                                        header = output[4].split("\n")[1]
                                        file_handler.write(f"{base_header},{header}\n")

                                out_base = f"{iter},{e},{gpu},{cc},{nb},{b},{stem},{'True' if edges != '' else 'False'},{blocksize}"
                                with open(
                                    f"{output_file}/api",
                                    "a",
                                ) as file_handler:
                                    file_handler.write(
                                        "\n".join([f"{out_base},{s}" for s in output[2].split("\n")[2:]])
                                    )
                                    file_handler.write("\n")

                                with open(
                                    f"{output_file}/kernel",
                                    "a",
                                ) as file_handler:
                                    file_handler.write(
                                        "\n".join([f"{out_base},{s}" for s in output[3].split("\n")[2:]])
                                    )
                                    file_handler.write("\n")

                                with open(
                                    f"{output_file}/memop",
                                    "a",
                                ) as file_handler:
                                    file_handler.write(
                                        "\n".join([f"{out_base},{s}" for s in output[4].split("\n")[2:]])
                                    )
                                    file_handler.write("\n")

def run_benchmark(f, n, gpus, environs, bulksizes, nbins, input_files, output_file=""):
    if not Path(output_file).exists():
        with open(output_file, "w") as file_handler:
            file_handler.write(
                f"{base_header},tfindbin,tfill,tstats,ttotal\n"
            )

    with open(output_file, "a") as file_handler, open(
        f"{output_file}-diff", "w"
    ) as diff_result:
        for gpu in gpus:
            for iter in range(n):
                for ei, e in enumerate(environs):
                    for bi, b in enumerate(bulksizes):
                        for nbi, nb in enumerate(nbins):
                            for edges in edges_list:
                                for nif, ipf in enumerate(input_files):
                                    input_file = f"{input_folder}/{ipf}"
                                    stem = Path(ipf).stem
                                    cmd = f""
                                    arg = f"-b{b} -h{nb} -f{input_file} {edges} -w"
                                    print(f"{cmd} {f} {arg}")

                                    r = subprocess.run(
                                        [*[s.strip("'") for s in shlex.split(f"{cmd} {f} {arg}", posix=False)]],
                                        env={**os.environ, e: "1"},
                                        stdout=subprocess.PIPE,
                                        check=True,
                                        # shell=True,
                                    )

                                    output = r.stdout.decode("utf-8").strip().split("\n")
                                    times = [o.split(":")[1] for o in output]
                                    out_base = f"{iter},{e},{gpu},{cc},{nb},{b},{stem},{'True' if edges != '' else 'False'},{blocksize}"
                                    file_handler.write(
                                        f"{out_base},{','.join(times)}\n"
                                    )
                                    file_handler.flush()

                                    r_file = f"{stem}_h{nb}_e{'1' if edges != '' else '0'}"
                                    print(f"zstd --rm -f -o {r_file}.out"),
                                    subprocess.run(
                                        f"zstd --rm -f -o {r_file}  {r_file}.out",
                                        check=True,
                                        shell=True,
                                    )

                                    diff_result.write(f"{out_base}\n")
                                    subprocess.run(
                                        [
                                            "diff",
                                            f"{r_file}",
                                            f"{input_folder}/expected/{r_file}",
                                        ],
                                        check=False,
                                        stdout=diff_result,
                                        stderr=diff_result,
                                    )
                                    diff_result.flush()

                                    # subprocess.run(
                                    #     [
                                    #         "rm",
                                    #         "-rf",
                                    #         f"{r_file}",
                                    #     ],
                                    #     check=False
                                    # )



if __name__ == "__main__":
    environs = [
        "CUDA_HIST",
        # "SYCL_HIST",
    ]

    n = 5
    nbins = [
        #       1,
        #       2,
        #       5,
        # 10,
        #       20,
        #       50,
        #       100,
        #       500,
        1000,
        #       5000,
        #       10000,
        #       50000,
        # 100000,   # 100K
        # 10000000, # 10M
    ]

    bulksizes = [
            #    1,
        # 2,
        # 4,
            #    8,
        # 16,
        # 32,
            #    64,
        # 128,
        # 256,
            #    512,
        # 1024,
        # 2048,
            #    4096,
        # 8192,
        # 16384,
               32768,
        # 65536,
        # 131072,
            #    262144,
            #    2097152,
            #    16777216,
            #    134217728,
    ]

    input_files = [
        "doubles_uniform_50000000.root",  # 50M
        "doubles_uniform_100000000.root",  # 100M
        "doubles_uniform_500000000.root",  # 500M
        "doubles_uniform_1000000000.root",  # 1B

        "doubles_constant-0.5_50000000.root",  # 50M
        "doubles_constant-0.5_100000000.root",  # 100M
        "doubles_constant-0.5_500000000.root",  # 500M
        "doubles_constant-0.5_1000000000.root",  # 1B

#        "doubles_normal-0.4-0.1_50000000.root",  # 50M
#        "doubles_normal-0.4-0.1_100000000.root",  # 100M
#        "doubles_normal-0.4-0.1_500000000.root",  # 500M
#        "doubles_normal-0.4-0.1_1000000000.root",  # 1B

        "doubles_normal-0.7-0.01_50000000.root",  # 50M
        "doubles_normal-0.7-0.01_100000000.root",  # 100M
        "doubles_normal-0.7-0.01_500000000.root",  # 500M
        "doubles_normal-0.7-0.01_1000000000.root",  # 1B
    ]

    gpus = [
        "A4000",
#        "A2",
#        "A6000",
#        "A100 -p fatq",
    ]

    input_folder = "/data/jolly/input"
    f = "/data/jolly/msc-project-sm86/benchmarks/histond_benchmark"
    if len(sys.argv) > 1:
        output_file = sys.argv[1]
    else:
        output_file = f"{time.strftime('%Y%m%d-%H%M%S')}"

    output_folder = "/data/jolly/msc-project-sm86/gpu-synchronous/l4-gpu"
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

