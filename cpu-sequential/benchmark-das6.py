import ROOT
import subprocess
from pathlib import Path

import os
import sys
import time


def run_benchmark(f, n, environs, bulksizes, nbins, input_files, output_file=""):
    if not Path(output_file).exists():
        with open(output_file, "w") as file_handler:
            file_handler.write(
                "iter,env,nbins,bulksize,input,edges,tfindbin,tfill,tstats,ttotal\n"
            )

    with open(output_file, "a") as file_handler:
        for iter in range(n):
            for ei, e in enumerate(environs):
                for nbi, nb in enumerate(nbins):
                    for bi, b in enumerate(bulksizes):
                        for nif, ipf in enumerate(input_files):
                            for edges in ["", "-e"]:
                                input_file = f"{input_folder}/{ipf}"
                                stem = Path(ipf).stem
                                arg = f"-b{b} -h{nb} -f{input_file} {edges}"
                                print(arg)
                                cmd = f"prun -v -np 1"

                                r = subprocess.run(
                                    f"{cmd} {f} {arg}",
                                    env={**os.environ, e: "1"},
                                    check=True,
                                    stdout=subprocess.PIPE,
                                    shell=True,
                                )

                                output = r.stdout.decode("utf-8").strip().split("\n")
                                times = [o.split(":")[1] for o in output]
                                file_handler.write(
                                    f"{iter},{e},{nb},{b},{stem},{'True' if edges != '' else 'False'},{','.join(times)}\n"
                                )
                                file_handler.flush()

                                if iter == 0 and bi == 0:
                                    subprocess.run(
                                        [
                                            "mv",
                                            f"{stem}_h{nb}_e{'1' if edges != '' else '0'}.out",
                                            f"{input_folder}/expected/{stem}_h{nb}_e{'1' if edges != '' else '0'}",
                                        ],
                                        check=True,
                                    )


if __name__ == "__main__":
    environs = [
        "CPU",
    ]

    n = 5
    nbins = [
        1,
        #       2,
        #       5,
        10,
        #       20,
        #       50,
        100,
        #       500,
        1000,
        #       5000,
        #       10000,
        # 50000,
    ]

    bulksizes = [
        #        1,
        # 2,
        # 4,
        #        8,
        # 16,
        # 32,
        #        64,
        # 128,
        # 256,
        #        512,
        # 1024,
        # 2048,
        #        4096,
        # 8192,
        # 16384,
        32768,
        # 65536,
        # 131072,
        #        262144,
    ]

    input_files = [
        "doubles_uniform_50000000.root",  # 50M
        "doubles_uniform_100000000.root",  # 100M
        "doubles_uniform_500000000.root",  # 500M
        "doubles_uniform_1000000000.root",  # 1B
    ]

    input_folder = "/var/scratch/jchen/input"
    f = "../benchmarks/histond_benchmark_timing"
    if len(sys.argv) > 1:
        output_file = sys.argv[1]
    else:
        output_file = f"{time.strftime('%Y%m%d-%H%M%S')}"

    output_folder = "das6-cpu"
    os.makedirs(output_folder, exist_ok=True)
    print(f"writing results to {output_folder}/{output_file}...")

    run_benchmark(
        f, n, environs, bulksizes, nbins, input_files, f"{output_folder}/{output_file}"
    )