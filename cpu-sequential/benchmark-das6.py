import ROOT
import subprocess
import glob
from pathlib import Path

import os
import sys
import re
import time

def run_benchmark(f, n, environs, bulksizes, nbins, nvals, output_file=""):
    if not Path(output_file).exists():
        with open(output_file, "w") as file_handler:
            # file_handler.write("iter,env,nbins,bulksize,input,edges,tfindbin,tfill,tstats,ttotal\n")
            file_handler.write("iter,env,nbins,bulksize,input,edges,ttotal\n")

    with open(output_file, "a") as file_handler:
        for iter in range(n):
            for ei, e in enumerate(environs):
                for nbi, nb in enumerate(nbins):
                    for bi, b in enumerate(bulksizes):
                        for nvi, nv in enumerate(nvals):
                            for edges in ["", "-e"]:
                                input_file = f"{input_folder}/doubles_uniform_{nv}.root"
                                arg = f"-b{b} -h{nb} -f{input_file} {edges}"
                                print(arg)
                                cmd = f"prun -v -np 1"

                                r = subprocess.run(f"{cmd} {f} {arg}",
                                    env={**os.environ, e: "1"},
                                    check=True,
                                    stdout=subprocess.PIPE,
                                    shell=True
                                )
                                output = r.stdout.decode("utf-8").strip().split("\n")
                                times =  [o.split(":")[1] for o in output]
                                edges_bool = edges == ""
                                file_handler.write(f"{iter},{e},{nb},{b},{input_file},{edges_bool},{','.join(times)}\n")
                                file_handler.flush()


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
        #65536,
        #131072,
#        262144,
    ]

    nvals = [
        # 1,
        # 10,
        # 100,
        # 1000,         # 1K
        # 10000,        # 10K
        # 100000,       # 100K
        # 1000000,      # 1M
        # 10000000,     # 10M
        50000000,     # 50M
     	100000000,    # 100M
        500000000,    # 500M
        1000000000,   # 1B
        #5000000000,   # 5B
        # 10000000000,  # 10B
    ]

    input_folder="/var/scratch/jchen/input"
    f = "../benchmarks/histond_benchmark_timing"
    if len(sys.argv) > 1:
        output_file = sys.argv[1]
    else:
        output_file = f"{time.strftime('%Y%m%d-%H%M%S')}"

    os.makedirs(f"das6-cpu", exist_ok=True)
    print(f"writing results to das6-cpu/{output_file}...")

    run_benchmark(
        f,
        n,
        environs,
        bulksizes,
        nbins,
        nvals,
        f"das6-cpu/{output_file}"
    )
