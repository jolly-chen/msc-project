import numpy as np
import pandas as pd
import ROOT
import subprocess
import glob
from pathlib import Path

import os
import sys
from io import StringIO
import re
import time
from tqdm.auto import tqdm

def run_benchmark(f, n, environs, bulksizes, nbins, nvals, output_file=""):
    if not Path(output_file).exists():
        with open(output_file, "w") as file_handler:
            file_handler.write("iter,env,nbins,bulksize,input,tfindbin,tfill,tstats\n")

    with open(output_file, "a") as file_handler:
        for iter in tqdm(range(n)):
            for ei, e in tqdm(enumerate(environs), leave=False):
                for bi, b in tqdm(enumerate(bulksizes), leave=False):
                    for nbi, nb in enumerate(nbins):
                        for nvi, nv in enumerate(nvals):
                            input_file = f"../input/doubles_uniform_{nv}.root"
                            arg = f"-b{b} -h{nb} -f{input_file}"
                            print(arg)

                            r = subprocess.run([f, *arg.split()],
                                env={**os.environ, e: "1"},
                                check=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL
                                )
                            output = r.stdout.decode("utf-8").split()
                            times =  [output[i].split(":")[1] for i in range(3)]
                            file_handler.write(f"{iter},{e},{nb},{b},{input_file},{','.join(times)}\n")
                            file_handler.flush()

    # Create a dictionary with file names as keys and a dataframe containing the statistics for each kernel/api call.
    return
if __name__ == "__main__":
    # environ = sys.argv[1]
    # os.environ[envion] = "1"
    environs = [
   	"CPU",
#        "CUDA_HIST",
        # "SYCL_HIST",
    ]

    n = 3
    nbins = [
        1,
        2,
        5,
        10,
        20,
        50,
        100,
        500,
        1000,
        # 5000,
        # 10000,
        # 50000,
    ]

    bulksizes = [
        1,
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
        # 32768,
        # 65536,
        # 131072,
        # 262144,
    ]

    nvals = [
        # 1,
        100000000,    # 100M
        500000000,    # 500M
        1000000000,   # 1B
        5000000000,   # 5B
        # 10000000000,  # 10B
    ]
    f = "./histond_benchmark"

    run_benchmark(
        f,
        n,
        environs,
        bulksizes,
        nbins,
        nvals,
        f"{'%s/'.format(sys.argv[1]) if len(sys.argv) > 1 else ''}walltime-{time.strftime('%Y%m%d-%H%M%S')}",
    )
