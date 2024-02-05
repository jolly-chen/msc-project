import ROOT
import subprocess
from pathlib import Path

import os
import sys
import time

base_header = "iter,env,nbins,bulksize,input,edges,tfindbin,tfill,tstats,ttotal"

def run_benchmark(f, n, environs, bulksizes, nbins, input_files, output_file=""):
    if not Path(output_file).exists():
        with open(output_file, "w") as file_handler:
            file_handler.write(
                f"{base_header}\n"
            )

    with open(output_file, "a") as file_handler:
        for iter in range(n):
            for ei, e in enumerate(environs):
                for nbi, nb in enumerate(nbins):
                    for bi, b in enumerate(bulksizes):
                        for edges in ["", "-e"]:
                            procs = []
                            write_result = '-w' if iter == 0 and bi == 0 else ''
                            for nif, ipf in enumerate(input_files):
                                input_file = f"{input_folder}/{ipf}"
                                stem = Path(ipf).stem
                                cmd = f""
                                arg = f"-b{b} -h{nb} -f{input_file} {edges} {write_result}"
                                print(f"{cmd} {f} {arg}")

                                p = subprocess.Popen(
                                    [*f"{cmd} {f} {arg}".split()],
                                    env={**os.environ, e: "1"},
                                    # check=True,
                                    stdout=subprocess.PIPE,
                                    # shell=True,
                                )
                                procs.append((p, stem))

                            for p, stem in procs:
                                # r.wait()
                                # output = r.stdout.decode("utf-8").strip().split("\n")
                                output, stderr = p.communicate()

                                if stderr:
                                    print("JOB FAILED:")
                                    print(stderr)
                                else:
                                    output = output.decode("utf-8").strip().split("\n")
#                                    print(output)
                                    times = [o.split(":")[1] for o in output]
                                    file_handler.write(
                                        f"{iter},{e},{nb},{b},{stem},{'True' if edges != '' else 'False'},{','.join(times)}\n"
                                    )
                                    file_handler.flush()

                                    if write_result:
                                        r_file = f"{stem}_h{nb}_e{'1' if edges != '' else '0'}"
                                        print(f"zstd --rm -f -o {input_folder}/expected/{r_file} {r_file}.out"),
                                        subprocess.run(
                                                f"zstd --rm -f -o {input_folder}/expected/{r_file} {r_file}.out",
                                            # [
                                            #     "zstd",
                                            #     "--rm",
                                            #     f"{r_file}.out",
                                            #     f"-o{r_file}",
                                            # ],
                                            check=True,
                                            shell=True,
                                        )

                                    # subprocess.run(
                                    #     [
                                    #         "mv",
                                    #         f"{stem}_h{nb}_e{'1' if edges != '' else '0'}.out",
                                    #         f"{input_folder}/expected/{stem}_h{nb}_e{'1' if edges != '' else '0'}",
                                    #     ],
                                    #     check=True,
                                    # )


if __name__ == "__main__":
    environs = [
        "CPU",
    ]

    n = 5
    nbins = [
        #       1,
        #       2,
        #       5,
        10,
        #       20,
        #       50,
        #       100,
        #       500,
        1000,
        #       5000,
        #       10000,
        #       50000,
        100000,   # 100K
        10000000, # 10M
    ]

    bulksizes = [
               1,
        # 2,
        # 4,
               8,
        # 16,
        # 32,
               64,
        # 128,
        # 256,
               512,
        # 1024,
        # 2048,
               4096,
        # 8192,
        # 16384,
               32768,
        # 65536,
        # 131072,
               262144,
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

        "doubles_normal-0.4-0.1_50000000.root",  # 50M
        "doubles_normal-0.4-0.1_100000000.root",  # 100M
        "doubles_normal-0.4-0.1_500000000.root",  # 500M
        "doubles_normal-0.4-0.1_1000000000.root",  # 1B

        "doubles_normal-0.7-0.01_50000000.root",  # 50M
        "doubles_normal-0.7-0.01_100000000.root",  # 100M
        "doubles_normal-0.7-0.01_500000000.root",  # 500M
        "doubles_normal-0.7-0.01_1000000000.root",  # 1B
    ]

    input_folder = "/data/jolly/input"
    f = "/data/jolly/msc-project-sm86/benchmarks/histond_benchmark"
    if len(sys.argv) > 1:
        output_file = sys.argv[1]
    else:
        output_file = f"{time.strftime('%Y%m%d-%H%M%S')}"

    output_folder = "/data/jolly/msc-project-sm86/cpu-sequential/l4-cpu"
    os.makedirs(output_folder, exist_ok=True)
    print(f"writing results to {output_folder}/{output_file}...")

    run_benchmark(
        f, n, environs, bulksizes, nbins, input_files, f"{output_folder}/{output_file}"
    )
