
create_project(TARGET histond_benchmark_cpu
               SOURCES ../benchmarks/histond_benchmark.C
               PACKAGES benchmark
               DEPENDENCIES ROOT::ROOTDataFrame ROOT::ROOTNTuple ROOT::MathCore ROOT::Core
                            benchmark::benchmark benchmark::benchmark_main)

# target_compile_definitions(histond_benchmark PUBLIC ROOT_RDF_CUDA TIME_FILL TIME_STATSset TIME_FINDBIN)
target_include_directories(histond_benchmark_cpu PUBLIC ${ROOT_DIR}/include)
# target_compile_definitions(histond_benchmark PUBLIC ROOT_RDF_CUDA LIKWID_PERFMON)

# target_include_directories(histond_benchmark PRIVATE /home/jollychen/likwid/include)
# target_link_directories(histond_benchmark PRIVATE /home/jollychen/likwid/lib)
# target_link_libraries(histond_benchmark PRIVATE likwid)

# ------------------------------------------------------------------------------------------------------------------------

create_project(TARGET histond_benchmark_cpu_timing
               SOURCES ../benchmarks/histond_benchmark.C
               PACKAGES benchmark
               DEPENDENCIES ROOT::ROOTDataFrame ROOT::ROOTNTuple ROOT::MathCore ROOT::Core
                            benchmark::benchmark benchmark::benchmark_main)
target_include_directories(histond_benchmark_cpu_timing PUBLIC ${ROOT_DIR}/include)
