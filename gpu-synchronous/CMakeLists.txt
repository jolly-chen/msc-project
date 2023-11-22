
create_project(TARGET histond_benchmark_gpu
               SOURCES ../benchmarks/histond_benchmark.C
               PACKAGES benchmark
               DEPENDENCIES ROOT::ROOTDataFrame ROOT::ROOTNTuple ROOT::MathCore ROOT::Core ROOT::CUDAHist
                            benchmark::benchmark benchmark::benchmark_main)

# target_compile_definitions(histond_benchmark_gpu PUBLIC ROOT_RDF_CUDA TIME_FILL TIME_STATSset TIME_FINDBIN)
target_compile_definitions(histond_benchmark_gpu PUBLIC ROOT_RDF_CUDA)
target_include_directories(histond_benchmark_gpu PUBLIC ${ROOT_DIR}/include)
# target_compile_definitions(histond_benchmark_gpu PUBLIC ROOT_RDF_CUDA LIKWID_PERFMON)

# target_include_directories(histond_benchmark_gpu PRIVATE /home/jollychen/likwid/include)
# target_link_directories(histond_benchmark_gpu PRIVATE /home/jollychen/likwid/lib)
# target_link_libraries(histond_benchmark_gpu PRIVATE likwid)

# ------------------------------------------------------------------------------------------------------------------------

create_project(TARGET histond_benchmark_gpu_timing
               SOURCES ../benchmarks/histond_benchmark.C
               PACKAGES benchmark
               DEPENDENCIES ROOT::ROOTDataFrame ROOT::ROOTNTuple ROOT::MathCore ROOT::Core ROOT::CUDAHist
                            benchmark::benchmark benchmark::benchmark_main)
target_compile_definitions(histond_benchmark_gpu_timing PUBLIC ROOT_RDF_CUDA TIME_FILL TIME_STATS TIME_FINDBIN)
target_include_directories(histond_benchmark_gpu_timing PUBLIC ${ROOT_DIR}/include)
