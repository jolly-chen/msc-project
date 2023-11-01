
create_project(TARGET histond_benchmark
               SOURCES histond_benchmark.C
               PACKAGES benchmark
               DEPENDENCIES ROOT::ROOTDataFrame ROOT::ROOTNTuple ROOT::MathCore ROOT::Core ROOT::CUDAHist
                            benchmark::benchmark benchmark::benchmark_main)
target_compile_definitions(histond_benchmark PUBLIC ROOT_RDF_CUDA)