
create_project(TARGET histond_benchmark
               SOURCES histond_benchmark.C
               PACKAGES benchmark
               DEPENDENCIES ROOT::ROOTDataFrame ROOT::ROOTNTuple ROOT::MathCore ROOT::Core
                            benchmark::benchmark benchmark::benchmark_main)

target_include_directories(histond_benchmark PUBLIC ${ROOT_DIR}/include)

if (cuda)
    target_link_libraries(histond_benchmark PUBLIC ROOT::CUDAHist)
    target_compile_definitions(histond_benchmark PUBLIC ROOT_RDF_CUDA)
endif()

if (sycl)
    target_link_libraries(histond_benchmark PUBLIC ROOT::SYCLHist)
    target_compile_definitions(histond_benchmark PUBLIC ROOT_RDF_SYCL)
endif()

# target_compile_definitions(histond_benchmark PUBLIC ROOT_RDF_CUDA LIKWID_PERFMON)
# target_include_directories(histond_benchmark PRIVATE /home/jollychen/likwid/include)
# target_link_directories(histond_benchmark PRIVATE /home/jollychen/likwid/lib)
# target_link_libraries(histond_benchmark PRIVATE likwid)

# ------------------------------------------------------------------------------------------------------------------------