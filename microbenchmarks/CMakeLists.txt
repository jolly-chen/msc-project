create_project(TARGET investigational_benchmarks
               SOURCES investigational_benchmarks.cu
               PACKAGES benchmark CUDA
               DEPENDENCIES benchmark::benchmark benchmark::benchmark_main
               EXTRA_LANGUAGES CUDA)
target_compile_options(investigational_benchmarks PUBLIC -Wno-pedantic)
# target_link_libraries(investigational_benchmarks PUBLIC pfm)

create_project(TARGET calibration_microbenchmarks
               SOURCES calibration_microbenchmarks.cu
               PACKAGES benchmark CUDA
               DEPENDENCIES benchmark::benchmark benchmark::benchmark_main
               EXTRA_LANGUAGES CUDA)

#create_project(TARGET simulate_cache
#               SOURCES simulate_cache.cpp)
#target_include_directories(simulate_cache PUBLIC ${Pycachesim_DIR})
