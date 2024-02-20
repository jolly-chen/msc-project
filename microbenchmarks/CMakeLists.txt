
create_project(TARGET cpu_calibration
               SOURCES cpu_calibration.cpp
               PACKAGES benchmark
               DEPENDENCIES benchmark::benchmark benchmark::benchmark_main)
# target_link_libraries(cpu-microbenchmarks PUBLIC pfm)

enable_language(CUDA)
create_project(TARGET investigational_benchmarks
               SOURCES investigational_benchmarks.cu
               PACKAGES benchmark CUDA
               DEPENDENCIES benchmark::benchmark benchmark::benchmark_main
               EXTRA_LANGUAGES CUDA)

create_project(TARGET gpu-microbenchmarks
               SOURCES gpu-microbenchmarks.cu
               PACKAGES benchmark CUDA
               DEPENDENCIES benchmark::benchmark benchmark::benchmark_main
               EXTRA_LANGUAGES CUDA)

#create_project(TARGET simulate_cache
#               SOURCES simulate_cache.cpp)
#target_include_directories(simulate_cache PUBLIC ${Pycachesim_DIR})
