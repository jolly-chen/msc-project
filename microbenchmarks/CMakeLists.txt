
create_project(TARGET cpu-microbenchmarks
               SOURCES cpu-microbenchmarks.cpp
               PACKAGES benchmark
               DEPENDENCIES benchmark::benchmark benchmark::benchmark_main)
