cmake_minimum_required(VERSION 3.4.3)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra -Wpedantic)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")
project(dummy)


function(create_project)
    CMAKE_PARSE_ARGUMENTS(ARG "" "TARGET" "SOURCES;DEPENDENCIES;PACKAGES;EXTRA_LANGUAGES" ${ARGN})

    project(${ARG_TARGET} LANGUAGES C CXX ${ARG_EXTRA_LANGUAGES})

    foreach(pkg ${ARG_PACKAGES})
        find_package(${pkg} REQUIRED)
    endforeach()

    foreach(src ${ARG_SOURCES})
        add_executable("${ARG_TARGET}" "${src}")
    endforeach()

    if (ARG_DEPENDENCIES)
        target_link_libraries("${ARG_TARGET}" PRIVATE ${ARG_DEPENDENCIES})
    endif()
endfunction()

#########################################

set(ROOT_DIR $ENV{ROOTSYS})
if (pfm)
    find_package(PFM REQUIRED)
endif()
if (adaptivecpp)
    find_package(AdaptiveCpp REQUIRED)
endif()

find_package(ROOT REQUIRED PATHS ${ROOT_DIR})

create_project(TARGET generate_ntuple
               SOURCES generate_ntuple.C
               DEPENDENCIES ROOT::ROOTNTuple ROOT::Gpad ROOT::Core ROOT::MathCore)


enable_language(CUDA)
create_project(TARGET print_properties
               SOURCES print_properties.cu
               PACKAGES CUDA
               EXTRA_LANGUAGES CUDA)

add_subdirectory(benchmarks)
add_subdirectory(microbenchmarks)

