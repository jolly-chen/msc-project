cmake_minimum_required(VERSION 3.4.3)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_ARCHITECTURES 75)
list(APPEND CMAKE_CXX_FLAGS -Wall)

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
find_package(ROOT REQUIRED PATHS ${ROOT_DIR})

create_project(TARGET generate_ntuple
               SOURCES generate_ntuple.C
               DEPENDENCIES ROOT::ROOTNTuple ROOT::Gpad ROOT::Core ROOT::MathCore)


add_subdirectory(benchmarks)
add_subdirectory(microbenchmarks)
