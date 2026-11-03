cmake_minimum_required(VERSION 3.18)

project(handout7 LANGUAGES C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # build/compile_commands.json for clangd/IntelliSense

# Executables are placed next to the sources so that the relative "data/" path used by
# the programs, ./plot_image.sh and compare.py work both from CLion and from a terminal.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/data ${CMAKE_CURRENT_SOURCE_DIR}/images)

# --- Sequential (plain C, builds everywhere) -----------------------------------
add_executable(sequential wave_2d_sequential.c)
target_compile_options(sequential PRIVATE -O2 -Wall -Wextra)
target_link_libraries(sequential PRIVATE m)

# --- Parallel (CUDA, only if nvcc is available, e.g. on the course GPU machine) ---
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # Build for the GPU in the machine you compile on; override with
    #   cmake -DCMAKE_CUDA_ARCHITECTURES=86 ..
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
        if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
            set(CMAKE_CUDA_ARCHITECTURES native)
        else()
            set(CMAKE_CUDA_ARCHITECTURES 70)
        endif()
    endif()

    add_executable(parallel wave_2d_parallel.cu)
    # Same flags as the Makefile: -fmad=false keeps results bit-comparable to the CPU
    target_compile_options(parallel PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O2 -fmad=false>)
    target_link_libraries(parallel PRIVATE m)
else()
    message(WARNING "No CUDA compiler (nvcc) found: only 'sequential' will be built. "
                    "Build on a machine with CUDA to get 'parallel'.")

    # --- Code insight without a GPU (e.g. macOS) ----------------------------------
    # Homebrew LLVM's clang can parse CUDA (host side only) against a header-only copy
    # of the CUDA toolkit. This gives CLion/clangd completion and error checking for
    # wave_2d_parallel.cu, including cooperative groups. The target is never linked;
    # building it is just a syntax/type check.
    set(CUDA_HEADERS_DIR "$ENV{HOME}/.local/cuda-12.4-headers" CACHE PATH
        "Header-only CUDA toolkit used for code insight when nvcc is unavailable")
    find_program(LLVM_CLANGXX clang++
        PATHS /opt/homebrew/opt/llvm/bin /usr/local/opt/llvm/bin NO_DEFAULT_PATH)

    if(EXISTS "${CUDA_HEADERS_DIR}/include/cooperative_groups.h" AND LLVM_CLANGXX)
        if(NOT CMAKE_CXX_COMPILER)
            set(CMAKE_CXX_COMPILER ${LLVM_CLANGXX}) # Apple clang can't find CUDA headers
        endif()
        enable_language(CXX)

        set_source_files_properties(wave_2d_parallel.cu PROPERTIES LANGUAGE CXX)
        add_library(parallel_intellisense OBJECT EXCLUDE_FROM_ALL wave_2d_parallel.cu)
        target_compile_options(parallel_intellisense PRIVATE
            -x cuda
            --cuda-path=${CUDA_HEADERS_DIR}
            --cuda-host-only
            -nocudalib
            -std=c++14
            -Wno-unknown-cuda-version)
        # CLion's engine doesn't run clang in CUDA mode; it only reads include paths and
        # macros from the flags. Define what nvcc defines implicitly so cooperative_groups.h
        # isn't empty. (It also ignores forced includes, so the .cu file includes
        # <cuda_runtime.h> explicitly.) Harmless for the real clang CUDA build above.
        target_include_directories(parallel_intellisense SYSTEM PRIVATE ${CUDA_HEADERS_DIR}/include)
        target_compile_definitions(parallel_intellisense PRIVATE __CUDACC__)
        message(STATUS "CUDA code insight enabled via ${LLVM_CLANGXX} + ${CUDA_HEADERS_DIR}")
    else()
        message(STATUS "CUDA code insight disabled (need 'brew install llvm' and headers in "
                       "${CUDA_HEADERS_DIR})")
    endif()
endif()

# --- Helpers (equivalents of the Makefile targets) --------------------------------
find_package(Python3 COMPONENTS Interpreter)

# cmake --build build --target plot   -> images/*.png via gnuplot
add_custom_target(plot
    COMMAND ./plot_image.sh data
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    USES_TERMINAL)

# cmake --build build --target movie  -> wave.mp4 via ffmpeg (run 'plot' first)
add_custom_target(movie
    COMMAND ffmpeg -y -an -i images/%5d.png -vcodec libx264 -pix_fmt yuv420p
            -profile:v baseline -level 3 -r 12 wave.mp4
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    USES_TERMINAL)

# cmake --build build --target check  -> runs both versions and compares output
if(TARGET parallel AND Python3_FOUND)
    add_custom_target(check
        COMMAND ${CMAKE_COMMAND} -E make_directory data data_sequential
        COMMAND ./sequential
        COMMAND ${CMAKE_COMMAND} -E copy_directory data data_sequential
        COMMAND ./parallel
        COMMAND ${Python3_EXECUTABLE} compare.py data_sequential/00000.dat data/00000.dat
        COMMAND ${Python3_EXECUTABLE} compare.py data_sequential/00075.dat data/00075.dat
        COMMAND ${CMAKE_COMMAND} -E rm -rf data_sequential
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDS sequential parallel
        USES_TERMINAL)
endif()

# cmake --build build --target clean_output  -> remove data/, images/, wave.mp4
add_custom_target(clean_output
    COMMAND ${CMAKE_COMMAND} -E rm -rf data images wave.mp4
    COMMAND ${CMAKE_COMMAND} -E make_directory data images
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
