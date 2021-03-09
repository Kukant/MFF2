cmake_minimum_required(VERSION 3.15)
project(02_levenshtein)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories(framework/internal)

add_executable(levpar
        framework/implementation.hpp
        framework/levenshtein.cpp)

#find_library(OMP omp)
#find_package(OpenMP REQUIRED)
#target_link_libraries(levpar PRIVATE OpenMP::OpenMP_CXX)

target_include_directories(levpar PRIVATE /usr/local/opt/libomp/include)
target_link_libraries(levpar ${OMP})

include_directories(serial)

add_executable(levser
        serial/implementation.hpp
        serial/levenshtein.cpp)
