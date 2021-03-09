cmake_minimum_required(VERSION 3.15)
project(02_levenshtein)

set(CMAKE_CXX_STANDARD 17)


include_directories(src/internal)
include_directories(src)


#find_library(OMP omp)
#find_package(OpenMP REQUIRED)
#target_link_libraries(levpar PRIVATE OpenMP::OpenMP_CXX)

add_executable(bla
        src/implementation.hpp
        src/potential.cpp)
