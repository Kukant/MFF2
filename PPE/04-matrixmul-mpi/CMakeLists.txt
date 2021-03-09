cmake_minimum_required(VERSION 3.15)
project(04_matrix_mult)

set(CMAKE_CXX_STANDARD 17)


include_directories(src)


#find_library(OMP omp)
#find_package(OpenMP REQUIRED)
#target_link_libraries(levpar PRIVATE OpenMP::OpenMP_CXX)


add_executable(matrix_mult
        src/main.cpp
        src/write_utils.hpp
        )
#target_include_directories(matrix_mult PRIVATE /usr/local/opt/open-mpi/include)
target_link_libraries(matrix_mult ${TBB})