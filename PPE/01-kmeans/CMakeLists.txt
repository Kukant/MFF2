cmake_minimum_required(VERSION 3.15)
project(01_kmeans)

set(CMAKE_CXX_STANDARD 14)
find_library(TBB tbb)

include_directories(framework/internal)

add_executable(frame
        framework/implementation.hpp
        framework/k-means.cpp)

target_include_directories(frame PRIVATE /usr/local/opt/tbb/include)
target_link_libraries(frame ${TBB})

add_executable(ser
        serial/implementation.hpp
        serial/k-means.cpp)
