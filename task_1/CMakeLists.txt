cmake_minimum_required(VERSION 3.10)

project(RandomWalk)

set(CMAKE_CXX_STANDARD 11)

add_executable(sequential src/sequential.cpp)
set_target_properties(sequential PROPERTIES CXX_COMPILER g++)
target_compile_options(sequential PRIVATE -O3)

find_package(MPI REQUIRED)
add_executable(parallel src/parallel.cpp)
set_target_properties(parallel PROPERTIES CXX_COMPILER mpic++)
target_link_libraries(parallel PRIVATE MPI::MPI_CXX)
target_include_directories(parallel PRIVATE ${MPI_INCLUDE_PATH})
target_compile_options(parallel PRIVATE -O3)
