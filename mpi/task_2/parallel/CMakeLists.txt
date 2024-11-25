find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

add_executable(parallel_exec src/parallel.cpp)
target_link_libraries(parallel_exec ${MPI_C_LIBRARIES})