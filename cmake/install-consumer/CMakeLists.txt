cmake_minimum_required(VERSION 3.24)

# The API is compiled as C. CXX is enabled solely so CMake can supply the C++
# runtime required by a static sgBLAS library's implementation.
project(sgblas_install_consumer LANGUAGES C CXX)

find_package(sgblas 0.1 CONFIG REQUIRED)

add_executable(sgblas_install_consumer main.c)
target_link_libraries(sgblas_install_consumer PRIVATE sgblas::sgblas)
