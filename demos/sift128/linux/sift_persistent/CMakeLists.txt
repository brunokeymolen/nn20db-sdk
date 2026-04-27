cmake_minimum_required(VERSION 3.16)
project(sift_persistent_demo C)

set(CMAKE_C_STANDARD 11)

# Point CMake at the SDK cmake config
set(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../..")
set(nn20db_DIR "${REPO_ROOT}/sdk/linux/current/cmake"
    CACHE PATH "Path to nn20dbConfig.cmake")

find_package(nn20db REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(HDF5 REQUIRED hdf5)

add_executable(sift_persistent_demo
    main.c
)

target_compile_options(sift_persistent_demo PRIVATE
    -O3 -Wall -Wextra
    ${HDF5_CFLAGS_OTHER}
)

target_include_directories(sift_persistent_demo PRIVATE
    ${HDF5_INCLUDE_DIRS}
)

target_link_directories(sift_persistent_demo PRIVATE
    ${HDF5_LIBRARY_DIRS}
)

target_link_libraries(sift_persistent_demo PRIVATE
    nn20db::nn20db
    ${HDF5_LIBRARIES}
)
