cmake_minimum_required(VERSION 3.12)
project(TwoLayerNet)

configure_file(directories.h.in ${CMAKE_CURRENT_LIST_DIR}/directories.h @ONLY)

set(CMAKE_CXX_STANDARD 14)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(TwoLayerNet main.cpp Dataset.cpp Dataset.h TwoLayerNet.cpp TwoLayerNet.h)
target_link_libraries (TwoLayerNet Eigen3::Eigen)
