cmake_minimum_required(VERSION 3.4...3.18)
project(LibraryName)

add_subdirectory(pybind11)
find_package(Eigen3 REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)

# Add subdirectories for source files
add_subdirectory(src/regression)
add_subdirectory(src/classification)
add_subdirectory(src/clustering)
add_subdirectory(src/dimensionality_reduction)
add_subdirectory(src/semi_supervised)
add_subdirectory(src/reinforcement_learning)
add_subdirectory(src/ensemble_learning)
add_subdirectory(src/neural_networks)
add_subdirectory(src/anomaly_detection)
add_subdirectory(src/bayesian_learning)
add_subdirectory(src/instance_based)
add_subdirectory(src/genetic_algorithms)
add_subdirectory(src/other)

# Pybind11 module
pybind11_add_module(LibraryName python/LibraryName.cpp)
target_link_libraries(LibraryName PRIVATE Eigen3::Eigen)
