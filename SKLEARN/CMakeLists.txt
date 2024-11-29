# Specify the minimum version of CMake
cmake_minimum_required(VERSION 3.14)

# Project information
project(SKLEARN LANGUAGES CXX)

# Specify C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# macOS-specific settings
if(APPLE)
    # Enforce architecture for macOS
    set(CMAKE_OSX_ARCHITECTURES "arm64")
    # Add framework search path for Python
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -F/opt/homebrew/Frameworks")
endif()

# Find Python (interpreter, libraries, and headers)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Debug Python paths
message(STATUS "Python executable: ${Python3_EXECUTABLE}")
message(STATUS "Python include dir: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python library: ${Python3_LIBRARIES}")

# Include directories
include_directories(${Python3_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)

# Find pybind11
find_package(pybind11 REQUIRED)
include_directories(${pybind11_INCLUDE_DIRS})

# Find Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

# Add Python bindings module
pybind11_add_module(SKLEARN python/SKLEARN.cpp)
target_include_directories(SKLEARN PRIVATE ${CMAKE_SOURCE_DIR}/include ${EIGEN3_INCLUDE_DIRS})
target_link_libraries(SKLEARN PRIVATE ${Python3_LIBRARIES} Eigen3::Eigen)

# Add a library for ElasticNetRegression
add_library(regression_ElasticNetRegression src/regression/ElasticNetRegression.cpp)
target_include_directories(regression_ElasticNetRegression PRIVATE ${CMAKE_SOURCE_DIR}/include ${EIGEN3_INCLUDE_DIRS})
target_link_libraries(regression_ElasticNetRegression Eigen3::Eigen)

# Add an executable for testing ElasticNetRegression
add_executable(test_ElasticNetRegression tests/regression/test_ElasticNetRegression.cpp)
target_include_directories(test_ElasticNetRegression PRIVATE ${CMAKE_SOURCE_DIR}/include ${EIGEN3_INCLUDE_DIRS})
target_link_libraries(test_ElasticNetRegression PRIVATE regression_ElasticNetRegression Eigen3::Eigen)

# Add testing framework
enable_testing()
add_test(NAME ElasticNetRegressionTest COMMAND test_ElasticNetRegression)

# Add more modules, libraries, and tests as needed
# Example:
# add_library(classification_SVM src/classification/SVM.cpp)
# add_executable(test_SVM tests/classification/test_SVM.cpp)
# target_link_libraries(test_SVM classification_SVM Eigen3::Eigen)
