cmake_minimum_required(VERSION 3.15)

# Project setup
project(Tencor VERSION 0.1.0)

set(OpenCV_DIR C:\Users\USMAN-PC\Downloads\opencv\build)

# Enable testing (if you have tests)
include(CTest)
enable_testing()

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include OpenCV directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Add executable with the correct file path
add_executable(Tencor src/main.cpp)

# Link OpenCV libraries
target_link_libraries(Tencor ${OpenCV_LIBS})

# Set CPack project version for packaging
set(CPACK_PROJECT_NAME "Tencor")
set(CPACK_PROJECT_VERSION "0.1.0")

# Include CPack for packaging
include(CPack)
