cmake_minimum_required(VERSION 3.5)
project(quadruped_control)

include_directories(${CMAKE_BINARY_DIR})
set(CMAKE_C_FLAGS "-O3 -ggdb -std=gnu99 -I -fPIC")
set(CMAKE_CXX_STANDARD 14)

find_package(RapidJSON)

include_directories(${CMAKE_BINARY_DIR})

add_subdirectory(third_party/qpOASES)
include_directories(include)
include_directories("/usr/local/include/eigen3") 
include_directories(include/sdk/include)
link_directories(include/sdk/lib)

set(EXTRA_LIBS -pthread libunitree_legged_sdk_amd64.so lcm)

set(CMAKE_CXX_FLAGS "-O3 -no-pie -ggdb -Wall ")


file(GLOB_RECURSE sources "include/*/*.h" "src/*/*.cpp" "BalanceController/*.cpp" "ConvexMPC/*.cpp" "include/*/*/*.h")

add_library(src ${sources})
target_link_libraries(src ${EXTRA_LIBS})

add_executable(quad_ctrl src/main.cpp)
target_link_libraries(quad_ctrl ${EXTRA_LIBS} src qpOASES)
