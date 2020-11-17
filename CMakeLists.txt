cmake_minimum_required(VERSION 3.2)

project(torchcxx C CXX)

# MUST append libtorch cmake path: /path/to/libtorch/share/cmake
list(APPEND CMAKE_PREFIX_PATH "/apps/lib/torchgpu/share/cmake")

find_package(gflags REQUIRED)
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

include_directories(${OpenCV_INCLUDE_DIRS} ${gflags_INCLUDE_DIR})

add_executable(torchcxx main.cc)
target_link_libraries(torchcxx ${TORCH_LIBRARIES} ${OpenCV_LIBS} ${GFLAGS_LIBRARIES})
