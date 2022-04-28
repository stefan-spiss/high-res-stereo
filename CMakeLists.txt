cmake_minimum_required(VERSION 3.20)
project(high_res_stereo)

# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CUDA_STANDARD 17)
# set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
# set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
# set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wextra -DDEBUG")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20 -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -std=c++20 -Wall -Wextra -DDEBUG")

find_package(Torch REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package(TBB REQUIRED)

add_subdirectory(cpp)

add_executable(highres-stereo highres-stereo.cpp)
target_link_libraries(highres-stereo "${TORCH_LIBRARIES}")
target_link_libraries(highres-stereo ${OpenCV_LIBS})
target_link_libraries(highres-stereo TBB::tbb)
