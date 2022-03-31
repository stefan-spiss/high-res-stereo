cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)
find_package(OpenCV 4 REQUIRED)

add_executable(highres-stereo highres-stereo.cpp)
target_link_libraries(highres-stereo "${TORCH_LIBRARIES}")
target_link_libraries(highres-stereo ${OpenCV_LIBS})
set_property(TARGET highres-stereo PROPERTY CXX_STANDARD 14)
