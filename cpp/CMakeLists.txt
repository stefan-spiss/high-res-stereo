set(source_files
    utils.cpp
    high_res_stereo.cpp
    )
set(include_dirs ${CMAKE_CURRENT_SOURCE_DIR})

set(library_name "${PROJECT_NAME}")

add_library(${library_name} SHARED ${source_files})
target_link_libraries(${library_name}
    PUBLIC ${TORCH_LIBRARIES}
    PUBLIC ${OpenCV_LIBS}
    PUBLIC TBB::tbb
    )
target_include_directories(${library_name}
    PUBLIC ${include_dirs}
    PUBLIC ${OPENCV_INCLUDE_DIRS}
    )

# Executable to test stereo matching library
add_executable(high_res_stereo_exec main.cpp)
target_link_libraries(high_res_stereo_exec
    PRIVATE ${library_name}
    )
