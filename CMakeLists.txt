cmake_minimum_required(VERSION 3.10)
project(Jetson_Inference_Project LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)

cmake_policy(SET CMP0104 NEW)
set(CMAKE_CUDA_ARCHITECTURES 87)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

set(TENSORRT_LIB "/usr/lib/aarch64-linux-gnu/libnvinfer.so")

set(COMMON_INCLUDES
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    "/usr/local/cuda/include"
    "/usr/include/aarch64-linux-gnu"
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set(COMMON_LIBS
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${TENSORRT_LIB}
    cudart
    m 
    pthread
)

set(APP_NAME "inference_app")
set(APP_SOURCES 
    "main.cpp" 
    "InferencePipeline.cpp" 
    "dbscan.cpp"
    "MemoryState.cpp"
    "preprocess.cu"   
)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
    add_executable(${APP_NAME} ${APP_SOURCES})
    set_target_properties(${APP_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    target_include_directories(${APP_NAME} PRIVATE ${COMMON_INCLUDES})
    target_link_libraries(${APP_NAME} ${COMMON_LIBS})
endif()