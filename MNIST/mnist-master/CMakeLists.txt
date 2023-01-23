cmake_minimum_required(VERSION 3.15)

project(parse_mnist)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(parse_mnist parse_mnist.cpp)


if (CMAKE_HOST_SYSTEM_NAME MATCHES "Linux")
    if (CMAKE_SYSTEM_NAME MATCHES "Linux")
        #set(OpenCV_DIR "/home/zz/soft/opencv-4.5.1/lib/cmake/opencv4" CACHE PATH "") # shared
        set(OpenCV_DIR "/home/zz/soft/opencv-4.5.1-static/lib/cmake/opencv4" CACHE PATH "") # static
    elseif (ANDROID)
        set(OpenCV_DIR "/home/zz/soft/opencv-4.5.1-android/sdk/native/jni" CACHE PATH "")
    endif()
elseif (CMAKE_HOST_SYSTEM_NAME MATCHES "Windows")
    if (CMAKE_SYSTEM_NAME MATCHES "Windows")
        set(OpenCV_DIR "E:/lib/opencv/4.5.0" CACHE PATH "")
    elseif (ANDROID)
        set(OpenCV_DIR "E:/soft/Android/opencv-4.5.0-android-sdk/sdk/native/jni" CACHE PATH "")
    endif()
elseif (CMAKE_HOST_SYSTEM_NAME MATCHES "Darwin")
    if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
        set(OpenCV_DIR "/usr/local/opencv-4.2/lib/cmake/opencv4" CACHE PATH "")
    elseif (ANDROID)
        set(OpenCV_DIR "/Users/chris/soft/opencv-4.5.0-android-sdk/sdk/native/jni" CACHE PATH "")
    endif()
endif()


find_package(OpenCV REQUIRED)
target_link_libraries(parse_mnist ${OpenCV_LIBS})

