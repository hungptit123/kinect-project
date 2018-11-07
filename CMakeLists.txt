cmake_minimum_required(VERSION 3.12)
project(Demo)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

# set library freenect
find_library(libusb-1.0 REQUIRED)
find_package(Threads REQUIRED)
set(FREENECT_INCLUDE_DIRS /usr/local/include/libfreenect)
set(FREENECT_LIBRARIES /usr/local/lib/libfreenect.so)
include_directories(${FREENECT_INCLUDE_DIRS})

#set(OpenCV_FOUND 1)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake-modules)
set(SOURCE_FILES main.cpp libfreenect.hpp)

add_executable(Demo ${SOURCE_FILES})

# install library OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(Demo ${OpenCV_LIBS} )

# install library PCL
find_package(PCL 1.7 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})
link_directories(${PCL_LIBRARY_DIRS})

# install library OpenGL
find_package (OpenGL REQUIRED)
find_package(GLUT REQUIRED)
include_directories(${OPENGL_INCLUDE_DIR}  ${GLUT_INCLUDE_DIRS})
target_link_libraries(Demo ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} ${CMAKE_THREAD_LIBS_INIT} ${PCL_LIBRARIES} ${FREENECT_LIBRARIES})
