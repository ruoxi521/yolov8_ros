cmake_minimum_required(VERSION 3.0.2)
project(yolov8_trt_ros)

## Compile as C++11, supported in ROS Kinetic and newer
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")  # enable c++11 (TensorRT requirement)
# add_compile_options(-std=c++11)
# set(CMAKE_C_FLAGS "-Wall -Wno-unused-result -Wno-unknown-pragmas -Wno-unused-variable -Wfatal-errors -fPIC ${CMAKE_C_FLAGS}")
# set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV 4 REQUIRED)

# Find CUDA
find_package(CUDA QUIET)
if(CUDA_FOUND)
  find_package(CUDA REQUIRED)
  message(STATUS "CUDA Version: ${CUDA_VERSION_STRINGS}")
  message(STATUS "CUDA Libraries: ${CUDA_LIBRARIES}")
  set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS}
    -03
    -gencode arch=compute_30,code=sm_30
    -gencode arch=compute_35,code=sm_35
    -gencode arch=compute_50,code=[sm_50,compute_50]
    -gencode arch=compute_52,code=[sm_52,compute_52]
    -gencode arch=compute_61,code=sm_61
    -gencode arch=compute_62,code=sm_62
  )
  add_definitions(-DGPU)
else()
  list(APPEND LIBRARIES "m")
endif()

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
# option(CUDA_USE_STATIC_CUDA_RUNTIME ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)


include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

include_directories(/home/ruoxi/3rdparty/TensorRT-8.0.1.6/include)
link_directories(/home/ruoxi/3rdparty/TensorRT-8.0.1.6/lib)

include_directories(${CUDA_INCLUDE_DIRS} ${TRT_INCLUDE_DIR})
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
include_directories(${OpenCV_INCLUDE_DIRS})

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  message_generation
  roscpp
  rospy
  std_msgs
  sensor_msgs
  vision_msgs
  actionlib
  yolov8_msgs
  nodelet
)

## System dependencies are found with CMake's conventions
find_package(Boost REQUIRED COMPONENTS system)


## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

## Generate added messages and services with any dependencies listed here
# generate_messages(
#   DEPENDENCIES
#   sensor_msgs
#   vision_msgs
# )

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## To declare and build dynamic reconfigure parameters within this
## package, follow these steps:
## * In the file package.xml:
##   * add a build_depend and a exec_depend tag for "dynamic_reconfigure"
## * In this file (CMakeLists.txt):
##   * add "dynamic_reconfigure" to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * uncomment the "generate_dynamic_reconfigure_options" section below
##     and list every .cfg file to be processed

## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
  INCLUDE_DIRS 
    include
  # LIBRARIES 
  #   ${PROJECT_NAME}_lib
  CATKIN_DEPENDS 
    nodelet 
    cv_bridge 
    image_transport 
    message_runtime 
    roscpp 
    rospy 
    sensor_msgs 
    vision_msgs
    yolov8_msgs
  # DEPENDS
  #   system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

## Declare a C++ library
add_library(yolov8_detect_nodelet
  src/yolov8_object_detector.cpp
  src/yolov8_object_detector_node.cpp
  src/yolov8_object_detector_nodelet.cpp
)

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(yolov8_detect_nodelet ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})



## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
 
set(detect_trt_dir ${PROJECT_SOURCE_DIR}/../../Yolov8-tensorrt)  

set(TensorRT_ROOT /home/ruoxi/3rdparty/TensorRT-8.0.1.6)   # 需要改成自己的TensorRT的路径
message(STATUS TensorRT_ROOT = ${TensorRT_ROOT})

include_directories(
  # ${detect_trt_dir}
  # ${detect_trt_dir}/cmake/common.cmake
  ${TensorRT_ROOT}/samples/common
)

# set(LIBS 
#   ${PROJECT_SOURCE_DIR}/../yolov8-tensorrt-learning/yolov8/build/libyolov8.so
# )

add_executable(yolov8_detect_node
  src/yolov8_object_detector_node.cpp
  src/yolov8_object_detector.cpp
)

add_dependencies(yolov8_detect_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Rename C++ executable without prefix
## The above recommended prefix causes long target names, the following renames the
## target back to the shorter version for ease of user use
## e.g. "rosrun someones_pkg node" instead of "rosrun someones_pkg someones_pkg_node"
# set_target_properties(${PROJECT_NAME}_node PROPERTIES OUTPUT_NAME node PREFIX "")

## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against

target_link_libraries(yolov8_detect_node
        ${catkin_LIBRARIES}
        ${OpenCV_LIBS}
        -lpthread cuda
        nvinfer
        cudart
        nvonnxparser
        nvinfer_plugin
        # ${LIBS}
        ${CUDA_cublas_LIBRARY}
        ${TensorRT_LIBRARIES}
        )

target_link_libraries(yolov8_detect_nodelet
        ${catkin_LIBRARIES}
        ${OpenCV_LIBS}
        -lpthread cuda
        nvinfer
        cudart
        nvonnxparser
        nvinfer_plugin
        # ${LIBS} 
        ${CUDA_cublas_LIBRARY}
        ${TensorRT_LIBRARIES}
        )


add_definitions(-O2 -pthread)
