# yolov8_ros
Deploy yolov8 inference accelerated by TensorRT as a ROS feature pack

```shell
ruoxi@ruoxi:~/SLAM/src/Graduation_Project_ws$ tree
.
└── yolov8_ros
    ├── README.md
    ├── yolov8_msgs
    │   ├── action
    │   │   └── CheckForObjects.action
    │   ├── CMakeLists.txt
    │   ├── include
    │   ├── msg
    │   │   ├── BoundingBoxes.msg
    │   │   ├── BoundingBox.msg
    │   │   └── Objectcount.msg
    │   ├── package.xml
    │   └── src
    ├── yolov8-tensorrt-learning
    │   ├── build
    │   ├── cmake
    │   │   └── common.cmake
    │   ├── data
    │   │   ├── 6406407.jpg
    │   │   ├── bus.jpg
    │   │   ├── video@copyright.md
    │   │   ├── videoplayback1.mp4
    │   │   ├── videoplayback2.mp4
    │   │   └── yolov8
    │   │       ├── engine
    │   │       │   ├── yolov8m.engine
    │   │       │   └── yolov8n.engine
    │   │       ├── onnx_not_dynamic
    │   │       │   ├── yolov8m.onnx
    │   │       │   ├── yolov8m-seg.onnx
    │   │       │   ├── yolov8n.onnx
    │   │       │   └── yolov8n-seg.onnx
    │   │       ├── yolov8m-b4.onnx
    │   │       ├── yolov8m-b4.trt
    │   │       ├── yolov8m-b8.onnx
    │   │       ├── yolov8m-b8.trt
    │   │       ├── yolov8n-b1.onnx
    │   │       ├── yolov8n-b1.trt
    │   │       ├── yolov8n-b4.onnx
    │   │       ├── yolov8n-b4.trt
    │   │       ├── yolov8n-b8.onnx
    │   │       └── yolov8n-b8.trt
    │   ├── README.md
    │   ├── usage.md
    │   ├── utils
    │   │   ├── common_include.h
    │   │   ├── kernel_function.cu
    │   │   ├── kernel_function.h
    │   │   ├── utils.cpp
    │   │   ├── utils.h
    │   │   ├── yolo.cpp
    │   │   └── yolo.h
    │   └── yolov8
    │       ├── app_yolov8.cpp
    │       ├── build.sh
    │       ├── CMakeLists.txt
    │       ├── decode_yolov8.cu
    │       ├── decode_yolov8.h
    │       ├── README.md
    │       ├── yolov8.cpp
    │       └── yolov8.h
    └── yolov8_trt_ros
        ├── CMakeLists.txt
        ├── include
        │   ├── image_interface.h
        │   └── yolov8_object_detector.hpp
        ├── launch
        ├── package.xml
        └── src
            ├── image_interface.cpp
            ├── yolov8_object_detector.cpp
            ├── yolov8_trt_ros_node.cpp
            └── yolov8_trt_ros_nodelet.cpp

60 directories, 189 files

```

```shell
├── CMakeLists.txt
├── images
│   ├── bus.jpg
│   ├── outtput
│   │   ├── l.jpg
│   │   ├── m.jpg
│   │   ├── n.jpg
│   │   ├── output.jpg
│   │   ├── s.jpg
│   │   └── x.jpg
│   └── zidane.jpg
├── logging.h
├── main1_onnx2trt.cpp
├── main2_trt_infer.cpp
├── models
│   ├── test
│   │   ├── yolov8m-seg.onnx
│   │   ├── yolov8n.engine
│   │   ├── yolov8n.onnx
│   │   ├── yolov8n-seg.engine
│   │   ├── yolov8n-seg.onnx
│   │   ├── yolov8n-seg.trt
│   │   └── yolov8n.trt
│   ├── yolov8n.engine
│   ├── yolov8n.onnx
│   ├── yolov8n-seg.engine
│   ├── yolov8n-seg.onnx
│   ├── yolov8s-seg.engine
│   └── yolov8s-seg.onnx
├── README.md
└── utils.h

16 directories, 93 files
```