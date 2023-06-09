/*
 * @Author: ruoxi521
 * @Date: 2023-04-03 16:59:52
 * @LastEditors: ruoxi521
 * @LastEditTime: 2023-05-16 22:34:59
 * @FilePath: /yolov8_ros/yolov8_trt_ros/include/segment/yolov8_instance_segmentation.hpp
 * @Description: 
 * 
 * Copyright (c) 2023 by suruoxi521@gmail.com, All Rights Reserved. 
 */

#pragma once  // 防止头文件被重复引用

//c++
#include <pthread.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

//ROS
#include <actionlib/server/simple_action_server.h>
#include <geometry_msgs/Point.h>
#include <image_transport/image_transport.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/Header.h>

// OpenCV
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

// yolov8_msgs segment
#include <yolov8_msgs/BoundingBox.h>
#include <yolov8_msgs/BoundingBoxes.h>
#include <yolov8_msgs/MaskBox.h>
#include <yolov8_msgs/MaskBoxes.h>
#include <yolov8_msgs/CheckForSegmentationAction.h>
#include <yolov8_msgs/SegmentationCount.h>

// GPU
#ifdef GPU
#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "curand.h"
#endif

extern "C" {
#include <yolov8/yolov8.h>
}

namespace yolov8_ros{

//! Mask box of the segmentation instance
typedef struct {
  float x, y, w, h, prob;
  int num, id;
  cv::Mat boxMask; //矩形框中添加掩膜
} MaskBox_;

// get the image with time stmap
typedef struct {
  IplImage* image;
  std_msgs::Header header;
} IplImageWithHeader_;

class Yolov8InstanceSegmentor {
  public:

    // 构造函数
    Yolov8InstanceSegmentor(ros::NodeHandle nh);

    // 析构函数
    ~Yolov8InstanceSegmentor();

    struct parameters
    {
      int cam_id;           // 相机输入端口
      int image_size;       // 输入图像尺寸
      int batch_size;       // batch大小
      std::string model_path;    // TensorRT模型（.trt 或者 .engine）
      std::string savepath;      // 推理结果存储路径
      bool show;            // 是否实时显示推理
      
      parameters()
      {
        ros::param::param<int>("/yolov8_infer/cam_id", cam_id, 0);
        ros::param::param<int>("/yolov8_infer/image_size", image_size, 640);
        ros::param::param<int>("/yolov8_infer/batch_size", batch_size, 1);
        ros::param::param<std::string>("/yolov8_infer/label_path", model_path, 
                                      "/home/ruoxi/Graduation_Project_ws/src/yolov8_ros/yolov8_trt_ros/models/detect/yolov8n-b1.onnx");
        ros::param::param<std::string>("/yolov8_infer/model_path", model_path, 
                                      "/home/ruoxi/Graduation_Project_ws/src/yolov8_ros/yolov8_trt_ros/result/detect");
        ros::param::param<bool>("/yolov8_infer/show", show, false);
      }
    };

    const struct parameters params;
  private:
    bool readParameters();
    void init();

    // callback of camera
    void cameraCallback(const sensor_msgs::ImageConstPtr& msg);

    // check for object action goal callback
    void checkForSegmentationActionGoalCB();

    // Check for objects action preempt(抢占) callback

    void checkForSegmentationActionPreemptCB();

    // check if a preempt for the check fo segment action has been requested
    bool isCheckingForSegments() const;

    // Publishes the Segment image
    bool publishSegmentImage(const cv::Mat& SegmentationImage);

    // Using
    using CheckForSegmentsActionServer = actionlib::SimpleActionServer<yolov8_msgs::CheckForSegmentationAction>;
    using CheckForSegmentsActionServerPtr = std::shared_ptr<CheckForSegmentsActionServer>;

    // ROS Part----------------------------------------------------

    ros::NodeHandle nh_;

    // Class labels
    int numClasses_;
    std::vector<std::string> classLabels_;

    // Check for segment action server
    CheckForSegmentsActionServerPtr checkForSegmentsActionServer_;

    // Advertise and subscribe to image topics
    image_transport::ImageTransport it_;

    // ROS subscriber and publisher
    image_transport::Subscriber imageSubscriber_;
    ros::Publisher instancePublisher_;
    ros::Publisher MaskBoxesPublisher_;

    // Segmented Instance
    std::vector<std::vector<MaskBox_> > MaskBoxes_;
    std::vector<int> MaskBoxesCounter_;
    yolov8_msgs::MaskBoxes MaskBoxesResults_;

    // Publisher of the Mask box image
    ros::Publisher segmentImagePublisher_;

    void segmentCallback(const sensor_msgs::ImageConstPtr& msg);

};

} /* namespace yolov8_ros*/
