/*
 * @Author: ruoxi521
 * @Date: 2023-03-31 11:34:37
 * @LastEditors: ruoxi521
 * @LastEditTime: 2023-05-23 10:04:01
 * @FilePath: /Graduation_Project_ws/src/yolov8_ros/yolov8_trt_ros/include/detect/yolov8_object_detector.hpp
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
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

// yolov8_msgs
#include <yolov8_msgs/BoundingBox.h>
#include <yolov8_msgs/BoundingBoxes.h>
#include <yolov8_msgs/CheckForObjectsAction.h>
#include <yolov8_msgs/ObjectCount.h>

#include "utils/logging.h"
#include "utils/utils.h"

namespace yolov8_ros {

class Yolov8ObjectDetector {
  public:
    explicit Yolov8ObjectDetector(const ros::NodeHandle& nh);

    ~Yolov8ObjectDetector();

    struct parameters
    {
      int image_size;                         // 输入图像尺寸
      int batch_size;                         // 每次处理的图像数量
      std::string model_path;                 // TensorRT模型（.trt 或者 .engine）
      std::string label_path;                   // 推理结果存储路径
      bool show;                              // 是否实时显示推理
      int CLASSES, Num_box;                   // 总类别数， 总锚框输出数
      float CONF_THRESHOLD, NMS_THRESHOLD;    // 阈值设置
      
      parameters()
      {
        ros::param::param<int>("/yolov8_infer/image_size", image_size, 640);
        ros::param::param<int>("/yolov8_infer/batch_size", batch_size, 1);
        ros::param::param<std::string>("/yolov8_infer/model_path", model_path, 
                                      "/home/ruoxi/Graduation_Project_ws/src/yolov8_ros/yolov8_trt_ros/models/detect/yolov8n-b1.trt");
        ros::param::param<std::string>("/yolov8_infer/label_path", label_path, 
                                      "/home/ruoxi/Graduation_Project_ws/src/yolov8_ros/yolov8_trt_ros/labels/coco_labels.txt");
        ros::param::param<int>("/yolov8/nms_threshold", CLASSES, 80);
        ros::param::param<int>("/yolov8/Num_box", Num_box, 8400);
        ros::param::param<float>("/yolov8/conf_threshold", CONF_THRESHOLD, 0.2);
        ros::param::param<float>("/yolov8/nms_threshold", NMS_THRESHOLD, 0.5);
      }
    };

    const struct parameters params;

 private:
    /*!
    * Reads and verifies the ROS parameters.
    * @return true if successful.
    */
    bool readParameters();

    /*!
    * Initialize the ROS connections.
    */
    void init();

    // bCheck for objects action goal callback.
    void checkForObjectsActionGoalCB();

    // Check for objects action preempt callback.
    void checkForObjectsActionPreemptCB();

    /*!
    * Check if a preempt for the check for objects action has been requested.
    * @return false if preempt has been requested or inactive.
    */
    bool isCheckingForObjects() const;

    /*!
    * Publishes the detection image.
    * @return true if successful.
    */
    bool publishDetectionImage(const cv::Mat& detectionImage);

    //! Using.
    using CheckForObjectsActionServer = actionlib::SimpleActionServer<yolov8_msgs::CheckForObjectsAction>;
    using CheckForObjectsActionServerPtr = std::shared_ptr<CheckForObjectsActionServer>;

    //! ROS node handle.
    ros::NodeHandle nh_;

    //! Class labels.
    int numClasses_;
    std::vector<std::string> classLabels_;

    //! Check for objects action server.
    CheckForObjectsActionServerPtr checkForObjectsActionServer_;

    //! Advertise and subscribe to image topics.
    image_transport::ImageTransport it_;

    //! ROS subscriber and publisher.
    image_transport::Subscriber imageSubscriber_;
    ros::Publisher boundingBoxesPublisher_;

    //! Detected objects.
    std::vector<int> rosBoxCounter_;

    yolov8_msgs::BoundingBoxes bboxes_msg;

    void detectCallback(const sensor_msgs::ImageConstPtr& msg);
  
};

} /* namespace yolov8_ros*/
