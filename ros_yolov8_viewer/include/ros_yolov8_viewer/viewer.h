/*
 * @Author: ruoxi521
 * @Date: 2023-05-23 10:02:36
 * @LastEditors: ruoxi521
 * @LastEditTime: 2023-06-07 23:19:31
 * @FilePath: /src/yolov8_ros/ros_yolov8_viewer/include/ros_yolov8_viewer/viewer.h
 * @Description: 
 * 
 * Copyright (c) 2023 by suruoxi521@gmail.com, All Rights Reserved. 
 */
#include "ros/ros.h"
#include "message_filters/subscriber.h"
#include "message_filters/time_synchronizer.h"
#include "image_transport/subscriber_filter.h"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>
#include "cv_bridge/cv_bridge.h"
#include "yolov8_msgs/BoundingBoxes.h"

#include <string>
#include <iostream>
#include <fstream>       // https://blog.csdn.net/allenlinrui/article/details/19639241

class ViewerNode
{
public:
  ViewerNode(const ros::NodeHandle& nh);
  //   ~ViewerNode();

  struct parameters
  {
    std::string boxes_topic;
    std::string image_topic;

    parameters()
    {
      ros::param::param<std::string>(ros::this_node::getName() + "/boxes_topic", boxes_topic, "inference");
      ros::param::param<std::string>(ros::this_node::getName() + "/image_topic", image_topic, "image_raw");
    }
  };
  const struct parameters params;

private:
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  int class_num;                        // 类别数目
  std::string label_path;               // 标签文件路径
  std::vector<std::string> labels;      // 标签名称列表

  message_filters::Subscriber<yolov8_msgs::BoundingBoxes> boxes_sub;
  image_transport::SubscriberFilter image_sub;

  message_filters::TimeSynchronizer<sensor_msgs::Image, yolov8_msgs::BoundingBoxes> sync;

  void syncCallback(const sensor_msgs::ImageConstPtr& image_msg, const yolov8_msgs::BoundingBoxesConstPtr& boxes_msg);
};