/*
 * @Author: ruoxi521
 * @Date: 2023-04-03 17:42:34
 * @LastEditors: ruoxi521
 * @LastEditTime: 2023-05-02 23:31:18
 * @FilePath: /yolov8_ros/yolov8_trt_ros/src/yolov8_instance_segment_nodelet.cpp
 * @Description: 
 * 
 * Copyright (c) 2023 by suruoxi521@gmail.com, All Rights Reserved. 
 */


#include <ros/ros.h>
#include <nodelet/nodelet.h>                          // nodelet
#include <segment/yolov8_instance_segmentation.hpp>
#include <pluginlib/class_list_macros.h>              // pluginlib

class Yolov8InstanceSegmentNodelet : public nodelet::Nodelet
{
  public:
    Yolov8InstanceSegmentNodelet() = default;

    ~Yolov8InstanceSegmentNodelet() {
        if (yolov8_segment_ros_) delete yolov8_segment_ros_;
    }
    
  private:
    virtual void onInit() {
        ros::NodeHandle NodeHandle("~");
        NodeHandle = getPrivateNodeHandle();

        yolov8_segment_ros_ = new yolov8_segment_ros::Yolov8InstanceSegmentor(NodeHandle);
    }
  yolov8_segment_ros::Yolov8InstanceSegmentor *yolov8_segment_ros_;
}

// declare as a pluginlib
PLUGINLIB_EXPORT_CLASS(Yolov8InstanceSegmentNodelet, nodelet::Nodelet);
