/*
 * @Author: ruoxi521
 * @Date: 2023-04-03 17:26:33
 * @LastEditors: ruoxi521
 * @LastEditTime: 2023-04-18 06:33:05
 * @FilePath: /Graduation_Project_ws/yolov8_ros/yolov8_trt_ros/src/yolov8_instance_segmentation_node.cpp
 * @Description: 
 * 
 * Copyright (c) 2023 by suruoxi521@gmail.com, All Rights Reserved. 
 */

#include <ros/ros.h>
#include <segment/yolov8_instance_segmentation.hpp>

int main(int argc, char** argv){
    ros::init(argc, argv, "yolov8_segmentation");
    ros::NodeHandle nodeHandle("~");
    yolov8_segment_ros::Yolov8InstanceSegmentor yolov8instancesegmentor(nodeHandle);

    ros::spin();
    return 0;
}