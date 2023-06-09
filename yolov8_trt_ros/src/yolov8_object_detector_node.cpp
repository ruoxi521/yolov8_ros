/*
 * @Author: ruoxi521
 * @Date: 2023-03-31 11:32:12
 * @LastEditors: ruoxi521
 * @LastEditTime: 2023-06-07 18:14:40
 * @FilePath: /src/yolov8_ros/yolov8_trt_ros/src/yolov8_object_detector_node.cpp
 * @Description: 
 * 
 * Copyright (c) 2023 by suruoxi521@gmail.com, All Rights Reserved. 
 */


#include <ros/ros.h>
#include <detect/yolov8_object_detector.hpp>

extern void setup(int image_size, int batch_size, std::string model_path,
		              std::string label_path, int classes_num, int num_box, 
		              float conf_threshold, float nms_threshold);
extern void destroy(void);
extern yolov8_msgs::BoundingBoxes infer(const sensor_msgs::ImageConstPtr& color_msgs, int batchsize);

namespace yolov8_ros{

Yolov8ObjectDetector::Yolov8ObjectDetector(const ros::NodeHandle& nh) : nh_ (nh), it_ (nh), params() 
{ 
  setup(params.image_size, params.batch_size, params.model_path, params.label_path, params.CLASSES, params.Num_box, params.CONF_THRESHOLD,params.NMS_THRESHOLD);
  imageSubscriber_ = it_.subscribe("input_topic", 10, &Yolov8ObjectDetector::detectCallback, this);
  boundingBoxesPublisher_ = nh_.advertise<yolov8_msgs::BoundingBoxes>("output_topic", 10); 
}

Yolov8ObjectDetector::~Yolov8ObjectDetector()
{
  destroy();
}

void Yolov8ObjectDetector::detectCallback(const sensor_msgs::ImageConstPtr& color_msg)
{
  int batches = 1;
  bboxes_msg = infer(color_msg, batches);
  boundingBoxesPublisher_.publish(bboxes_msg);
}

}

int main(int argc, char** argv) {
  ros::init(argc, argv, "yolov8_object_detector");
  ros::NodeHandle nodeHandle("~");
  yolov8_ros::Yolov8ObjectDetector Detect(nodeHandle);

  ros::spin();
  return 0;
}

