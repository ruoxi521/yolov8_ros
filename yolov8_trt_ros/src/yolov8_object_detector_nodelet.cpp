/*
 * @Author: ruoxi521
 * @Date: 2023-03-31 11:32:52
 * @LastEditors: ruoxi521
 * @LastEditTime: 2023-05-21 20:38:53
 * @FilePath: /Graduation_Project_ws/src/yolov8_ros/yolov8_trt_ros/src/yolov8_object_detector_nodelet.cpp
 * @Description: 
 * 
 * Copyright (c) 2023 by suruoxi521@gmail.com, All Rights Reserved. 
 */


#include <ros/ros.h>
#include <nodelet/nodelet.h>
#include <detect/yolov8_object_detector.hpp>
#include <pluginlib/class_list_macros.h>


extern void setup(int image_size, int batch_size, std::string model_path,
		              std::string label_path, int classes_num, int num_box, 
		              float conf_threshold, float nms_threshold);
extern void destroy(void);
extern yolov8_msgs::BoundingBoxes infer(const sensor_msgs::ImageConstPtr& color_msgs, int batchsize);

class Yolov8ObjectDetectorNodelet : public nodelet::Nodelet {
    public:
        Yolov8ObjectDetectorNodelet() = default;

        ~Yolov8ObjectDetectorNodelet() {
        if (yolov8_detector_ros_) delete yolov8_detector_ros_;
        }

    private:
        // nodelet initialization function by virtual function
        virtual void onInit(){
            ros::NodeHandle NodeHandle("~");
            NodeHandle = getPrivateNodeHandle();

            // init yolov8 object detector
            yolov8_detector_ros_ = new yolov8_ros::Yolov8ObjectDetector(NodeHandle);
        }
    yolov8_ros::Yolov8ObjectDetector *yolov8_detector_ros_;
};

// Declare as a Plug-in
PLUGINLIB_EXPORT_CLASS(Yolov8ObjectDetectorNodelet, nodelet::Nodelet);