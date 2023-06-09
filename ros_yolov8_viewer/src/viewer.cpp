#include "ros_yolov8_viewer/viewer.h"

ViewerNode::ViewerNode(const ros::NodeHandle& nh)
  : nh_(nh)
  , it_(nh)
  , params()
  , image_sub(it_, params.image_topic, 1, image_transport::TransportHints("compressed"))
  , boxes_sub(nh_, params.boxes_topic, 1)
  , sync(image_sub, boxes_sub, 60)
{
  sync.registerCallback(boost::bind(&ViewerNode::syncCallback, this, _1, _2));
}

// TODO
void load_label(std::vector<std::string>& labels, std::ifstream label_file)
{ 
	if (!label_file.is_open()) {
		ROS_INFO("Label Not Found!!!");
	}

  std::string line;
  while (getline(label_file, line));
  {
    labels.push_back(line);
  }

  std::cout << "Labels: " << labels.size() << std::endl;
}

void ViewerNode::syncCallback(const sensor_msgs::ImageConstPtr& image_msg,
                              const yolov8_msgs::BoundingBoxesConstPtr& boxes_msg)
{
  // 生成随机颜色
  std::vector<cv::Scalar> color;

  srand(time(0));
  ros::param::get("class_num", class_num);

  const std::vector<std::string> labels = {
      "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
      "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
      "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
      "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
      "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
      "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
      "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
      "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
      "hair drier", "toothbrush"
  };

  // 为不同类别生成不同的对应颜色
  for (int i = 0; i < class_num; i++) {
    int b = rand() % 256;   
    int g = rand() % 256;
    int r = rand() % 256;
    color.push_back(cv::Scalar(b, g, r));
  }

  cv::Mat image = cv_bridge::toCvShare(image_msg, "bgr8")->image;
  for (auto obj : boxes_msg->bounding_boxes)
  {
    obj.Class = labels[obj.id].c_str();
    std::stringstream ss;
    ss << obj.Class << ':' << obj.probability;
    // cv::rectangle(image, cvPoint(obj.xmin, obj.ymin),
    //               cvPoint(obj.xmin + obj.width, obj.ymin + obj.height),
    //               cv::Scalar(255, 242, 35));
    // cv::putText(image, ss.str(), cvPoint(obj.xmin, obj.ymin + 20), cv::FONT_HERSHEY_PLAIN, 1.0f,
    //             cv::Scalar(0, 255, 255));
    cv::rectangle(image, cvPoint(obj.xmin, obj.ymin),
                  cvPoint(obj.xmin + obj.width, obj.ymin + obj.height),
                  color[obj.id], 2, 8);
    cv::putText(image, ss.str(), cvPoint(obj.xmin, obj.ymin + 20), cv::FONT_HERSHEY_SIMPLEX, 1.0f,
                color[obj.id], 2);
    std::cout << "result: " << ss.str() << std::endl;
  }

  cv::imshow("yolov8_viewer", image);
  cv::waitKey(30);
}

int main(int argc, char* argv[])
{
  ros::init(argc, argv, "yolov8_viewer");

  ros::NodeHandle nh("~");
  ViewerNode viewer(nh);

  // ros::param::get("label_path", label_path);
  // std::ifstream label_file(label_path.c_str());
  // load_label(labels, label_file);

  ros::spin();

  return 0;
}