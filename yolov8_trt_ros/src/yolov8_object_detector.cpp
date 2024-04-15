/*
 * @Author: ruoxi521
 * @Date: 2023-03-31 11:33:34
 * @LastEditors: ruoxi521
 * @LastEditTime: 2024-04-11 23:57:01
 * @FilePath: /yolov8_ros/yolov8_trt_ros/src/yolov8_object_detector.cpp
 * @Description: 
 * 
 * Copyright (c) 2023 by suruoxi521@gmail.com, All Rights Reserved. 
 */

#include <detect/yolov8_object_detector.hpp>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "NvInferPlugin.h"

#include "utils/logging.h"
#include "utils/utils.h"

#include <opencv2/opencv.hpp>
#include <string>
#include <unordered_map>
#include <chrono>
#include <cublas_v2.h>

using namespace std;
using namespace nvinfer1;

// network and the input/output
static const int INPUT_H = 640;
static const int INPUT_W = 640;
static const int INPUT_C = 3;
static const int CLASSES = 80;
static const int Num_box = 8400;
static int OUTPUT_SIZE = Num_box * (CLASSES + 4);  // detect
static float NMS_THRESHOLD;
static float CONF_THRESHOLD;

char* trtModelStream{ nullptr };
size_t size{ 0 };

IRuntime* runtime;
ICudaEngine* engine;
IExecutionContext* context;
cudaStream_t stream;

int nbBindings, inputIndex, outputIndex0, outputIndex1;
void* buffers[2]; 

bool is_initialized = false;

const char* INPUT_BLOB_NAME = "images";
const char* OUTPUT_BLOB_NAME = "output0"; 				 //detect;
std::vector<std::string> labels;                         // labels 

int new_h, new_w, padh, padw;
float ratio_h, ratio_w;

static Logger gLogger;                    			     // log

struct OutputDet {
    int id;            // 类别 id
    float confidence;  // 置信度 
    cv::Rect box;      // 矩形框
};

// imageprpoprocess
int cvImageToTensor(const cv::Mat& image, float* tensor)
{
	cv::Mat src = image;
    if (src.empty()) 
	{
		std::cout << "imag load failed" << std::endl; 
		return 1;
	}
    int img_width = src.cols;
    int img_height = src.rows;
    std::cout << "宽高" << img_width << " " << img_height << std::endl;

	// static float data[3 * INPUT_H * INPUT_W];
    cv::Mat pre_img;

	// std::cout << "debug0" << std::endl;
	
    std::vector<int> padsize;           // resize
    pre_img = preprocess_img(src, INPUT_H, INPUT_W, padsize);
    // int new_h = padsize[0],  new_w = padsize[1],  padh = padsize[2],  padw = padsize[3];
    // float ratio_h = (float)src.rows / new_h;
    // float ratio_w = (float)src.cols / new_w;
    new_h = padsize[0];
    new_w = padsize[1];
    padh = padsize[2];
    padw = padsize[3];
    ratio_h = (float)src.rows / new_h;
    ratio_w = (float)src.cols / new_w;

    // [1, 3, INPUT_H, INPUT_W]
    int i = 0;
    for (int row = 0; row < INPUT_H; ++row)
    {
        uchar* uc_pixel = pre_img.data + row * pre_img.step;
        // pre_img.step = width * 3  每一行有 width 个 3 通道的值
        for (int col = 0; col < INPUT_W; ++col)
        {
            tensor[i] = (float)uc_pixel[2] /255.0;
            tensor[i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
            tensor[i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
            uc_pixel += 3;
            ++i;
        } 
    }
    
}


// Init Parameters
void setup(int image_size, int batch_size, std::string model_path,
		   std::string label_path, int classes_num, int num_box, 
		   float conf_threshold, float nms_threshold )
{
	OUTPUT_SIZE = num_box * (classes_num + 4);
    NMS_THRESHOLD = nms_threshold;
    CONF_THRESHOLD = conf_threshold;

	// load TensorRT model
	std::ifstream label_file(label_path.c_str());
	std::ifstream trt_file(model_path, std::ios::binary);
	if (!label_file.is_open()) {
		ROS_INFO("Label Not Found!!!");
    	is_initialized = false;
	}
	if (!trt_file.good()) {
		ROS_INFO("TensorRT Model Not Found!!!");
    	is_initialized = false;
	}
	else {
		// load labels
		labels = loadLabelsFromFile(label_path);
        
		// TRT buffer
        ROS_INFO("Begin loading Model...");
        trt_file.seekg(0, trt_file.end);            // 指向文件的最后地址
        size = trt_file.tellg();                // 把文件长度告诉给 size

        std::cout << "\nfile: " << model_path << std::endl;
        std::cout << "size is:  " << size  << std::endl;

        trt_file.seekg(0, trt_file.beg);            // 指向文件的最后地址
        trtModelStream = new char [size];   // 开辟一个 char 长度是文件的长度
        assert(trtModelStream);
        trt_file.read(trtModelStream, size);    // 将文件内容传给 trtModelStream
        trt_file.close(); 
    
		// Runtime
		ROS_INFO("Begin creating Runtime...");
		
		runtime = createInferRuntime(gLogger);
		assert(runtime != nullptr);

		bool didInitPlugins = initLibNvInferPlugins(&gLogger, "");

        // std::cout << "debug0" << std::endl;
		engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);

        // std::cout << "debug1" << std::endl;
		assert(engine != nullptr);

        // std::cout << "debug2" << std::endl;

		context = engine->createExecutionContext();

        // std::cout << "debug3" << std::endl;
		assert(context != nullptr);
		delete[] trtModelStream;

		CHECK(cudaStreamCreate(&stream));
		ROS_INFO("End loading Runtime...");

		// Input and output buffer pointers that we pass to the engine - the engine requires exactly
    	// IEngine::getNbBindings(),
    	// of these, but in this case we know that there is exactly 1 input and 2 output.

		nbBindings = engine->getNbBindings();

		inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
		outputIndex0 = engine->getBindingIndex(OUTPUT_BLOB_NAME);
	}

	const ICudaEngine& engine = context->getEngine();

    // Pointers to input and output device buffers to pass to engine
    assert(engine.getNbBindings() == 3);

    // know the name of the input and output for binding the buffers
    // const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    // const int outputIndex0 = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    // outputIndex0 = engine.getBindingIndex(OUTPUT_BLOB_NAME);

	is_initialized = true;
}

yolov8_msgs::BoundingBoxes infer(const sensor_msgs::ImageConstPtr& color_msgs, int batchsize)
{
	yolov8_msgs::BoundingBoxes bboxes;						  // 定义返回值

	// preprocessing
	cv::Mat image =cv_bridge::toCvShare(color_msgs, "rgb8")->image;
	cv::Size imsize = image.size();
	cv::resize(image, image, cv::Size(INPUT_W, INPUT_H));
	float input_data[INPUT_C * INPUT_H * INPUT_W];

	cvImageToTensor(image, input_data);

    auto start = std::chrono::system_clock::now();
    
	// Create GPU buffers on device
	CHECK(cudaMalloc(&buffers[inputIndex], batchsize * INPUT_C * INPUT_H * INPUT_W * sizeof(float)));
	CHECK(cudaMalloc(&buffers[outputIndex0], batchsize * OUTPUT_SIZE * sizeof(float)));

	// Create stream
	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));

	// DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
	CHECK(cudaMemcpyAsync(buffers[inputIndex], input_data, batchsize * INPUT_C * INPUT_H * INPUT_W * sizeof(float),
							cudaMemcpyHostToDevice, stream));

	// compute infer rate
	auto t_start = chrono::high_resolution_clock::now();
	// context->execute(1, &buffers[0]);
    context->enqueue(batchsize, buffers, stream, nullptr);
	auto t_end = chrono::high_resolution_clock::now();
	float total = chrono::duration<float, milli>(t_end - t_start).count();

    ROS_INFO("Finished Infer...");

	float prob[OUTPUT_SIZE];
	
	// Host memory for outputs.
	std::vector<int> classIds;     		// id array
	std::vector<float> confidences;  	// confidence relate with id 
	std::vector<cv::Rect> boxes;  		// id relate with boundingbox

	// back process
    // std::cout << "debug1" << std::endl;

	CHECK(cudaMemcpyAsync(prob, buffers[outputIndex0], batchsize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
	CHECK(cudaMemcpyAsync(prob, buffers[outputIndex1], sizeof(int), cudaMemcpyDeviceToHost, stream));
	cudaStreamSynchronize(stream);

    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex0]));
    
	int net_length = CLASSES + 4;
    cv::Mat out =  cv::Mat(net_length, Num_box, CV_32F, prob);
    
    // std::cout << "debug2" << std::endl;

    // 输出：1*net_length*Num_box;     
    // 所以每个 box 的属性是每隔 Num_box 取一个值，共 net_length 个值
    for (int i = 0; i < Num_box; i++)
    {   
        // std::cout << "debug-----0" << std::endl;
        // cv::Mat scores = out(Rect(i, 0, 1, CLASSES)).clone();
        cv::Mat scores = out(Rect(i, 4, 1, CLASSES)).clone();
        // std::cout << "debug-----1" << std::endl;
        Point classIdPoint;
        double max_class_score;
        // std::cout << "debug-----2" << std::endl;

        minMaxLoc(scores, 0, &max_class_score, 0, &classIdPoint);
        max_class_score = (float)max_class_score;
        
        if (max_class_score >= CONF_THRESHOLD) {
            float x = (out.at<float>(0, i) - padw) * ratio_w;  
            float y = (out.at<float>(1, i) - padh) * ratio_h;
            float w =  out.at<float>(2, i) * ratio_w;
            float h =  out.at<float>(3, i) * ratio_h;
            int left = MAX((x - 0.5 * w), 0);
            int  top = MAX((y - 0.5 * h), 0);
            int  width = (int)w;
            int height = (int)h;
            if (width <= 0 || height <= 0) {
                continue;
            }
            classIds.push_back(classIdPoint.y);
            confidences.push_back(max_class_score);
            boxes.push_back(Rect(left, top, width, height));
        }
    }
	
	// NMS 执行非最大抑制以消除具有较低置信度的冗余重叠框
    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD, nms_result);
    std::vector<OutputDet> output;

    // std::cout << "debug4" << std::endl;

    cv::Rect holeImgRect(0, 0, image.cols, image.rows);
    for (int i = 0; i < nms_result.size(); ++i)
    {
        int idx = nms_result[i];
        OutputDet result;
        yolov8_msgs::BoundingBox bbox;        // yolov8_msgs
        result.id = classIds[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx]& holeImgRect;
        output.push_back(result);

        bbox.id = result.id;
        bbox.probability = result.confidence;
        bbox.xmin = result.box.x;
        bbox.ymin = result.box.y;
        bbox.width = result.box.width;
        bbox.height = result.box.height;
        bbox.Class = labels[bbox.id].c_str();

        bboxes.bounding_boxes.push_back(bbox);
        bboxes.inference_time_ms = total;
    }
	ROS_INFO("Finish Back Process...");
	
    bboxes.header = color_msgs->header;

    auto end = std::chrono::system_clock::now();
    std::cout << "后处理时间：" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    
	return bboxes;  // 返回当前帧的检测结果
}

void destroy(void)
{
  if (is_initialized)
  {
    runtime->destroy();
    engine->destroy();
    context->destroy();
	
    // Release the stream and the buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex0]));
  }
  is_initialized = false;
}

void DrawPred_Det(cv::Mat& img, std::vector<OutputDet> result) {
    // 生成随机颜色
    std::vector<cv::Scalar> color;
    srand(time(0));

    // 为不同类别生成不同的对应颜色
    for (int i = 0; i < CLASSES; i++) {
        int b = rand() % 256;   
        int g = rand() % 256;
        int r = rand() % 256;
        color.push_back(cv::Scalar(b, g, r));
    }

    for (int i = 0; i < result.size(); i++ ) {
        int left,top;           // BoungingBox 的左边和上边
        int color_num = i;      
        left = result[i].box.x;
        top  = result[i].box.y;
        rectangle(img, result[i].box, color[result[i].id], 2, 8);

        char label[labels.size()];  // 标签

        // 打印标签和置信度
        sprintf(label, "%s:%.2f", labels[result[i].id].c_str(), result[i].confidence);

        int baseline;           // 起始位置
        cv::Size labelSize = getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        top = max(top, labelSize.height);
        putText(img, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 1, color[result[i].id], 2);
        // putText(img, coco80.at(int(*label)), Point(left, top), FONT_HERSHEY_SIMPLEX, 1, color[result[i].id], 2);
        std::cout << "idx: " << label << std::endl;
    }
}