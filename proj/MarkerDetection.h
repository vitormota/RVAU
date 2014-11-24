#include <opencv2/imgproc/imgproc.hpp>

#define THRESHOLD_VALUE 127
#define THRESHOLD_MAX_VALUE 255

void binarizeImage(cv::Mat img, cv::Mat &dst);