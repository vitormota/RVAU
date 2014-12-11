#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#define CONNECTIVITY_8 1
#define CONNECTIVITY_4 2

cv::Mat twoPass(cv::Mat data, int connectivity);