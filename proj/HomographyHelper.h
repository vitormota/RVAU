#include <opencv2/core/core.hpp>

#define MARKER_SIZE 32
#define ERROR_ALLOWED 0.9

void cameraPoseFromHomography(const cv::Mat& H, cv::Mat& pose, const cv::Mat K);
void initMarkerDatabase();
void matchPoints(std::vector<cv::Point2f> points, cv::Mat colorImage, cv::Mat dst, const cv::Mat K, const cv::Mat distCoef);