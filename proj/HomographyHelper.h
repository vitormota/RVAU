#include <opencv2/core/core.hpp>

void cameraPoseFromHomography(const cv::Mat& H, cv::Mat& pose);
void initMarkerDatabase();
void matchPoints(std::vector<cv::Point2f> points, cv::Mat colorImage, int index, cv::Mat dst);