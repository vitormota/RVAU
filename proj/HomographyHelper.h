#include <opencv2/core/core.hpp>

#define MARKER_SIZE 32
#define ERROR_ALLOWED 0.15

void cameraPoseFromHomography(const cv::Mat& H, cv::Mat& pose, const cv::Mat K);
void initMarkerDatabase();
void matchPoints(std::vector<cv::Point2f> points, cv::Mat binImage, cv::Mat &dst, const cv::Mat K, const cv::Mat distCoef);
void setScale(double x, double y, double z);
void setTranslation(double x, double y, double z);
void setRotation(double angleX, double angleY, double angleZ);