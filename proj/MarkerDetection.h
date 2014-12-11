#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#define THRESHOLD_VALUE 127
#define THRESHOLD_MAX_VALUE 255
#define THRESHOLD_VARIATION 2
#define THREASHOLD_STEP 2
#define PERCENTAGE_SIZE 1/2
#define FLOODFILL_ITERATIONS 5

void binarizeImage(cv::Mat img, cv::Mat &dst);
void findBlobs(cv::Mat img, std::vector<cv::KeyPoint> &keyPoints);
void findBlobsContours(cv::Mat img, cv::Mat colorImage);
void initMarkerDatabase();