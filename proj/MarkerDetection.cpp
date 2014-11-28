#include "MarkerDetection.h"
#include <iostream>
#include <cfloat>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

SimpleBlobDetector::Params getBlobDetectorParams();

void binarizeImage(Mat img, Mat &dst){
	Mat img_gray = img;

	if( img.channels() == 3 )
		cvtColor(img, img_gray, CV_RGB2GRAY);
	else if( img.channels() != 2 )
		throw "Unknow image format. Must be color image or grayscale";

	threshold(img_gray, dst, THRESHOLD_VALUE, THRESHOLD_MAX_VALUE, THRESH_BINARY);
}

void findBlobs(Mat img, vector<KeyPoint> &keyPoints){
	Mat out, binary, mask;

	binarizeImage(img, binary);

	SimpleBlobDetector detector(getBlobDetectorParams());

	detector.detect( binary, keyPoints );
	drawKeypoints( img, keyPoints, out, CV_RGB(0,255,0), DrawMatchesFlags::DEFAULT);
	
	
	for( int i=0; i < keyPoints.size(); i++){
		KeyPoint keyPoint = keyPoints.at(i);
	
		mask = Mat::zeros(binary.rows+2, binary.cols+2,CV_8U);	

		for( int j=0; j < 10; j++){
			int x = keyPoint.pt.x + (rand() % (int)keyPoint.size * PERCENTAGE_SIZE) - (keyPoint.size * PERCENTAGE_SIZE / 2 );
			int y = keyPoint.pt.y + (rand() % (int)keyPoint.size * PERCENTAGE_SIZE) - (keyPoint.size * PERCENTAGE_SIZE / 2 );
			floodFill(binary, mask, Point(x,y), 255, 0, Scalar(), Scalar(), 4+(255<<8)+FLOODFILL_MASK_ONLY);	
		}
		findBlobsContours(mask);
	}
}

void findBlobsContours(cv::Mat img){
	vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

	Mat dst = Mat::zeros(img.rows, img.cols, CV_8UC3);

    findContours( img, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE );

	// iterate through all the top-level contours,
    // draw each connected component with its own random color
    int idx = 0;
	for( idx =0 ; idx < contours.size(); idx++ )
    {
        Scalar color( rand()&255, rand()&255, rand()&255 );
        drawContours( dst, contours, idx, color, 1, 8, hierarchy );
    }
}

SimpleBlobDetector::Params getBlobDetectorParams(){
	SimpleBlobDetector::Params params;
	params.filterByCircularity=false;
	params.filterByConvexity=false;
	params.filterByInertia=false;

	params.filterByColor=true;
	params.blobColor=255;

	params.filterByArea=true;
	params.minArea=50;
	params.maxArea=FLT_MAX;

	params.minThreshold=THRESHOLD_VALUE-THRESHOLD_VARIATION;
	params.maxThreshold=THRESHOLD_VALUE+THRESHOLD_VARIATION;
	params.thresholdStep=THREASHOLD_STEP;

	params.minDistBetweenBlobs=50;
	return params;
}
