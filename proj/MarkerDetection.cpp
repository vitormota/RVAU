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

void verifySeedPoint(int &x, int &y, Mat img){
	if( x < 0 )
		x=0;
	if( y < 0 )
		y=0;
	if( x >= img.cols )
		x=img.cols-1;
	if( y >= img.rows )
		y=img.rows-1;
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

			verifySeedPoint(x,y,binary);			

			floodFill(binary, mask, Point(x,y), 255, 0, Scalar(), Scalar(), 4+(255<<8)+FLOODFILL_MASK_ONLY);	
		}
		findBlobsContours(mask);
	}
}

double distanceToLine(Point begin, Point end, Point x){
	end-=begin;
	x-=begin;

	double area = x.cross(end);
	return area / sqrt(end.x*end.x + end.y * end.y );
}

void findBlobsContours(cv::Mat img){
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	Mat dst = Mat::zeros(img.rows, img.cols, CV_8UC3);

	findContours( img, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE );

	for( int i=0; i < contours.size(); i++ ){
		Scalar color( rand()&255, rand()&255, rand()&255 );
		drawContours( dst, contours, i, color, 1, 8, hierarchy );
		Point start =contours[i][0], end = contours[i][contours[i].size()/2];
		cout << "Starting Point: " << start << ", Ending Point: " << end << endl;
		line(dst,start,end,color,1);
		double max =0, min=0;
		Point maxPoint, minPoint;
		for(int j=1; j < contours[i].size(); j++ ){
			Point p = contours[i][j];
			double d = distanceToLine(start,end,p);
			if( d > max ){
				max = d;
				maxPoint = p;
			}else if( d < min ){
				min = d;
				minPoint = p;
			}
		}

		circle(dst,maxPoint,5,Scalar(0,0,255));
		circle(dst,minPoint,5,Scalar(255,0,0));
	}

	cout << "Contours size: " << contours.size() << endl;

	imshow("contours", dst);
	waitKey();
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
