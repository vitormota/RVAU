#include "MarkerDetection.h"
#include <iostream>
#include <cfloat>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

SimpleBlobDetector::Params getBlobDetectorParams();

#define PI 3.14159265

//5 degrees
const double errorDegree = 10 * PI / 180;

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
	imshow("keypoints", out);

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

void calculateFarthestPoints(Point begin, Point end, vector<Point> points, Point &maxPoint, Point &minPoint){
	double max=0, min=0;
	for(int i=0; i < points.size(); i++){
		Point p = points[i];
		double d = distanceToLine(begin,end,p);
		if( d > max ){
			max = d;
			maxPoint = p;
		}else if( d < min ){
			min = d;
			minPoint = p;
		}
	}
}

double calculateDerivative(Point begin, Point end){
	Point rect = end-begin;
	if( rect.x == 0 )
		return 99999999;
	else
		return (double)rect.y / (double) rect.x;
}

void getSize(Point v1, Point v2, Point v3, Point v4, int &height, int &width){
	int maxDx =0, maxDy=0;
	maxDx=max(abs(v1.x-v2.x),maxDx);
	maxDy=max(abs(v1.y-v2.y),maxDy);

	maxDx=max(abs(v1.x-v3.x),maxDx);
	maxDy=max(abs(v1.y-v3.y),maxDy);

	maxDx=max(abs(v1.x-v4.x),maxDx);
	maxDy=max(abs(v1.y-v4.y),maxDy);

	maxDx=max(abs(v2.x-v3.x),maxDx);
	maxDy=max(abs(v2.y-v3.y),maxDy);

	maxDx=max(abs(v2.x-v4.x),maxDx);
	maxDy=max(abs(v2.y-v4.y),maxDy);

	maxDx=max(abs(v4.x-v3.x),maxDx);
	maxDy=max(abs(v4.y-v3.y),maxDy);

	height = maxDx;
	width = maxDy;
}

void findBlobsContours(cv::Mat img){
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	Mat dst = Mat::zeros(img.rows, img.cols, CV_8UC3);

	findContours( img, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE );

	for( int i=0; i < contours.size(); i++ ){
		//Scalar color( rand()&255, rand()&255, rand()&255 );
		//drawContours( dst, contours, i, color, 1, 8, hierarchy );

		Point start =contours[i][0], end = contours[i][contours[i].size()/2];

		Point vertice1, vertice2;
		calculateFarthestPoints(start,end,contours[i],vertice1,vertice2);

		Point vertice3,vertice4;
		calculateFarthestPoints(vertice1,vertice2,contours[i],vertice3,vertice4);

		//m(Rect1) +/- = m(Rect3)
		double mRect1 = calculateDerivative(vertice1,vertice3);
		double mRect3 = calculateDerivative(vertice2,vertice4);
		//m(Rect2) +/- = m(Rect4)
		double mRect2 = calculateDerivative(vertice1,vertice4);
		double mRect4 = calculateDerivative(vertice2,vertice3);

		int widthElement, heightElement;
		getSize(vertice1,vertice2,vertice3,vertice4,widthElement,heightElement);

		if( atan(mRect1) + errorDegree > atan(mRect3) && atan(mRect1) - errorDegree < atan(mRect3) &&
			atan(mRect2) + errorDegree > atan(mRect4) && atan(mRect2) - errorDegree < atan(mRect4) &&
			( widthElement < img.size().width * 0.9 || heightElement < img.size().height * 0.9)){

			line(dst,vertice1,vertice3,Scalar(255,0,0),1); //Rect1
			line(dst,vertice1,vertice4,Scalar(0,0,255),1); //Rect2
			line(dst,vertice2,vertice4,Scalar(255,0,0),1); //Rect3
			line(dst,vertice2,vertice3,Scalar(0,0,255),1); //Rect4

			imshow("points", dst);
		}
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
