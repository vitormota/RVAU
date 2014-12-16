#include "HomographyHelper.h"
#include "DrawingObject.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

vector<Mat> markers;
vector<Point2f> dstPoints;
vector<vector<Point3f>> markerPoints;
vector<Point3f> objectPoints;

void initMarkerDatabase(){
	markers.push_back(imread("marca1.png",CV_LOAD_IMAGE_GRAYSCALE));
	markers.push_back(imread("marca2.png",CV_LOAD_IMAGE_GRAYSCALE));
	markers.push_back(imread("marca3.png",CV_LOAD_IMAGE_GRAYSCALE));
	markers.push_back(imread("marca4.png",CV_LOAD_IMAGE_GRAYSCALE));

	vector<Point2f> marker1Points;
	marker1Points.push_back(Point2f(0,0));
	marker1Points.push_back(Point2f(MARKER_SIZE,0));
	marker1Points.push_back(Point2f(MARKER_SIZE,MARKER_SIZE));
	marker1Points.push_back(Point2f(0,MARKER_SIZE));

	vector<Point2f> marker2Points;
	marker2Points.push_back(Point2f(MARKER_SIZE,0));
	marker2Points.push_back(Point2f(MARKER_SIZE,MARKER_SIZE));
	marker2Points.push_back(Point2f(0,MARKER_SIZE));
	marker2Points.push_back(Point2f(0,0));

	vector<Point2f> marker3Points;
	marker3Points.push_back(Point2f(MARKER_SIZE,MARKER_SIZE));
	marker3Points.push_back(Point2f(0,MARKER_SIZE));
	marker3Points.push_back(Point2f(0,0));
	marker3Points.push_back(Point2f(MARKER_SIZE,0));

	vector<Point2f> marker4Points;
	marker4Points.push_back(Point2f(0,MARKER_SIZE));
	marker4Points.push_back(Point2f(0,0));
	marker4Points.push_back(Point2f(MARKER_SIZE,0));
	marker4Points.push_back(Point2f(MARKER_SIZE,MARKER_SIZE));

	vector<Point3f> marker1Points3;
	marker1Points3.push_back(Point3f(0,0,0));
	marker1Points3.push_back(Point3f(MARKER_SIZE,0,0));
	marker1Points3.push_back(Point3f(MARKER_SIZE,MARKER_SIZE,0));
	marker1Points3.push_back(Point3f(0,MARKER_SIZE,0));

	vector<Point3f> marker2Points3;
	marker2Points3.push_back(Point3f(MARKER_SIZE,0,0));
	marker2Points3.push_back(Point3f(MARKER_SIZE,MARKER_SIZE,0));
	marker2Points3.push_back(Point3f(0,MARKER_SIZE,0));
	marker2Points3.push_back(Point3f(0,0,0));

	vector<Point3f> marker3Points3;
	marker3Points3.push_back(Point3f(MARKER_SIZE,MARKER_SIZE,0));
	marker3Points3.push_back(Point3f(0,MARKER_SIZE,0));
	marker3Points3.push_back(Point3f(0,0,0));
	marker3Points3.push_back(Point3f(MARKER_SIZE,0,0));

	vector<Point3f> marker4Points3;
	marker4Points3.push_back(Point3f(0,MARKER_SIZE,0));
	marker4Points3.push_back(Point3f(0,0,0));
	marker4Points3.push_back(Point3f(MARKER_SIZE,0,0));
	marker4Points3.push_back(Point3f(MARKER_SIZE,MARKER_SIZE,0));

	markerPoints.push_back(marker1Points3);
	markerPoints.push_back(marker2Points3);
	markerPoints.push_back(marker3Points3);
	markerPoints.push_back(marker4Points3);

	dstPoints.push_back(Point2f(0,0));
	dstPoints.push_back(Point2f(MARKER_SIZE,0));
	dstPoints.push_back(Point2f(MARKER_SIZE,MARKER_SIZE));
	dstPoints.push_back(Point2f(0,MARKER_SIZE));

	objectPoints.push_back(Point3f(0,0,0)); objectPoints.push_back(Point3f(MARKER_SIZE,0,0)); objectPoints.push_back(Point3f(MARKER_SIZE,MARKER_SIZE,0)); objectPoints.push_back(Point3f(0,MARKER_SIZE,0));
	objectPoints.push_back(Point3f(0,0,-MARKER_SIZE)); objectPoints.push_back(Point3f(MARKER_SIZE,0,-MARKER_SIZE)); objectPoints.push_back(Point3f(MARKER_SIZE,MARKER_SIZE,-MARKER_SIZE)); objectPoints.push_back(Point3f(0,MARKER_SIZE,-MARKER_SIZE));
}

void matchPoints(vector<Point2f> points, Mat colorImage, Mat dst, const Mat K, const Mat distCoef){
	Mat homo = findHomography(points,dstPoints);
	Mat warpHomo;

	warpPerspective(colorImage,warpHomo,homo,Size(MARKER_SIZE,MARKER_SIZE));

	for(int m=0; m < markers.size();m++){
		Mat compare;
		bitwise_xor(warpHomo,markers[m],compare);
		int white=0;
		for(int y=0; y < compare.size().height; y++){
			for(int x=0; x < compare.size().width; x++){
				if( compare.at<uchar>(y,x) == 0 ){
					white++;
				}
			}
		}

		//90% dos pixels iguais
		if( white >= MARKER_SIZE*MARKER_SIZE*ERROR_ALLOWED ){
			Mat rvec, tvec;
			solvePnP(markerPoints[m], points, K, distCoef,rvec,tvec);

			

			vector<Point2f> imgPoints;
			projectPoints(objectPoints,rvec,tvec,K,distCoef,imgPoints);

			DrawingObject obj(imgPoints);
			obj.draw(dst);

			imshow("Homo", warpHomo);
			break;
		}

	}
	imshow("points", dst);
}
