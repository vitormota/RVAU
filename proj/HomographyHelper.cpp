#include "HomographyHelper.h"
#include "DrawingObject.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <omp.h>

using namespace cv;
using namespace std;


vector<Mat> markers;
vector<Point2f> dstPoints;
vector<Point3f> markerPoints;
vector<vector<Point3f>> objectPoints;
const int MAX_ONES = MARKER_SIZE*MARKER_SIZE*ERROR_ALLOWED;
Point3f scale, translation;
double rotX = 0, rotY = 0, rotZ = 0;

void setScale(double x, double y, double z){
	scale.x = x;
	scale.y = y;
	scale.z = z;
}

void setTranslation(double x, double y, double z){
	translation.x = x;
	translation.y = y;
	translation.z = z;
}

void setRotation(double angleX, double angleY, double angleZ){
	rotX = angleX;
	rotY = angleY;
	rotZ = angleZ;
}

void initMarkerDatabase(){
	markers.push_back(imread("marca1.png", CV_LOAD_IMAGE_GRAYSCALE));
	markers.push_back(imread("marca2.png", CV_LOAD_IMAGE_GRAYSCALE));
	markers.push_back(imread("marca3.png", CV_LOAD_IMAGE_GRAYSCALE));
	markers.push_back(imread("marca4.png", CV_LOAD_IMAGE_GRAYSCALE));
	markers.push_back(imread("kanji1.png", CV_LOAD_IMAGE_GRAYSCALE));
	markers.push_back(imread("kanji2.png", CV_LOAD_IMAGE_GRAYSCALE));
	markers.push_back(imread("kanji3.png", CV_LOAD_IMAGE_GRAYSCALE));
	markers.push_back(imread("kanji4.png", CV_LOAD_IMAGE_GRAYSCALE));
	markers.push_back(imread("m1.png", CV_LOAD_IMAGE_GRAYSCALE));
	markers.push_back(imread("m2.png", CV_LOAD_IMAGE_GRAYSCALE));
	markers.push_back(imread("m3.png", CV_LOAD_IMAGE_GRAYSCALE));
	markers.push_back(imread("m4.png", CV_LOAD_IMAGE_GRAYSCALE));

	markerPoints.push_back(Point3f(0, 0, 0));
	markerPoints.push_back(Point3f(MARKER_SIZE, 0, 0));
	markerPoints.push_back(Point3f(MARKER_SIZE, MARKER_SIZE, 0));
	markerPoints.push_back(Point3f(0, MARKER_SIZE, 0));

	dstPoints.push_back(Point2f(0,0));
	dstPoints.push_back(Point2f(MARKER_SIZE,0));
	dstPoints.push_back(Point2f(MARKER_SIZE,MARKER_SIZE));
	dstPoints.push_back(Point2f(0,MARKER_SIZE));

	vector<Point3f> objectPoints1;
	objectPoints1.push_back(Point3f(0,0,0)); objectPoints1.push_back(Point3f(MARKER_SIZE,0,0)); 
	objectPoints1.push_back(Point3f(MARKER_SIZE,MARKER_SIZE,0)); objectPoints1.push_back(Point3f(0,MARKER_SIZE,0));
	objectPoints1.push_back(Point3f(0,0,-MARKER_SIZE)); objectPoints1.push_back(Point3f(MARKER_SIZE,0,-MARKER_SIZE)); 
	objectPoints1.push_back(Point3f(MARKER_SIZE,MARKER_SIZE,-MARKER_SIZE)); objectPoints1.push_back(Point3f(0,MARKER_SIZE,-MARKER_SIZE));

	vector<Point3f> objectPoints2;
	objectPoints2.push_back(Point3f(MARKER_SIZE,0,0)); objectPoints2.push_back(Point3f(MARKER_SIZE,MARKER_SIZE,0));
	objectPoints2.push_back(Point3f(0,MARKER_SIZE,0)); objectPoints2.push_back(Point3f(0,0,0)); 
	objectPoints2.push_back(Point3f(MARKER_SIZE,0,-MARKER_SIZE)); objectPoints2.push_back(Point3f(MARKER_SIZE,MARKER_SIZE,-MARKER_SIZE)); 
	objectPoints2.push_back(Point3f(0,MARKER_SIZE,-MARKER_SIZE)); objectPoints2.push_back(Point3f(0,0,-MARKER_SIZE));

	vector<Point3f> objectPoints3;
	objectPoints3.push_back(Point3f(MARKER_SIZE,MARKER_SIZE,0)); objectPoints3.push_back(Point3f(0,MARKER_SIZE,0)); 
	objectPoints3.push_back(Point3f(0,0,0)); objectPoints3.push_back(Point3f(MARKER_SIZE,0,0)); 
	objectPoints3.push_back(Point3f(MARKER_SIZE,MARKER_SIZE,-MARKER_SIZE)); objectPoints3.push_back(Point3f(0,MARKER_SIZE,-MARKER_SIZE)); 
	objectPoints3.push_back(Point3f(0,0,-MARKER_SIZE)); objectPoints3.push_back(Point3f(MARKER_SIZE,0,-MARKER_SIZE)); 

	vector<Point3f> objectPoints4;
	objectPoints4.push_back(Point3f(0,MARKER_SIZE,0)); objectPoints4.push_back(Point3f(0,0,0)); 
	objectPoints4.push_back(Point3f(MARKER_SIZE,0,0)); objectPoints4.push_back(Point3f(MARKER_SIZE,MARKER_SIZE,0)); 
	objectPoints4.push_back(Point3f(0,MARKER_SIZE,-MARKER_SIZE)); objectPoints4.push_back(Point3f(0,0,-MARKER_SIZE));
	objectPoints4.push_back(Point3f(MARKER_SIZE,0,-MARKER_SIZE)); objectPoints4.push_back(Point3f(MARKER_SIZE,MARKER_SIZE,-MARKER_SIZE)); 

	objectPoints.push_back(objectPoints1);
	objectPoints.push_back(objectPoints2);
	objectPoints.push_back(objectPoints3);
	objectPoints.push_back(objectPoints4);

	scale.x=1;
	scale.y=1;
	scale.z=1;

	translation.x = 0;
	translation.y = 0;
	translation.z = 0;
}

Point3f multiplyVectors(Point3f v1, Point3f v2){
	Point3f result;

	result.x = v1.x*v2.x;
	result.y = v1.y*v2.y;
	result.z = v1.z*v2.z;

	return result;
}

Point3f rotateX(Point3f v, double angle){
	Point3f result;

	result.x = v.x;
	result.y = v.y*cos(angle) - v.z*sin(angle);
	result.z = v.y*sin(angle) + v.z*cos(angle);

	return result;
}

Point3f rotateY(Point3f v, double angle){
	Point3f result;

	result.x = v.x*cos(angle) + v.z*sin(angle);
	result.y = v.y;
	result.z = v.z*cos(angle) - v.x*sin(angle);

	return result;
}

Point3f rotateZ(Point3f v, double angle){
	Point3f result;

	result.x = v.x*cos(angle) - v.y*sin(angle);
	result.y = v.x*sin(angle) + v.y*cos(angle);
	result.z = v.z;

	return result;
}

Point3f applyRotation(Point3f v, double rotX, double rotY, double rotZ){
	Point3f result;

	result = rotateX(v,rotX);
	result = rotateY(result,rotY);
	result = rotateZ(result,rotZ);

	return result;
}

vector<Point3f> applyTransformation(vector<Point3f> object){
	vector<Point3f> transformed;

	for(int i=0; i<object.size();i++){

		Point3f transformedPoint = applyRotation(object[i],rotX,rotY,rotZ);

		transformedPoint = multiplyVectors(transformedPoint,scale);

		transformedPoint.x = transformedPoint.x + translation.x;
		transformedPoint.y = transformedPoint.y + translation.y;
		transformedPoint.z = transformedPoint.z + translation.z;

		transformed.push_back(transformedPoint);
	}

	return transformed;
}

Scalar getColor(int markerIdx){
	int m = markerIdx / 4;

	switch(m){
	case 0:
		return Scalar(0,0,255);
	case 1:
		return Scalar(0,255,0);
	case 2:
		return Scalar(255,0,0);
	default:
		return Scalar(255,255,255);
	}
}

void matchPoints(vector<Point2f> points, Mat binImage, Mat &dst, const Mat K, const Mat distCoef){

	//Calculate homography between the image plane and the marker plane
	Mat homo = findHomography(points, dstPoints);

	//Transform the image to the marker plane: frontal view with size 32x32
	Mat warpHomo;
	warpPerspective(binImage, warpHomo, homo, Size(MARKER_SIZE, MARKER_SIZE),INTER_LINEAR);

	

	//Iterate through all elements of markers vector to compare with all markers in all orientations
	#pragma omp parallel for
	for (int m = 0; m < markers.size(); m++){

		Mat compare;
		int count;

		//Compare the image with the marker in the database resulting in a map of 0's and 1's
		//0's mean different pixels
		//1's mean similar pixels
		bitwise_xor(warpHomo, markers[m], compare);

		count = countNonZero(compare);

		//If it has a sufficient number of similar pixels
		if (count < MAX_ONES){

			//Finds the object pose from 3D-2D point correspondences [rvec|tvec] extrinsic parameters
			//markerPoints 3D Points in object coordinate space (marker plane)
			//points are the corresponding points in the image 2D space
			Mat rvec, tvec;
			solvePnP(markerPoints, points, K, distCoef, rvec, tvec);

			//Projects 3D points to an image plane
			//objectPoints are the 3D points of the object in the object coordinate space (marker plane)
			vector<Point2f> imgPoints;
			projectPoints(applyTransformation(objectPoints[m % 4]),rvec,tvec,K,distCoef,imgPoints);

			DrawingObject obj(imgPoints, getColor(m));
			obj.draw(dst);
		}
	}
}
