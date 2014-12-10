#include "HomographyHelper.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

vector<Mat> markers;
vector<Point2f> dstPoints;
vector<vector<Point2f>> markerPoints;

void initMarkerDatabase(){
	markers.push_back(imread("marca1.png",CV_LOAD_IMAGE_GRAYSCALE));
	markers.push_back(imread("marca2.png",CV_LOAD_IMAGE_GRAYSCALE));
	markers.push_back(imread("marca3.png",CV_LOAD_IMAGE_GRAYSCALE));
	markers.push_back(imread("marca4.png",CV_LOAD_IMAGE_GRAYSCALE));

	vector<Point2f> marker1Points;
	marker1Points.push_back(Point2f(0,0));
	marker1Points.push_back(Point2f(32,0));
	marker1Points.push_back(Point2f(32,32));
	marker1Points.push_back(Point2f(0,32));

	vector<Point2f> marker2Points;
	marker2Points.push_back(Point2f(32,0));
	marker2Points.push_back(Point2f(32,32));
	marker2Points.push_back(Point2f(0,32));
	marker2Points.push_back(Point2f(0,0));

	vector<Point2f> marker3Points;
	marker3Points.push_back(Point2f(32,32));
	marker3Points.push_back(Point2f(0,32));
	marker3Points.push_back(Point2f(0,0));
	marker3Points.push_back(Point2f(32,0));

	vector<Point2f> marker4Points;
	marker4Points.push_back(Point2f(0,32));
	marker4Points.push_back(Point2f(0,0));
	marker4Points.push_back(Point2f(32,0));
	marker4Points.push_back(Point2f(32,32));

	markerPoints.push_back(marker1Points);
	markerPoints.push_back(marker2Points);
	markerPoints.push_back(marker3Points);
	markerPoints.push_back(marker4Points);

	dstPoints.push_back(Point2f(0,0));
	dstPoints.push_back(Point2f(32,0));
	dstPoints.push_back(Point2f(32,32));
	dstPoints.push_back(Point2f(0,32));
}

/*void cameraPoseFromHomography(const Mat& H, Mat& pose)
{
	pose = Mat::eye(3, 4, CV_32FC1); //3x4 matrix
	float norm1 = (float)norm(H.col(0)); 
	float norm2 = (float)norm(H.col(1));
	float tnorm = (norm1 + norm2) / 2.0f;

	Mat v1 = H.col(0);
	Mat v2 = pose.col(0);

	cv::normalize(v1, v2); // Normalize the rotation

	pose.col(0) = v2;

	v1 = H.col(1);
	v2 = pose.col(1);

	cv::normalize(v1, v2);

	v1 = pose.col(0);
	v2 = pose.col(1);

	Mat v3 = v1.cross(v2);  //Computes the cross-product of v1 and v2
	Mat c2 = pose.col(2);
	v3.copyTo(c2);      

	pose.col(3) = H.col(2) / tnorm; //vector t [R|t]
}*/

void cameraPoseFromHomography(const Mat& H, Mat& pose)
{
	pose = H.colRange(Range(0,2));

	Mat v1 = H.col(0);
	Mat v2 = H.col(1);
	Mat v3 = v1.cross(v2);

	cout << norm(v1) << endl << norm(v2) << endl << endl;

	normalize(v3,v3);

	hconcat(pose,v3,pose);
	hconcat(pose,H.col(2),pose);

	normalize(pose,pose);
}

void matchPoints(vector<Point2f> points, Mat colorImage, int index, Mat dst){
	Mat homo = findHomography(points,dstPoints);
	Mat warpHomo;

	warpPerspective(colorImage,warpHomo,homo,Size(32,32));

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
		if( white >= 32*32*0.9 ){
			//Calcular nova homografia
			Mat inverseHomo = findHomography(markerPoints[index],points);
			Mat pose;
			cameraPoseFromHomography(inverseHomo,pose);

			vector<Point3f> input;

			input.push_back(Point3f(0,0,0)); input.push_back(Point3f(32,0,0)); input.push_back(Point3f(32,32,0)); input.push_back(Point3f(0,32,0));
			input.push_back(Point3f(0,0,0)); input.push_back(Point3f(0,0,32)); //input.push_back(Point3f(0,0,10));
			perspectiveTransform(input,points,pose);

			for( int p=0; p < points.size()-1; p++){
				line(dst,points[p],points[p+1],Scalar(255,255,255));
			}

			imshow("Homo", warpHomo);
			break;
		}

	}
	imshow("points", dst);
}
