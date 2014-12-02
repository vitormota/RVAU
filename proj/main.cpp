#include "MarkerDetection.h"
#include "BlobDetection.h"
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

#define cam 1

using namespace std;
using namespace cv;


int openWebcam(){
	VideoCapture cap; // open the video camera no. 0
	cap.open(cam);

	if (!cap.isOpened())  // if not success, exit program
	{
		cout << "Cannot open the video cam" << endl;
		return -1;
	}

	double dWidth = cap.get(CV_CAP_PROP_FRAME_WIDTH); //get the width of frames of the video
	double dHeight = cap.get(CV_CAP_PROP_FRAME_HEIGHT); //get the height of frames of the video

	cout << "Frame size : " << dWidth << " x " << dHeight << endl;

	namedWindow("MyVideo",CV_WINDOW_AUTOSIZE); //create a window called "MyVideo"

	while (1)
	{
		Mat frame;
		bool bSuccess = cap.read(frame); // read a new frame from video
		if (!bSuccess) //if not success, break loop
		{
			cout << "Cannot read a frame from video stream" << endl;
			break;
		}
		imshow("MyVideo", frame); //show the frame in "MyVideo" window

		/*Mat dst;
		binarizeImage(frame,dst);
		imshow("Binary", dst);*/
		vector<KeyPoint> keyPoints;
		findBlobs(frame, keyPoints);
		//imshow("TwoPass", twoPass(frame,CONNECTIVITY_8));

		if (waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
		{
			cout << "esc key is pressed by user" << endl;
			break; 
		}
	}
	return 0;
}

int main(int argc, char** argv){

	try{
		if(openWebcam()<0)
			return -1;
	}catch(Exception e){
		cout << "Ola";
	}

	return 0;
}