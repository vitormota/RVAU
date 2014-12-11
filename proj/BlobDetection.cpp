#include "BlobDetection.h"
#include <set>
#include <iostream>

using namespace cv;
using namespace std;

set<int> getNeighbours(Mat data, Mat labels, int row, int column, int connectivity){

	set<int> neighbours;
	int currValue = (int)data.at<uchar>(row,column);

	if( row > 0){
		int topValue = (int)data.at<uchar>(row-1,column);
		if( currValue == topValue ){
			neighbours.insert(labels.at<uchar>(row-1,column));
		}
	}

	if( column > 0 ){
		int leftValue = (int)data.at<uchar>(row,column-1);
		if( currValue == leftValue ){
			neighbours.insert(labels.at<uchar>(row,column-1));
		}
	}

	if( connectivity == CONNECTIVITY_8 ){
		if( row > 0 && column > 0 ){
			int topLeftValue = (int)data.at<uchar>(row-1,column-1);
			if( currValue == topLeftValue )
				neighbours.insert(labels.at<uchar>(row-1,column-1));
		}

		if( row > 0 && column < data.size().width-1 ){
			int topRightValue = (int)data.at<uchar>(row-1,column+1);
			if( currValue == topRightValue )
				neighbours.insert(labels.at<uchar>(row-1,column+1));
		}
	}

	return neighbours;
}

int findLabel(vector<set<int>>linked,int label){
	for(int i=0; i < linked.size(); i++){
		if( linked[i].find(label) != linked[i].end() ){
			return i;
		}
	}
}

Mat twoPass(Mat data, int connectivity){
	
	vector<set<int>> linked;

	Mat labels = Mat::zeros(data.rows,data.cols,CV_8U);
	int nextLabel = 0;

	for( int row=0; row < data.rows; row++){
		for(int column=0; column < data.cols; column++){
			set<int> neighbours = getNeighbours(data,labels,row,column,connectivity);
			

			if( neighbours.empty() ){
				set<int> nextSet;
				nextSet.insert(nextLabel);
				linked.push_back(nextSet);
				labels.at<uchar>(row,column) = nextLabel;
				nextLabel++;
			}else{
				labels.at<uchar>(row,column) = *(neighbours.begin());
				for(set<int>::iterator it= neighbours.begin(); it!= neighbours.end(); it++){
					linked[*it].insert(neighbours.begin(), neighbours.end());
				}
			}
		}
	}

	for( int row=0; row < data.rows; row++){
		for(int column=0; column < data.cols; column++){
			if((int)labels.at<uchar>(row,column) != 0 ){
				labels.at<uchar>(row,column) = findLabel(linked,labels.at<uchar>(row,column));
			}
		}
	}

	return labels;
}