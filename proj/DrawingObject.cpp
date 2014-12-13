#include "DrawingObject.h"
#include <iostream>

DrawingObject::DrawingObject(std::vector<cv::Point2f>& points)
{
	addEdge(points[0],points[1], cv::Scalar(255,255,255));
	addEdge(points[1],points[2], cv::Scalar(255,255,255));
	addEdge(points[2],points[3], cv::Scalar(255,255,255));
	addEdge(points[3],points[0], cv::Scalar(0,0,255));

	addEdge(points[4],points[5], cv::Scalar(255,255,255));
	addEdge(points[5],points[6], cv::Scalar(255,255,255));
	addEdge(points[6],points[7], cv::Scalar(255,255,255));
	addEdge(points[7],points[4], cv::Scalar(0,0,255));

	addEdge(points[0],points[4], cv::Scalar(255,255,255));
	addEdge(points[1],points[5], cv::Scalar(255,255,255));
	addEdge(points[2],points[6], cv::Scalar(255,255,255));
	addEdge(points[3],points[7], cv::Scalar(0,0,255));
}


DrawingObject::~DrawingObject(void)
{
}


std::vector<cv::Point3f> DrawingObject::getCubePoints(){
	std::vector<cv::Point3f> input;

	input.push_back(cv::Point3f(0,0,0)); input.push_back(cv::Point3f(32,0,0)); input.push_back(cv::Point3f(32,32,0)); input.push_back(cv::Point3f(0,32,0));
	input.push_back(cv::Point3f(0,0,-32)); input.push_back(cv::Point3f(32,0,-32)); input.push_back(cv::Point3f(32,32,-32)); input.push_back(cv::Point3f(0,32,-32));

	return input;
}

void DrawingObject::addEdge(cv::Point2f& p1, cv::Point2f& p2, cv::Scalar color){
	PointPair* edge = new PointPair;
	edge->left = &p1;
	edge->right = &p2;
	edge->color = color;
	edges.push_back(edge);
}

void DrawingObject::addEdge(PointPair& edge){
	edges.push_back(&edge);
}

void DrawingObject::addEdges(std::vector<cv::Point2f> points, cv::Scalar color){
	for(int i=0; i < points.size()-1; i++){
		addEdge(points[i],points[i+1],color);
	}
}

void DrawingObject::draw(cv::Mat& img){
	for(int i=0; i<edges.size(); i++){
		cv::line(img, (*edges[i]->left), (*edges[i]->right), edges[i]->color);
	}
}