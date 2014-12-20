#include "DrawingObject.h"
#include <iostream>

DrawingObject::DrawingObject(std::vector<cv::Point2f>& points, cv::Scalar color)
{
	cv::Scalar white = cv::Scalar(255,255,255);

	addEdge(points[0],points[1], color);
	addEdge(points[1],points[2], white);
	addEdge(points[2],points[3], white);
	addEdge(points[3],points[0], white);

	addEdge(points[4],points[5], color);
	addEdge(points[5],points[6], white);
	addEdge(points[6],points[7], white);
	addEdge(points[7],points[4], white);

	addEdge(points[0],points[4], color);
	addEdge(points[1],points[5], color);
	addEdge(points[2],points[6], white);
	addEdge(points[3],points[7], white);
}

DrawingObject::~DrawingObject(void)
{
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