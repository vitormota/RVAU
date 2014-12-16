#pragma once
#include <opencv2/core/core.hpp>

struct PointPair{
	cv::Point2f* left;
	cv::Point2f* right;
	cv::Scalar color;
};

class DrawingObject
{
public:
	DrawingObject(std::vector<cv::Point2f>& points);
	~DrawingObject(void);
	void addEdge(cv::Point2f& p1, cv::Point2f& p2, cv::Scalar color);
	void addEdge(PointPair& edge);
	void addEdges(std::vector<cv::Point2f> points, cv::Scalar color);
	void draw(cv::Mat& img);

private:
	std::vector<PointPair*> edges;
};


