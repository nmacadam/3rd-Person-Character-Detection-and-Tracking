#pragma once
#include <opencv2/core/mat.hpp>

class BackgroundSegmentor
{
public:
	BackgroundSegmentor(double minimumContourArea);

	void Update(cv::Mat& src, cv::Mat& dst);

private:
	double minimumContourArea;
};
