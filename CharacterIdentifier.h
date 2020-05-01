#pragma once
#include <opencv2/core/async.hpp>

class CharacterIdentifier
{
public:
	void ManualFind(cv::Mat& frame);
	void Find(cv::Mat& frame, int initialWidth, int initialHeight, int verticalOffset);

	cv::Rect2d bbox;
private:
	int lastROICount = -1;
	bool EvaluateROI(cv::Mat& frame, cv::Rect& roi);
};

