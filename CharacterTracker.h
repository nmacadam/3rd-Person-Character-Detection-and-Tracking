#pragma once
#include <opencv2/tracking/tracker.hpp>
#include "CharacterIdentifier.h"

class CharacterTracker
{
public:
	CharacterTracker(cv::Mat& frame);

	bool Update(cv::Mat& frame);
	void DrawBBox(cv::Mat& frame);

private:
	CharacterIdentifier identifier;
	cv::Ptr<cv::Tracker> tracker;
};
