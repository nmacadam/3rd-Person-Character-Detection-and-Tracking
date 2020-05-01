#include "CharacterIdentifier.h"
#include <opencv2/highgui.hpp>
#include <opencv2/tracking/feature.hpp>
#include <iostream>

// Use manual ROI selection to target the character
void CharacterIdentifier::ManualFind(cv::Mat& frame)
{
	// Display bounding box. 
	bbox = cv::selectROI(frame, false);
	rectangle(frame, bbox, cv::Scalar(255, 0, 0), 2, 1);
}

// Automatically detect character
void CharacterIdentifier::Find(cv::Mat& frame, int initialWidth, int initialHeight, int verticalOffset)
{
	// Set initial location for character search
	cv::Point seed((frame.cols / 2) - (60 / 2), (frame.rows / 2) + verticalOffset);
	cv::Rect seedRect(seed.x, seed.y, initialWidth, initialHeight);

	// Grow the ROI until the character is located
	while (!EvaluateROI(frame, seedRect))
	{
		seedRect.width += 10;
		seedRect.height += 10;
	}
	
	bbox = seedRect;

	//system("pause");
	
    // Display bounding box. 
    rectangle(frame, bbox, cv::Scalar(255, 0, 0), 2, 1);
}

bool CharacterIdentifier::EvaluateROI(cv::Mat& frame, cv::Rect& roi)
{
	if (roi.width >= frame.cols || roi.height >= frame.rows)
	{
		std::cout << "ROI exceeded window size" << std::endl;
		return true;
	}
	
	cv::Mat frameROI(frame, roi);
	
	if (frameROI.empty())
	{
		std::cout << "ROI empty" << std::endl;
		return false;
	}

	// Apply canny edge detection
	cv::Mat blurSrc;
	blur(frameROI, blurSrc, cv::Size(3, 3));
	cv::Mat edges;
	cv::Canny(blurSrc, edges, 100, 300);

	// Track the change in edges between each growth
	int currentNonZeroCount = cv::countNonZero(edges);
	int countDelta = currentNonZeroCount - lastROICount;

	//std::cout << "| Current: " << currentNonZeroCount << "| Last: " << lastROICount << "| Delta: " << countDelta << std::endl;
	
	//imshow("Current ROI Edges", edges);
	
	lastROICount = currentNonZeroCount;

	if (countDelta < 15 || frameROI.cols > 100)
	{
		return true;
	}

	return false;
}
