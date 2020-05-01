#pragma once
#include <opencv2/core/mat.hpp>

class UIDetector
{
public:
	UIDetector(cv::Mat frame);

	void Update(cv::Mat& currentFrame, cv::Mat& dst);

	void ShowBoundingBox(cv::Mat& dst);

	cv::Mat& GetFrameScore() { return this->frameScore; }

private:
	int scoreRange;
	float similarityThreshold;
	int subScoreBufferSize = 20;
	int subScoreFrameInterval = 5;

	int storedFrames = 0;

	cv::Mat frame;
	cv::Mat lastFrame;
	cv::Mat score;
	cv::Mat bufferDifference;

	cv::Mat sscore;

	cv::Mat frameScore;

	std::vector<cv::Mat> subScores;
};
