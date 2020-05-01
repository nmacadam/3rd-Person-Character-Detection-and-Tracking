#include "UIDetector.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

UIDetector::UIDetector(cv::Mat frame)
{
	this->frame = frame;
	this->lastFrame = cv::Mat::zeros(frame.rows, frame.cols, frame.type());
	this->score = cv::Mat::ones(frame.rows, frame.cols, frame.type()) * 255;
	cvtColor(this->score, this->score, cv::COLOR_BGR2GRAY);
	this->sscore = cv::Mat::ones(frame.rows, frame.cols, frame.type()) * 255;
	cvtColor(this->sscore, this->sscore, cv::COLOR_BGR2GRAY);

	this->bufferDifference = cv::Mat::ones(frame.rows, frame.cols, frame.type()) * 255;
	cvtColor(this->bufferDifference, this->bufferDifference, cv::COLOR_BGR2GRAY);
}

void UIDetector::Update(cv::Mat& currentFrame, cv::Mat& dst)
{
	lastFrame = frame.clone();
	frame = currentFrame.clone();
	//imshow("current frame", currentFrame);
	//imshow("last frame", lastFrame);

	cv::Mat difference;
	subtract(currentFrame, lastFrame, difference);
	//imshow("raw difference", difference);

	cv::Mat bwDiff;
	cvtColor(difference, bwDiff, cv::COLOR_BGR2GRAY);
	//imshow("bw difference", bwDiff);

	cv::Mat invDifference;
	bitwise_not(bwDiff, invDifference);

	cv::Mat frameScore;
	cv::Mat toBuffer;
	threshold(invDifference, frameScore, 225, 255, cv::THRESH_BINARY);
	//imshow("frame difference score", frameScore);

	cv::multiply(score, frameScore, score);

	this->frameScore = frameScore;

	//cv::imshow("final score", score);

	dst = score;
}

void UIDetector::ShowBoundingBox(cv::Mat& dst)
{
	cv::Mat contour = score.clone();
	cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(10, 10));

	cv::dilate(contour, contour, element, cv::Point(-1, -1), 1);

	//cv::imshow("Dilation", contour);

	cv::RNG rng(12345);

	std::vector<std::vector<cv::Point> > contours;
	findContours(contour, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

	std::vector<std::vector<cv::Point> > contours_poly(contours.size());
	std::vector<cv::Rect> boundRect(contours.size());
	for (size_t i = 0; i < contours.size(); i++)
	{
		double area = contourArea(contours[i]);
		if (area > 25)
		{
			float epsilon = 0.1 * cv::arcLength(contours[i], true);
			approxPolyDP(contours[i], contours_poly[i], epsilon, true);
			boundRect[i] = cv::boundingRect(contours_poly[i]);
		}
	}

	cv::Mat drawing = cv::Mat::zeros(score.size(), CV_8UC3);

	for (size_t i = 0; i < contours.size(); i++)
	{
		cv::Scalar color = cv::Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
		rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), color, 2);
	}

	dst = drawing;
}
