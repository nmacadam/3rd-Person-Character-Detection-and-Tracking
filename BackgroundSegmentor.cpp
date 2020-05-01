#include "BackgroundSegmentor.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

BackgroundSegmentor::BackgroundSegmentor(double minimumContourArea)
{
    this->minimumContourArea = minimumContourArea;
}

void BackgroundSegmentor::Update(cv::Mat& src, cv::Mat& dst)
{	
    // Sharpen filter kernel
    Mat kernel = (Mat_<float>(3, 3) <<
        1, 1, 1,
        1, -8, 1,
        1, 1, 1);
	
    Mat imgLaplacian;
    filter2D(src, imgLaplacian, CV_32F, kernel);
    Mat sharp;
    src.convertTo(sharp, CV_32F);
    Mat imgResult = sharp - imgLaplacian;
	
    // convert back to 8bits gray scale
    imgResult.convertTo(imgResult, CV_8UC3);
    imgLaplacian.convertTo(imgLaplacian, CV_8UC3);
    //imshow( "Laplace Filtered Image", imgLaplacian );
    //imshow("New Sharped Image", imgResult);
	
    // Create binary image from source image
    Mat bw;
    cvtColor(imgResult, bw, COLOR_BGR2GRAY);
    threshold(bw, bw, 40, 255, THRESH_BINARY | THRESH_OTSU);
    //imshow("Binary Image", bw);

	// Resample the binary image by blurring and re-thresholding
    Mat resampledBw;
    GaussianBlur(bw, resampledBw, Size(15, 15), 15);
    //imshow("Blurred Binary Image", resampledBw);
    threshold(resampledBw, resampledBw, 0, 255, THRESH_BINARY | THRESH_OTSU);
    //imshow("Resampled Binary Image", resampledBw);

	// Apply laplacian to get continuous edges
    Mat resampledLaplacian;
    filter2D(resampledBw, resampledLaplacian, CV_32F, kernel);
    resampledLaplacian.convertTo(resampledLaplacian, CV_8UC3);
    //imshow("Resampled Laplacian", resampledLaplacian);

	// Dilate edges
    int dilation_size = 3;
    Mat element = getStructuringElement(MORPH_ELLIPSE,
        Size(2 * dilation_size + 1, 2 * dilation_size + 1),
        Point(dilation_size, dilation_size));
    dilate(resampledLaplacian, resampledLaplacian, element);

	// Invert the edges
    Mat inverse;
    bitwise_not(resampledLaplacian, inverse);
	//imshow("Inversed Laplacian", inverse);

	// Get contours
    Mat inverse_8u;
    inverse.convertTo(inverse_8u, CV_8U);
    // Find total markers
    vector<vector<Point>> contours;
    findContours(inverse_8u, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	// Remove contours with too small an area
    vector<vector<Point>>::iterator c_itr;
    int itr = 0;
    int erased_ct = 0;
    for (c_itr = contours.begin(); c_itr != contours.end(); )
    {
    	double area = contourArea(*c_itr);
        if (area < minimumContourArea)
        {
            c_itr = contours.erase(c_itr);
            erased_ct++;
        }
        else
        {
            c_itr++;
            itr++;
        }
    }
    //std::cout << "Erased " << erased_ct << " contours\n";
	
    // Create the marker image for contours
    Mat markers = Mat::zeros(inverse.size(), CV_32SC3);
	
    // Draw the foreground markers
    for (size_t i = 0; i < contours.size(); i++)
    {
        Scalar color(rand() & 255, rand() & 255, rand() & 255);
        drawContours(markers, contours, static_cast<int>(i), color, FILLED);
    }
	Mat drawMarkers;
    markers.convertTo(dst, CV_8UC3);
    //imshow("Segmented Output", dst);
}