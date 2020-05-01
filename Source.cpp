#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include "CharacterTracker.h"
#include "BackgroundSegmentor.h"
#include "UIDetector.h"

using namespace cv;
using namespace std;

const int alpha_slider_max = 100;
int ui_alpha_slider;
int bg_alpha_slider;
double ui_alpha;
double bg_alpha;

static void on_ui_trackbar(int, void*)
{
    ui_alpha = (double)ui_alpha_slider / alpha_slider_max;
}

static void on_bg_trackbar(int, void*)
{
    bg_alpha = (double)bg_alpha_slider / alpha_slider_max;
}

int main(int argc, char** argv)
{
    // Read video
    VideoCapture video("Mario.mp4");

    // Exit if video is not opened
    if (!video.isOpened())
    {
        cout << "Could not read video file" << endl;
        return 1;
    }

    // Read first frame 
    Mat frame;
    Mat scoreImage;
	Mat uiBBox;
    Mat bgClassified;
    bool ok = video.read(frame);

    namedWindow("Tracking", WINDOW_AUTOSIZE);

    createTrackbar("UI BBOX", "Tracking", &ui_alpha_slider, alpha_slider_max, on_ui_trackbar);
    on_ui_trackbar(ui_alpha_slider, 0);
    createTrackbar("BACKGROUND", "Tracking", &bg_alpha_slider, alpha_slider_max, on_bg_trackbar);
    on_bg_trackbar(bg_alpha_slider, 0);

    UIDetector uiDetector(frame);
    CharacterTracker cTracker(frame);
    BackgroundSegmentor bgClassifier(1000);

    int frame_ctr = 0;
    while (video.read(frame))
    {
        frame_ctr++;
    	
        // Start timer
        double timer = (double)getTickCount();

        bgClassifier.Update(frame, bgClassified);

        uiDetector.Update(frame, scoreImage);
        uiDetector.ShowBoundingBox(uiBBox);
    	
        // Update the character tracking result
        bool ok = cTracker.Update(frame);

        // Calculate Frames per second (FPS)
        float fps = getTickFrequency() / ((double)getTickCount() - timer);

        // Handle slider values
        Mat modifiedFrame;
        add(frame, uiBBox * ui_alpha, modifiedFrame);
        add(modifiedFrame, bgClassified * bg_alpha, modifiedFrame);

        // Display FPS on frame
        putText(frame, "FPS : " + std::to_string(fps), Point(100, 50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50, 170, 50), 2);
    	
        // Display frame.
        imshow("Original", frame);

        cv::resize(modifiedFrame, modifiedFrame, modifiedFrame.size() * 2);
        resizeWindow("Tracking", modifiedFrame.size());
        imshow("Tracking", modifiedFrame);
        imshow("UI Detection", uiBBox);
        imshow("Classified", bgClassified);

        // Exit if ESC pressed.
        int k = waitKey(1);
        if (k == 27)
        {
            break;
        }

    }
	
    destroyAllWindows();
    return 0;
}