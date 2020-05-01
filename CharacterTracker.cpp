#include "CharacterTracker.h"

CharacterTracker::CharacterTracker(cv::Mat& frame)
{
    tracker = cv::TrackerCSRT::create();
    identifier.Find(frame, 40, 40, -25);
    tracker->init(frame, identifier.bbox);
}

bool CharacterTracker::Update(cv::Mat& frame)
{
    // Update the tracking result
    cv::Point seed((frame.cols / 2) - (60 / 2), (frame.rows / 2) + (-25));
    double distance = sqrt(((seed.x - identifier.bbox.x) * (seed.x - identifier.bbox.x)) + ((seed.y - identifier.bbox.y) * (seed.y - identifier.bbox.y)));
    bool ok = tracker->update(frame, identifier.bbox) && distance < 100 && identifier.bbox.width < 150;

	if (ok)
	{
        DrawBBox(frame);
	}
    else
    {
        putText(frame, "Tracking failure detected", cv::Point(100, 80), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 255), 2);
        identifier.Find(frame, 40, 40, -25);
    	
        tracker->clear();
        tracker = cv::TrackerCSRT::create();
        tracker->init(frame, identifier.bbox);
    }

    return ok;
}

void CharacterTracker::DrawBBox(cv::Mat& frame)
{
    // Tracking success : Draw the tracked object
    rectangle(frame, identifier.bbox, cv::Scalar(255, 0, 0), 2, 1);
}
