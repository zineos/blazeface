#include <stdio.h>
#include <algorithm>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>

#include "FaceDetector.h"

using namespace std;

int main(int argc, char** argv)
{

    string imgPath;
    if  (argc = 1)
    {
        imgPath = "../sample.jpg";
    }
    else if (argc = 2)
    {
        imgPath = argv[1];
    }
    string param = "../model/blaceface.param";
    string bin = "../model/blaceface.bin";
    const int max_side = 320;

    // slim or RFB
    Detector detector(param, bin, max_side, false);
    // retinaface
    // Detector detector(param, bin, true);
    Timer timer;
    for	(int i = 0; i < 100; i++){


        cv::Mat img = cv::imread(imgPath.c_str());

        // scale
        float scale = 1.f;

        // float long_side = std::max(img.cols, img.rows);
        // float scale = max_side/long_side;
        
        // cv::Mat padded;
        cv::Mat padded = img;
        // int padding = 3;
        // padded.create(long_side+ 2*padding, long_side+ 2*padding, img.type());
        // padded.setTo(cv::Scalar::all(0));

        // img.copyTo(padded(cv::Rect(padding, padding, img.cols, img.rows)));

        // cv::Mat img_scale;
        // cv::Size size = cv::Size(padded.cols*scale, padded.rows*scale);
        // cv::resize(padded, img_scale, cv::Size(padded.cols*scale, padded.rows*scale));

        if (img.empty())
        {
            fprintf(stderr, "cv::imread %s failed\n", imgPath.c_str());
            return -1;
        }
        std::vector<bbox> boxes;

        timer.tic();

        detector.Detect(padded, boxes);
        timer.toc("----total timer:");

        // draw image
        for (int j = 0; j < boxes.size(); ++j) {
            cv::Rect rect(boxes[j].x1/scale, boxes[j].y1/scale, boxes[j].x2/scale - boxes[j].x1/scale, boxes[j].y2/scale - boxes[j].y1/scale);
            cv::rectangle(img, rect, cv::Scalar(0, 0, 255), 1, 8, 0);
            char test[80];
            sprintf(test, "%f", boxes[j].s);

            cv::putText(img, test, cv::Size((boxes[j].x1/scale), boxes[j].y1/scale), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 255, 255));
            cv::circle(img, cv::Point(boxes[j].point[0]._x / scale, boxes[j].point[0]._y / scale), 1, cv::Scalar(0, 0, 225), 4);
            cv::circle(img, cv::Point(boxes[j].point[1]._x / scale, boxes[j].point[1]._y / scale), 1, cv::Scalar(0, 255, 225), 4);
            cv::circle(img, cv::Point(boxes[j].point[2]._x / scale, boxes[j].point[2]._y / scale), 1, cv::Scalar(255, 0, 225), 4);
            cv::circle(img, cv::Point(boxes[j].point[3]._x / scale, boxes[j].point[3]._y / scale), 1, cv::Scalar(0, 255, 0), 4);
            cv::circle(img, cv::Point(boxes[j].point[4]._x / scale, boxes[j].point[4]._y / scale), 1, cv::Scalar(255, 0, 0), 4);
        }
        cv::imwrite("test.png", img);
    }
    return 0;
}

