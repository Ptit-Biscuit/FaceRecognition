#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include "opencv2/face.hpp"

#include <iostream>
#include <fstream>
#include <sstream>

#include <unistd.h>
#include <time.h>

using namespace cv;

void detectAndDisplay(Mat *frame);

static void read_csv(const string &filename, std::vector <Mat> &images, std::vector<int> &labels, char separator = ';') {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input, please check filename";
    }
    string line, imgPath, lblClass;
    while (getline(file, line)) {
        stringstream lines(line);
        getline(lines, imgPath, separator);
        getline(lines, lblClass);
        if (!imgPath.empty() && !lblClass.empty()) {
            images.push_back(imread(imgPath, 0));
            labels.push_back(atoi(lblClass.c_str()));
        }
    }
}

CascadeClassifier face_cascade;
Ptr <face::FaceRecognizer> model;

int main(int argc, const char **argv) {
    CommandLineParser parser(argc, argv,
                             "{face_cascade|./haarcascade_frontalface_default.xml|Path to face cascade.}"
                             "{csv_file|./data.csv|Path to csv file.}");
    parser.printMessage();

    String face_cascade_name = parser.get<String>("face_cascade");
    if (!face_cascade.load(face_cascade_name)) {
        std::cout << "Error loading face cascade" << std::endl;
        return -1;
    };

    String csv_file_name = parser.get<String>("csv_file");
    std::vector<Mat> images;
    std::vector<int> labels;
    try {
        read_csv(csv_file_name, images, labels);
    } catch (cv::Exception &e) {
        std::cerr << "Error opening file \"" << csv_file_name << "\". Reason: " << e.msg << std::endl;
        return -1;
    }

    model = face::LBPHFaceRecognizer::create();
    model->train(images, labels);

    VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cout << "Error opening video capture" << std::endl;
        return -1;
    }

    Mat frame;
    int count = 0;
    time_t start, end;
    time(&start);

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            cout << "No frame captured ..." << std::endl;
            break;
        }

        // Frame rate
        if (++count == 300) {
            time(&end);
            std::cout << "\033[034;1mFps : " << count / difftime(end, *start) << "\033[0m" << std::endl;
            sleepcp(2000);
            time(&start);
            count = 0;
        }

        detectAndDisplay(&frame, images[0].cols, images[0].rows);

        if (waitKey(10) == 27) {
            break;
        }
    }
    return 0;
}

void detectAndDisplay(Mat *frame, int imgWidth, int imgHeight) {
    Mat grayFrame, resized;
    cvtColor(*frame, grayFrame, COLOR_BGR2GRAY);
    equalizeHist(grayFrame, grayFrame);

    std::vector <Rect> faces;
    face_cascade.detectMultiScale(grayFrame, faces);

    for (size_t i = 0; i < faces.size(); i++) {
        resize(grayFrame(faces[i]), resized, Size(imgWidth, imgHeight), 1.0, 1.0, INTER_CUBIC);
        std::cout << "Prediction : " << model->predict(resized) << std::endl;
    }
}