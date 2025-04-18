// Preprocess.cpp

#include "Preprocess.h"

///////////////////////////////////////////////////////////////////////////////////////////////////
void preprocess(cv::Mat& imgOriginal, cv::Mat& imgGrayscale, cv::Mat& imgThresh) {
    imgGrayscale = extractValue(imgOriginal);                           // extract value channel only from original image to get imgGrayscale

    cv::Mat imgMaxContrastGrayscale = maximizeContrast(imgGrayscale);       // maximize contrast with top hat and black hat

    cv::Mat imgBlurred;

    cv::GaussianBlur(imgMaxContrastGrayscale, imgBlurred, GAUSSIAN_SMOOTH_FILTER_SIZE, 0);          // gaussian blur

                // call adaptive threshold to get imgThresh
    cv::adaptiveThreshold(imgBlurred, imgThresh, 255.0, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 19, 9);

}

///////////////////////////////////////////////////////////////////////////////////////////////////
cv::Mat extractValue(cv::Mat& imgOriginal) {
    cv::Mat imgHSV;
    std::vector<cv::Mat> vectorOfHSVImages;
    cv::Mat imgValue;

    cv::cvtColor(imgOriginal, imgHSV, cv::COLOR_BGR2HSV);

    cv::split(imgHSV, vectorOfHSVImages);

    imgValue = vectorOfHSVImages[2];

    return(imgValue);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
cv::Mat maximizeContrast(cv::Mat& imgGrayscale) {
    cv::Mat imgTopHat;
    cv::Mat imgBlackHat;
    cv::Mat imgGrayscalePlusTopHat;
    cv::Mat imgGrayscalePlusTopHatMinusBlackHat;

    cv::Mat structuringElement = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

    cv::morphologyEx(imgGrayscale, imgTopHat, cv::MORPH_TOPHAT, structuringElement);
    cv::morphologyEx(imgGrayscale, imgBlackHat, cv::MORPH_BLACKHAT, structuringElement);

    imgGrayscalePlusTopHat = imgGrayscale + imgTopHat;
    imgGrayscalePlusTopHatMinusBlackHat = imgGrayscalePlusTopHat - imgBlackHat;

    return(imgGrayscalePlusTopHatMinusBlackHat);
}

#include <iostream>  // Add this include for std::cerr
#include <opencv2/opencv.hpp>  // OpenCV includes

///////////////////////////////////////////////////////////////////////////////////////////////////
// Add Canny edge detection and display imgCanny
void preprocess(cv::Mat& imgOriginal, cv::Mat& imgGrayscale, cv::Mat& imgThresh, cv::Mat& imgCanny) {
    imgGrayscale = extractValue(imgOriginal);                           // extract value channel only from original image to get imgGrayscale

    cv::Mat imgMaxContrastGrayscale = maximizeContrast(imgGrayscale);       // maximize contrast with top hat and black hat

    cv::Mat imgBlurred;
    cv::GaussianBlur(imgMaxContrastGrayscale, imgBlurred, GAUSSIAN_SMOOTH_FILTER_SIZE, 0);  // gaussian blur

    // call adaptive threshold to get imgThresh
    cv::adaptiveThreshold(imgBlurred, imgThresh, 255.0, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 19, 9);

    // Apply Canny edge detection
    cv::Canny(imgThresh, imgCanny, 100, 200);  // You can adjust these thresholds

    // Check if imgCanny is empty
    if (imgCanny.empty()) {
        std::cerr << "Error: imgCanny is empty!" << std::endl;
        return; // Exit function early if the image is empty
    }

    // Display the Canny edge-detected image
    cv::imshow("Canny Edge Detection", imgCanny);
    cv::waitKey(0);  // Wait for a key press to close the window
}
