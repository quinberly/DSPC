#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <filesystem>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

// Sauvola Thresholding Function
Mat sauvolaThreshold(const Mat& srcGray, int windowSize = 25, double k = 0.5, double R = 128.0)
{
    Mat threshImg = Mat::zeros(srcGray.size(), CV_8UC1);

    Mat intImg, sqIntImg;
    integral(srcGray, intImg, sqIntImg, CV_64F);

    int halfWin = windowSize / 2;
    for (int y = halfWin; y < srcGray.rows - halfWin; ++y)
    {
        for (int x = halfWin; x < srcGray.cols - halfWin; ++x)
        {
            int x0 = x - halfWin;
            int y0 = y - halfWin;
            int x1 = x + halfWin;
            int y1 = y + halfWin;

            int area = (x1 - x0 + 1) * (y1 - y0 + 1);

            double sum = intImg.at<double>(y1 + 1, x1 + 1) - intImg.at<double>(y1 + 1, x0) - intImg.at<double>(y0, x1 + 1) + intImg.at<double>(y0, x0);
            double sqSum = sqIntImg.at<double>(y1 + 1, x1 + 1) - sqIntImg.at<double>(y1 + 1, x0) - sqIntImg.at<double>(y0, x1 + 1) + sqIntImg.at<double>(y0, x0);

            double m = sum / area;
            double s = sqrt((sqSum - (sum * sum) / area) / area);

            double T = m * (1 + k * ((s / R) - 1));
            threshImg.at<uchar>(y, x) = (srcGray.at<uchar>(y, x) > T) ? 255 : 0;
        }
    }

    return threshImg;
}

int main()
{
    string folderPath = "C:/Users/User/Desktop/DSPC/DSPC/images";
    string outputFolder = "C:/Users/User/Desktop/DSPC/DSPC/output";
    fs::create_directories(outputFolder);

    for (const auto& entry : fs::directory_iterator(folderPath))
    {
        string imagePath = entry.path().string();
        Mat image = imread(imagePath);

        if (image.empty())
        {
            cerr << "Failed to load image: " << imagePath << endl;
            continue;
        }

        // Convert to grayscale and filter
        Mat gray, filtered;
        cvtColor(image, gray, COLOR_BGR2GRAY);
        bilateralFilter(gray, filtered, 9, 75, 75); // Fine-tune or remove this
        medianBlur(filtered, filtered, 5);  // Fine-tune or remove this

        // Edge detection with adjusted thresholds
        Mat edges;
        Canny(filtered, edges, 50, 200); // Adjust thresholds for better edge detection

        // Find contours with more checks
        vector<vector<Point>> contours;
        findContours(edges, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        vector<Point> plateContour;
        for (const auto& contour : contours)
        {
            double peri = arcLength(contour, true);
            vector<Point> approx;
            approxPolyDP(contour, approx, 0.02 * peri, true); // Adjust approximation accuracy
            if (approx.size() == 4)
            {
                // Check aspect ratio of the bounding rectangle
                Rect plateRect = boundingRect(approx);
                double aspectRatio = (double)plateRect.width / plateRect.height;
                if (aspectRatio > 2 && aspectRatio < 5)  // Adjust aspect ratio thresholds
                {
                    plateContour = approx;
                    break;
                }
            }
        }

        if (!plateContour.empty())
        {
            Rect plateRect = boundingRect(plateContour);
            Mat plate = image(plateRect).clone();

            // Convert plate to grayscale and apply Sauvola thresholding
            Mat plateGray;
            cvtColor(plate, plateGray, COLOR_BGR2GRAY);
            Mat plateSauvola = sauvolaThreshold(plateGray);

            // Save cropped and processed plate
            string fileName = fs::path(imagePath).filename().string();
            string outPath = outputFolder + "/sauvola_" + fileName;
            imwrite(outPath, plateSauvola);

            // Display result
            imshow("Detected Plate (Sauvola)", plateSauvola);
            waitKey(0);
        }
        else
        {
            cout << "No license plate detected in: " << imagePath << endl;
        }
    }

    return 0;
}  