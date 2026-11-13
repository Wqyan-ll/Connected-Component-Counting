#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

bool countObjects(const std::string& imagePath, cv::Mat& result, cv::Mat& binary)
{
    const cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (image.empty())
    {
        std::cerr << "Cannot read " << imagePath << "\n";
        return false;
    }

    const std::string filename = imagePath.substr(imagePath.find_last_of("/\\") + 1);
    const bool countHoles = filename == "2182.png";
    // Restrict hole detection to the right half of the wheel.
    cv::Rect detectionArea(0, 0, image.cols, image.rows);
    if (countHoles)
    {
        detectionArea.x = image.cols / 2;
        detectionArea.width -= detectionArea.x;
    }
    cv::Mat gray;
    cv::cvtColor(image(detectionArea), gray, cv::COLOR_BGR2GRAY);
    cv::medianBlur(gray, gray, 3);
    cv::Mat foreground;
    const int thresholdMode = countHoles ? cv::THRESH_BINARY_INV : cv::THRESH_BINARY;
    const double thresholdValue = cv::threshold(
        gray, foreground, 0, 255, thresholdMode | cv::THRESH_OTSU);
    if (!countHoles && cv::countNonZero(foreground) > foreground.total() / 2)
    {
        cv::bitwise_not(foreground, foreground);
    }

    binary = cv::Mat::zeros(image.size(), CV_8UC1);
    cv::Mat labels, stats, centers;
    const int labelCount = cv::connectedComponentsWithStats(
        foreground, labels, stats, centers, 8, CV_32S);

    const int minArea = 500; // Reject dust and small marks in the source image.
    cv::Mat annotated = image.clone();
    int count = 0;
    for (int label = 1; label < labelCount; ++label)
    {
        const cv::Rect box(
            stats.at<int>(label, cv::CC_STAT_LEFT),
            stats.at<int>(label, cv::CC_STAT_TOP),
            stats.at<int>(label, cv::CC_STAT_WIDTH),
            stats.at<int>(label, cv::CC_STAT_HEIGHT));
        // Reject clipped objects and the dark frame.
        const bool touchesBorder = box.x == 0 || box.y == 0
            || box.br().x == foreground.cols || box.br().y == foreground.rows;
        if (touchesBorder || stats.at<int>(label, cv::CC_STAT_AREA) < minArea)
        {
            continue;
        }

        if (countHoles)
        {
            // Keep complete, roughly round holes in the size range seen on this wheel.
            const int shortSide = std::min(box.width, box.height);
            const int longSide = std::max(box.width, box.height);
            if (shortSide < 30 || longSide > 110 || shortSide < 0.75 * longSide)
            {
                continue;
            }
        }

        const cv::Point offset = detectionArea.tl();
        binary(box + offset).setTo(255, labels(box) == label);
        ++count;
        const cv::Point center(cvRound(centers.at<double>(label, 0)),
            cvRound(centers.at<double>(label, 1)));
        cv::rectangle(annotated, box + offset, cv::Scalar(0, 255, 0), 2);
        cv::putText(annotated, std::to_string(count), center + offset,
            cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    }

    const std::string caption = filename + "  |  Total: " + std::to_string(count);
    cv::copyMakeBorder(annotated, result, 40, 0, 0, 0,
        cv::BORDER_CONSTANT, cv::Scalar(245, 245, 245));
    cv::putText(result, caption, cv::Point(12, 27),
        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(20, 20, 20), 2, cv::LINE_AA);

    std::cout << filename << " | Otsu threshold: " << thresholdValue
        << " | Total: " << count << "\n";
    return true;
}

cv::Mat placeSideBySide(const std::vector<cv::Mat>& images)
{
    constexpr int gap = 12;
    int height = 0;
    int width = -gap;
    for (const cv::Mat& image : images)
    {
        height = std::max(height, image.rows);
        width += image.cols + gap;
    }

    cv::Mat combined(height, width, images.front().type(), cv::Scalar::all(32));
    int x = 0;
    for (const cv::Mat& image : images)
    {
        image.copyTo(combined(cv::Rect(x, 0, image.cols, image.rows)));
        x += image.cols + gap;
    }
    return combined;
}

int main(int argc, char* argv[])
{
    std::vector<std::string> imagePaths;
    bool showWindows = true;
    for (int i = 1; i < argc; ++i)
    {
        const std::string argument = argv[i];
        if (argument == "--no-gui")
        {
            showWindows = false;
        }
        else
        {
            imagePaths.push_back(argument);
        }
    }
    if (imagePaths.empty())
    {
        imagePaths = {"2181.png", "2182.png", "2183.png"};
    }

    std::vector<cv::Mat> results, masks;
    for (const std::string& imagePath : imagePaths)
    {
        cv::Mat result, binary;
        if (!countObjects(imagePath, result, binary))
        {
            return 1;
        }
        results.push_back(result);
        masks.push_back(binary);
    }

    const cv::Mat combined = placeSideBySide(results);
    const std::string outputPath = "count_result.png";
    if (!cv::imwrite(outputPath, combined))
    {
        std::cerr << "Cannot save " << outputPath << "\n";
        return 1;
    }

    if (showWindows)
    {
        const cv::Mat combinedMasks = placeSideBySide(masks);
        cv::namedWindow("Binary masks", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
        cv::namedWindow("Counting results", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
        cv::imshow("Binary masks", combinedMasks);
        cv::imshow("Counting results", combined);
        cv::waitKey(0);
    }
    return 0;
}
