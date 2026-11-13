#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <string>

int main(int argc, char* argv[])
{
    std::string imagePath = "2181.png";
    bool showWindows = true;
    bool hasImagePath = false;
    for (int i = 1; i < argc; ++i)
    {
        const std::string argument = argv[i];
        if (argument == "--no-gui")
        {
            showWindows = false;
        }
        else if (!hasImagePath)
        {
            imagePath = argument;
            hasImagePath = true;
        }
        else
        {
            std::cerr << "Usage: OpenCVTemplate.exe [image path] [--no-gui]\n";
            return 1;
        }
    }

    const cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (image.empty())
    {
        std::cerr << "Cannot read " << imagePath << "\n";
        return 1;
    }

    // Make the dark paper clips white in the mask for connected-component analysis.
    cv::Mat gray, binary;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::medianBlur(gray, gray, 3);
    const double thresholdValue = cv::threshold(
        gray, binary, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

    cv::Mat labels, stats, centers;
    const int labelCount = cv::connectedComponentsWithStats(
        binary, labels, stats, centers, 8, CV_32S);

    const int minArea = 500; // Reject dust and small marks in the source image.
    cv::Mat result = image.clone();
    int count = 0;
    for (int label = 1; label < labelCount; ++label)
    {
        const cv::Rect box(
            stats.at<int>(label, cv::CC_STAT_LEFT),
            stats.at<int>(label, cv::CC_STAT_TOP),
            stats.at<int>(label, cv::CC_STAT_WIDTH),
            stats.at<int>(label, cv::CC_STAT_HEIGHT));
        // The black strip along the image edge also becomes foreground.
        const bool touchesBorder = box.x == 0 || box.y == 0
            || box.br().x == image.cols || box.br().y == image.rows;
        if (touchesBorder || stats.at<int>(label, cv::CC_STAT_AREA) < minArea)
        {
            continue;
        }

        ++count;
        const cv::Point center(cvRound(centers.at<double>(label, 0)),
            cvRound(centers.at<double>(label, 1)));
        cv::rectangle(result, box, cv::Scalar(0, 255, 0), 2);
        cv::putText(result, std::to_string(count), center,
            cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    }

    const std::string totalText = "Total: " + std::to_string(count);
    cv::rectangle(result, cv::Rect(12, 10, 125, 28), cv::Scalar(245, 245, 245), cv::FILLED);
    cv::putText(result, totalText, cv::Point(20, 30),
        cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(20, 20, 20), 2, cv::LINE_AA);

    std::cout << "Otsu threshold: " << thresholdValue << "\n";
    std::cout << totalText << "\n";
    const std::string outputPath = "count_result.png";
    if (!cv::imwrite(outputPath, result))
    {
        std::cerr << "Cannot save " << outputPath << "\n";
        return 1;
    }

    if (showWindows)
    {
        cv::namedWindow("Binary mask", cv::WINDOW_NORMAL);
        cv::namedWindow("Paper clip count", cv::WINDOW_NORMAL);
        cv::imshow("Binary mask", binary);
        cv::imshow("Paper clip count", result);
        cv::waitKey(0);
    }
    return 0;
}
