/* FAST corner detector algorithm implementation.
 * @file
 * @date 2018-10-16
 * @author Anonymous
 */

#include "cvlib.hpp"

#include <ctime>

namespace cvlib
{
// static
cv::Ptr<corner_detector_fast> corner_detector_fast::create()
{
    return cv::makePtr<corner_detector_fast>();
}

void corner_detector_fast::detect(cv::InputArray image, CV_OUT std::vector<cv::KeyPoint>& keypoints, cv::InputArray /*mask = cv::noArray()*/)
{
    keypoints.clear();

    cv::Mat img = image.getMat();

    if (img.channels() > 1)
        cv::cvtColor(img, img, cv::COLOR_RGB2GRAY);

    const int threshold = 40;
    const int t = 9; 

    const int circleOffsets[16][2] = {
        {0, 3}, {1, 2}, {2, 2}, {3, 1},
        {3, 0}, {3, -1}, {2, -2}, {1, -3},
        {0, -3}, {-1, -3}, {-2, -2}, {-3, -1},
        {-3, 0}, {-3, 1}, {-2, 2}, {-1, 3}
    };

    for (int y = 3; y < img.rows - 3; y++)
    {
        for (int x = 3; x < img.cols - 3; x++)
        {
            int centerPixel = img.at<uchar>(y, x);

            // Вектор для хранения информации о яркости пикселя: 1 - центральный пиксель темнее, 2 - ярче
            std::vector<int> binaryCircle(16, 0);
            for (int i = 0; i < 16; i++)
            {
                int offsetY = circleOffsets[i][0];
                int offsetX = circleOffsets[i][1];
                int neighborPixel = img.at<uchar>(y + offsetY, x + offsetX);

                if (neighborPixel > centerPixel + threshold)
                    binaryCircle[i] = 1; 
                else if (neighborPixel < centerPixel - threshold)
                    binaryCircle[i] = 2;
            }

            int count = 0;
            int currentValue = binaryCircle[0];
            for (int i = 0; i < (16 + t - 1); i++) // Зацикленный проход
            {
                if (currentValue == 0)
                {
                    count = 0;
                    currentValue = binaryCircle[(i+1) % 16];
                    continue;
                }

                if (binaryCircle[i % 16] == currentValue)
                {
                    count++;
                    if (count >= t)
                    {
                        keypoints.emplace_back(cv::KeyPoint(cv::Point2f(x, y), 7));
                        break;
                    }
                }
                else
                {
                    currentValue = binaryCircle[i % 16];
                    count = 1;
                }
            }
        }
    }
}

void corner_detector_fast::compute(cv::InputArray image, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors)
{
    cv::Mat img = image.getMat();
    const int desc_length = 16; // Простая длина дескриптора
    descriptors.create(static_cast<int>(keypoints.size()), desc_length, CV_8U);
    auto desc_mat = descriptors.getMat();
    desc_mat.setTo(0);

    for (size_t i = 0; i < keypoints.size(); ++i)
    {
        const cv::KeyPoint& kp = keypoints[i];
        uchar* desc_ptr = desc_mat.ptr<uchar>(static_cast<int>(i));

        // Пример: записать значения яркости в шаблон 4x4 вокруг ключевой точки
        for (int dy = -2; dy <= 1; ++dy)
        {
            for (int dx = -2; dx <= 1; ++dx)
            {
                int px = kp.pt.x + dx;
                int py = kp.pt.y + dy;
                desc_ptr[(dy + 2) * 4 + (dx + 2)] = img.at<uchar>(py, px);
            }
        }
    }
}

void corner_detector_fast::detectAndCompute(cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors, bool useProvidedKeypoints)
{
    if (!useProvidedKeypoints)
    {
        detect(image, keypoints, mask);
    }
    compute(image, keypoints, descriptors);
}
} // namespace cvlib
