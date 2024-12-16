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

    if (img.channels() > 1)
        cv::cvtColor(img, img, cv::COLOR_RGB2GRAY);

    const int patchSize = 31;    // Размер патча вокруг ключевой точки
    const int desc_length = 256; // Длина дескриптора (256 бит)

    // Случайные смещения для бинарных тестов (инициализируются один раз)
    static std::vector<cv::Point> offsets1, offsets2;
    if (offsets1.empty() || offsets2.empty())
    {
        cv::RNG rng(42); // Инициализация генератора случайных чисел
        for (int i = 0; i < desc_length; ++i)
        {
            offsets1.emplace_back(rng.uniform(-patchSize / 2, patchSize / 2),
                                  rng.uniform(-patchSize / 2, patchSize / 2));
            offsets2.emplace_back(rng.uniform(-patchSize / 2, patchSize / 2),
                                  rng.uniform(-patchSize / 2, patchSize / 2));
        }
    }

    descriptors.create(static_cast<int>(keypoints.size()), desc_length / 8, CV_8U);
    auto desc_mat = descriptors.getMat();
    desc_mat.setTo(0);

    for (size_t i = 0; i < keypoints.size(); ++i)
    {
        const cv::KeyPoint& kp = keypoints[i];
        uchar* desc_ptr = desc_mat.ptr<uchar>(static_cast<int>(i));

        if (kp.pt.x < patchSize / 2 || kp.pt.x >= img.cols - patchSize / 2 ||
            kp.pt.y < patchSize / 2 || kp.pt.y >= img.rows - patchSize / 2)
        {
            continue; // Пропускаем ключевые точки, для которых невозможно вычислить дескриптор
        }

        // Формируем бинарный дескриптор BRIEF
        for (int j = 0; j < desc_length; ++j)
        {
            cv::Point p1 = cv::Point(kp.pt) + offsets1[j];
            cv::Point p2 = cv::Point(kp.pt) + offsets2[j];

            int intensity1 = img.at<uchar>(p1.y, p1.x);
            int intensity2 = img.at<uchar>(p2.y, p2.x);

            if (intensity1 < intensity2)
            {
                desc_ptr[j / 8] |= (1 << (7 - (j % 8))); // Устанавливаем бит
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
