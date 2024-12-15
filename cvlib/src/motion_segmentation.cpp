/* Split and merge segmentation algorithm implementation.
 * @file
 * @date 2018-09-18
 * @author Anonymous
 */

#include "cvlib.hpp"

#include <iostream>

namespace cvlib
{

void gaussian_segmentation(const cv::Mat& image, cv::Mat& fgmask, cv::Mat& background_mean, cv::Mat& background_variance, double rho) {
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            float pixel = image.at<uchar>(y, x);
            float mean = background_mean.at<float>(y, x);
            float variance = background_variance.at<float>(y, x);
            float std_dev = std::sqrt(variance);

            if (std::abs(pixel - mean) > 3 * std_dev) { // 3-сигма правило
                fgmask.at<uchar>(y, x) = 255;
            } else {
                fgmask.at<uchar>(y, x) = 0;
            }

            // Обновление модели
            background_mean.at<float>(y, x) = rho * pixel + (1 - rho) * mean;
            background_variance.at<float>(y, x) = rho * (pixel - mean) * (pixel - mean) + (1 - rho) * variance;
        }
    }
}


void motion_segmentation::apply(cv::InputArray _image, cv::OutputArray _fgmask, double rho)
{  
    cv::Mat grayscale_image;
    if (_image.channels() > 1) {
        cv::cvtColor(_image, grayscale_image, cv::COLOR_RGB2GRAY);
    } else {
        grayscale_image = _image.getMat();
    }

    static cv::Mat background_mean(grayscale_image.size(), CV_32FC1, cv::Scalar(0));
    static cv::Mat background_variance(grayscale_image.size(), CV_32FC1, cv::Scalar(15)); 
    cv::Mat fgmask(grayscale_image.size(), CV_8UC1, cv::Scalar(0));

    gaussian_segmentation(grayscale_image, fgmask, background_mean, background_variance, rho);

    fgmask.copyTo(_fgmask);
}
} // namespace cvlib
