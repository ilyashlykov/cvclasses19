/* FAST corner detector algorithm testing.
 * @file
 * @date 2018-09-05
 * @author Anonymous
 */

#include <catch2/catch.hpp>

#include "cvlib.hpp"

using namespace cvlib;

TEST_CASE("simple check", "[corner_detector_fast]")
{
    auto fast = corner_detector_fast::create();
    cv::Mat image(10, 10, CV_8UC1);

    SECTION("empty image")
    {
        std::vector<cv::KeyPoint> out;
        fast->detect(image, out);
        REQUIRE(out.empty());
    }

    SECTION("single corner") 
    {
        cv::Mat image(10, 10, CV_8UC1, cv::Scalar(0));
        image.at<uchar>(5, 5) = 255; // Устанавливаем пиксельный угол


        std::vector<cv::KeyPoint> out;
        fast->detect(image, out);
        REQUIRE(out.size() == 1); // Должен быть найден один угол
        REQUIRE(out[0].pt.x == 5); 
        REQUIRE(out[0].pt.y == 5);
    }

    SECTION("multiple corners") 
    {
  
        // Создаем черный квадрат 10x10
        cv::Mat black_block = cv::Mat::zeros(10, 10, CV_8UC1);
        cv::Mat white_block = cv::Mat::ones(2, 2, CV_8UC1) * 255;

        // Горизонтальная и вертикальная сборка
        cv::Mat top_padding = black_block(cv::Range(0, 4), cv::Range::all());
        cv::Mat bottom_padding = black_block(cv::Range(6, 10), cv::Range::all());
        cv::Mat left_padding = black_block(cv::Range::all(), cv::Range(0, 4));
        cv::Mat right_padding = black_block(cv::Range::all(), cv::Range(6, 10));

        cv::Mat center_row;
        cv::hconcat(std::vector<cv::Mat>{left_padding(cv::Range(4, 6), cv::Range::all()), 
                                        white_block, 
                                        right_padding(cv::Range(4, 6), cv::Range::all())}, 
                    center_row);

        cv::Mat full_image;
        cv::vconcat(std::vector<cv::Mat>{top_padding, center_row, bottom_padding}, full_image);

        std::vector<cv::KeyPoint> out;
        fast->detect(full_image, out);
        REQUIRE(out.size() == 4); 
    }

    SECTION("no corners in uniform image") 
    {
        cv::Mat image(10, 10, CV_8UC1, cv::Scalar(127)); 
        std::vector<cv::KeyPoint> out;
        fast->detect(image, out);
        REQUIRE(out.empty()); 
    }

    SECTION("detectAndCompute") 
    {
        cv::Mat image(10, 10, CV_8UC1, cv::Scalar(0));
        image.at<uchar>(5, 5) = 255;

        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        fast->detectAndCompute(image, cv::noArray(), keypoints, descriptors);

        REQUIRE(keypoints.size() == 1); 
        REQUIRE(descriptors.rows == 1); 
        REQUIRE(descriptors.cols > 0);  
    }

    SECTION("boundary conditions") 
    {
        cv::Mat image(10, 10, CV_8UC1, cv::Scalar(0));
        image.at<uchar>(0, 0) = 255;

        std::vector<cv::KeyPoint> out;
        fast->detect(image, out);

        REQUIRE(out.empty()); 
    }
}
