/* Split and merge segmentation algorithm testing.
 * @file
 * @date 2018-09-05
 * @author Anonymous
 */

#include <catch2/catch.hpp>

#include "cvlib.hpp"

using namespace cvlib;

TEST_CASE("constant image", "[split_and_merge]")
{
    const cv::Mat image(100, 100, CV_8UC1, cv::Scalar{15});

    const auto res = split_and_merge(image, 1);
    REQUIRE(image.size() == res.size());
    REQUIRE(image.type() == res.type());
    REQUIRE(cv::Scalar(15) == cv::mean(res));
}

TEST_CASE("simple regions", "[split_and_merge]")
{
    SECTION("2x2")
    {
        const cv::Mat reference = (cv::Mat_<char>(2, 2) << 1, 1, 1, 1);
        cv::Mat image = (cv::Mat_<char>(2, 2) << 0, 1, 2, 3);
        auto res = split_and_merge(image, 10);
        REQUIRE(image.size() == res.size());
        REQUIRE(image.type() == res.type());

        REQUIRE(0 == cv::countNonZero(reference - res));

        res = split_and_merge(image, 0);
        REQUIRE(0 == cv::countNonZero(image - res));
    }

    SECTION("3x3")
    {
        const cv::Mat reference = (cv::Mat_<char>(3, 3) << 4,4,4,4,4,4,4,4,4);
        cv::Mat image = (cv::Mat_<char>(3,3) << 0,1,2,3,4,5,6,7,8);
        auto res = split_and_merge(image, 10);
        REQUIRE(image.size() == res.size());
        REQUIRE(image.type() == res.type());
        REQUIRE(0 == cv::countNonZero(reference - res));
        
        res = split_and_merge(image, 0);
        REQUIRE(0 == cv::countNonZero(image - res));
    }
}

TEST_CASE("compex regions", "[split_and_merge]")
{
    SECTION("2x2")
    {
        const cv::Mat reference = (cv::Mat_<char>(2, 2) << 3,3,8,8);
        cv::Mat image = (cv::Mat_<char>(2, 2) << 1, 5, 7, 9);
        auto res = split_and_merge(image, 2);
        REQUIRE(image.size() == res.size());
        REQUIRE(image.type() == res.type());

        REQUIRE(0 == cv::countNonZero(reference - res));
    }

    SECTION("3x3")
    {
        const cv::Mat reference = (cv::Mat_<char>(3, 3) << 1,1,1,7,1,8,7,1,1);
        cv::Mat image = (cv::Mat_<char>(3, 3) << 1,1,1, 6,1,8,9,1,4);
        auto res = split_and_merge(image, 2);
        REQUIRE(image.size() == res.size());
        REQUIRE(image.type() == res.type());

        REQUIRE(0 == cv::countNonZero(reference - res));
        res = split_and_merge(image, 1);
        REQUIRE(0 == cv::countNonZero(image - res));
    }

    SECTION("4x4")
    {
        const cv::Mat reference = (cv::Mat_<char>(4, 4) << 8,8,2,2,2,2,2,2,2,2,7,7,2,2,7,7);
        cv::Mat image = (cv::Mat_<char>(4, 4) << 9,8,1,1,3,2,4, 2,1,2, 7, 7, 1, 3, 8 , 7);
        auto res = split_and_merge(image, 2);
        REQUIRE(image.size() == res.size());
        REQUIRE(image.type() == res.type());
        REQUIRE(0 == cv::countNonZero(reference - res));

        res = split_and_merge(image, 0);
        REQUIRE(0 == cv::countNonZero(image - res));
    }
}
