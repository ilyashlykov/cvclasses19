#include <cvlib.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

int demo_image_stitching(int argc, char* argv[])
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
        return -1;
    }

    const auto main_wnd = "Original frame";
    const auto pano_wnd = "Panorama";

    cv::namedWindow(main_wnd);
    cv::namedWindow(pano_wnd);

    cv::Mat frame;
    cv::Mat pano;

    cvlib::Stitcher stitcher(1.5);
    bool is_initialized = false;

    int key = 0;

    while (key != 27)
    {
        cap >> frame;
        if (frame.empty())
        {
            std::cerr << "Ошибка: не удалось захватить кадр." << std::endl;
            break;
        }

        cv::imshow(main_wnd, frame);

        key = cv::waitKey(30);
        if (key == 32)
        {
            try
            {
                if (!is_initialized)
                {
                    stitcher.initialize(frame);
                    is_initialized = true;
                }
                else
                {
                    stitcher.stitch(frame, pano);
                    cv::imshow(pano_wnd, pano);
                }
            }
            catch (const std::exception& ex)
            {
                std::cerr << ex.what() << std::endl;
            }
        }
    }

    cv::destroyAllWindows();
    return 0;
}