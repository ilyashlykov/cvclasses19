// /* Split and merge segmentation algorithm implementation.
//  * @file
//  * @date 2018-09-05
//  * @author Anonymous
//  */

#include "cvlib.hpp"
#include <vector>

struct Region
{
    cv::Point topLeft;
    cv::Point bottomRight;
    std::vector<cv::Point> pixelIndices;  
    cv::Mat mean;
    cv::Mat dev;
    cv::Mat image;
};
namespace
{
void split_image(cv::Mat image, std::vector<Region>& regions, double stddev, cv::Point offset = cv::Point(0, 0))
{
    if (image.empty())
    {
        return;
    }
    
    cv::Mat mean, dev;
    cv::meanStdDev(image, mean, dev);

    if (dev.at<double>(0) <= stddev)
    {
        cv::Point topLeft = offset;
        cv::Point bottomRight = offset + cv::Point(image.cols - 1, image.rows - 1);
        
        std::vector<cv::Point> pixelCoords;
        for (int y = 0; y < image.rows; ++y)
        {
            for (int x = 0; x < image.cols; ++x)
            {
                pixelCoords.push_back(offset + cv::Point(x, y));  
            }
        }

        regions.push_back(Region{topLeft, bottomRight, pixelCoords, mean, dev, image});
        return;
    }

    const auto width = image.cols;
    const auto height = image.rows;

    if (width > 1 && height > 1)
    {
        split_image(image(cv::Range(0, height / 2), cv::Range(0, width / 2)), regions, stddev, offset);
        split_image(image(cv::Range(0, height / 2), cv::Range(width / 2, width)), regions, stddev, offset + cv::Point(width / 2, 0));
        split_image(image(cv::Range(height / 2, height), cv::Range(width / 2, width)), regions, stddev, offset + cv::Point(width / 2, height / 2));
        split_image(image(cv::Range(height / 2, height), cv::Range(0, width / 2)), regions, stddev, offset + cv::Point(0, height / 2));
    }

    if (width == 1 && height > 1)
    {
        split_image(image(cv::Range(0, height / 2), cv::Range(0, 1)), regions, stddev, offset);
        split_image(image(cv::Range(height / 2, height), cv::Range(0, 1)), regions, stddev, offset + cv::Point(0, height / 2));
    }
    if (width > 1 && height == 1)
    {
        split_image(image(cv::Range(0, 1), cv::Range(0, width / 2)), regions, stddev, offset);
        split_image(image(cv::Range(0, 1), cv::Range(width / 2, width)), regions, stddev, offset + cv::Point(width / 2, 0));
    }
}
}

bool is_neighbour(const Region& region1, const Region& region2)
{
    cv::Point topLeft1 = region1.topLeft;
    cv::Point bottomRight1 = region1.bottomRight;

    cv::Point topLeft2 = region2.topLeft;
    cv::Point bottomRight2 = region2.bottomRight;

    bool horizontal_neighbour = std::abs(bottomRight1.x - topLeft2.x) <= 1 || std::abs(bottomRight2.x - topLeft1.x) <= 1;

    bool vertical_neighbour = std::abs(bottomRight1.y - topLeft2.y) <= 1 || std::abs(topLeft1.y - bottomRight2.y) <= 1;

    return horizontal_neighbour && vertical_neighbour;
}


bool predicat(cv::Mat image, Region& region1, Region& region2, double stddev)
{
    std::vector<cv::Point> combined_pixelIndices = region1.pixelIndices;
    combined_pixelIndices.insert(combined_pixelIndices.end(), region2.pixelIndices.begin(), region2.pixelIndices.end());

    std::vector<uchar> combined_pixels;
    for (const cv::Point& coord : combined_pixelIndices)
    {
        combined_pixels.push_back(image.at<uchar>(coord.y, coord.x));
    }

    cv::Mat combined_mat(combined_pixels);
    cv::Scalar mean, dev;
    cv::meanStdDev(combined_mat, mean, dev);

    if (dev[0] <= stddev) 
    {
        region1.pixelIndices = combined_pixelIndices;
        region1.mean = mean; 

        return true;
    }

    return false;
}

void merge_regions2(cv::Mat image, std::vector<Region>& regions, double stddev)
{
    bool merged_flag = true;

    while (merged_flag)
    {
        merged_flag = false;
        std::vector<bool> active(regions.size(), true); 

        for (size_t i = 0; i < regions.size(); ++i)
        {
            if (!active[i]) continue; 

            for (size_t j = i + 1; j < regions.size(); ++j)
            {
                if (!active[j]) continue; 

                if (!is_neighbour(regions[i], regions[j]))
                    continue;

                if (predicat(image, regions[i], regions[j], stddev))
                {
                    active[j] = false;  
                    merged_flag = true;
                    break;  
                }
            }
        }
        auto it = regions.begin();
        for (size_t i = 0; i < active.size(); ++i)
        {
            if (!active[i])
                it = regions.erase(it);
            else
                ++it;
        }
    }
}


namespace cvlib
{
cv::Mat split_and_merge(const cv::Mat& image, double stddev)
{
    std::vector<Region> regions;
    cv::Mat res = image.clone();

    split_image(res, regions, stddev);
    merge_regions2(res, regions, stddev);

    for (const auto& region : regions)
    {
        for (const cv::Point& coord : region.pixelIndices)
        {
            res.at<uchar>(coord.y, coord.x) = static_cast<uchar>(region.mean.at<double>(0));
        }
    }

    return res;
}
} // namespace cvlib
