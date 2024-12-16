#include "cvlib.hpp"
#include <limits>
#include <cmath>
#include <algorithm>

namespace cvlib
{

float computeSSD(const cv::Mat& query, const cv::Mat& train, int queryIdx, int trainIdx)
{
    float ssd = 0.0f;
    float diff;
    for (int col = 0; col < query.cols; ++col)
    {
        diff = query.at<float>(queryIdx, col) - train.at<float>(trainIdx, col);
        ssd += diff * diff;
    }
    return ssd;
}

void descriptor_matcher::knnMatchImpl(cv::InputArray queryDescriptors, 
                                      std::vector<std::vector<cv::DMatch>>& matches, 
                                      int k /*unhandled*/, 
                                      cv::InputArrayOfArrays masks /*unhandled*/, 
                                      bool compactResult /*unhandled*/)
{
    if (trainDescCollection.empty())
        return;

    auto q_desc = queryDescriptors.getMat();  
    auto& t_desc = trainDescCollection[0];    

    q_desc.convertTo(q_desc, CV_32F);
    t_desc.convertTo(t_desc, CV_32F);

    matches.resize(q_desc.rows);

    for (int i = 0; i < q_desc.rows; ++i)
    {
        float best_dist = FLT_MAX, second_best_dist = FLT_MAX;
        int best_idx = -1;

        // Находим два лучших совпадения
        for (int j = 0; j < t_desc.rows; ++j)
        {
            float ssd = computeSSD(q_desc, t_desc, i, j);

            if (ssd < best_dist)
            {
                second_best_dist = best_dist;
                best_dist = ssd;
                best_idx = j;
            }
            else if (ssd < second_best_dist)
            {
                second_best_dist = ssd;
            }
        }

        // Применяем Ratio Test (сравнение отношения)
        if (best_dist / second_best_dist < ratio_)
        {
            matches[i].emplace_back(i, best_idx, best_dist);
        }
    }
}

void descriptor_matcher::radiusMatchImpl(cv::InputArray queryDescriptors, std::vector<std::vector<cv::DMatch>>& matches, float maxDistance,
                                         cv::InputArrayOfArrays masks /*unhandled*/, bool compactResult /*unhandled*/)
{
    knnMatchImpl(queryDescriptors, matches, 1, masks, compactResult);
}

} // namespace cvlib