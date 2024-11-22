// /* Split and merge segmentation algorithm implementation.
//  * @file
//  * @date 2018-09-05
//  * @author Anonymous
//  */

// #include "cvlib.hpp"
// #include "vector"

// struct Region
// {
//     cv::Mat image;
//     double mean;
//     double dev;
// };

// namespace
// {
// void split_image(cv::Mat image, std::vector<Region>& regions, double stddev)
// {
//     if (image.empty())
//     {
//         return ;
//     }
//     cv::Mat mean;
//     cv::Mat dev;

//     // double mean = meanMat.at<double>(0);
//     // double dev = devMat.at<double>(0);

//     cv::meanStdDev(image, mean, dev);

//     if (dev.at<double>(0) <= stddev)
//     {
//         image.setTo(mean);
//         regions.push_back(Region{image, mean.at<double>(0), dev.at<double>(0)});
//         return;
//     }

//     const auto width = image.cols;
//     const auto height = image.rows;

//     if (width > 1 && height > 1)
//     {
//         split_image(image(cv::Range(0, height / 2), cv::Range(0, width / 2)), regions, stddev);
//         split_image(image(cv::Range(0, height / 2), cv::Range(width / 2, width)), regions, stddev);
//         split_image(image(cv::Range(height / 2, height), cv::Range(width / 2, width)), regions, stddev);
//         split_image(image(cv::Range(height / 2, height), cv::Range(0, width / 2)), regions, stddev);
//     }

//     else if (width == 1 && height > 1)
//     {
//         split_image(image(cv::Range(0, height / 2), cv::Range(0, 1)), regions, stddev);
//         split_image(image(cv::Range(height / 2, height), cv::Range(0, 1)), regions, stddev);
//     }

//     else if (width > 1 && height == 1)
//     {
//         split_image(image(cv::Range(0, 1), cv::Range(0, width/2)), regions, stddev);
//         split_image(image(cv::Range(0, 1), cv::Range(width/2, width)), regions, stddev);
//     }
// }
// } // namespace


// void merge_regions(std::vector<Region>& regions, double stddev)
// {
//     bool merged = true;
//     while (merged)
//     {
//         merged = false;
//         for (size_t i = 0; i < regions.size() - 1; ++i)
//         {
//             for (size_t j = i + 1; j < regions.size(); ++j)
//             {
//                 // Проверка соседних регионов и объединение, если их stddev близки
//                 if (std::abs(regions[i].dev - regions[j].dev) <= stddev)
//                 {
//                     // Объединение регионов
//                     cv::hconcat(regions[i].image, regions[j].image, regions[i].image);
//                     regions[i].mean = (regions[i].mean + regions[j].mean) / 2;
//                     regions[i].dev = (regions[i].dev + regions[j].dev) / 2;

//                     // Удаление второго региона после объединения
//                     regions.erase(regions.begin() + j);
//                     merged = true;
//                     break;
//                 }
//             }
//             if (merged)
//                 break;
//         }
//     }
// }

// cv::Mat assemble_image(const std::vector<Region>& regions, cv::Size originalSize)
// {
//     cv::Mat result = cv::Mat::zeros(originalSize, regions[0].image.type());

//     for (const auto& region : regions)
//     {
//         region.image.copyTo(result(cv::Rect(region.topLeft, region.image.size())));
//     }

//     return result;
// }

// namespace cvlib
// {
// cv::Mat split_and_merge(const cv::Mat& image, double stddev)
// {

//     std::vector<Region> regions;
//     // split part
//     cv::Mat res = image.clone(); // Клонируем изображение
//     split_image(res, regions, stddev);

//     // merge part
//     merge_regions(res, regions, stddev);
    
//     image = assemble_image(regions, original_size)
//     return res;
// }
// } // namespace cvlib

//-----_----------_--_---_--_-__-_-_-_-_-_-___------_-_-_-_---_-__________-_-_----_--_-_--_-_-_-_-_-_-_-_-_-_-_-_-_-_-__-_-_-_-_-_-_--_-_-_-_-_-_-_-_-_-_--_-_____-__-_-_--_-_-_-____-_--_-_--_-_----_--_-_--_-_--_----_--_-__-_--_-_---_--_----_-

#include "cvlib.hpp"
#include <vector>

struct Region
{
    cv::Point topLeft;
    cv::Point bottomRight;
    std::vector<cv::Point> pixelIndices;  // Индексы пикселей в изображении (одномерный индекс в массиве)
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
        // Создание нового региона
        cv::Point topLeft = offset;
        cv::Point bottomRight = offset + cv::Point(image.cols - 1, image.rows - 1);
        
        // Храним координаты пикселей в исходном изображении
        std::vector<cv::Point> pixelCoords;
        for (int y = 0; y < image.rows; ++y)
        {
            for (int x = 0; x < image.cols; ++x)
            {
                pixelCoords.push_back(offset + cv::Point(x, y));  // Координаты относительно исходного изображения
            }
        }

        // Добавление нового региона в список
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

    // Проверяем соседство по горизонтали
    bool horizontal_neighbour = std::abs(bottomRight1.x - topLeft2.x) <= 1 || std::abs(bottomRight2.x - topLeft1.x) <= 1;

    // Проверяем соседство по вертикали
    bool vertical_neighbour = std::abs(bottomRight1.y - topLeft2.y) <= 1 || std::abs(topLeft1.y - bottomRight2.y) <= 1;

    return horizontal_neighbour && vertical_neighbour;
}


bool predicat(cv::Mat image, Region& region1, Region& region2, double stddev)
{
    // Объединяем координаты пикселей из двух регионов
    std::vector<cv::Point> combined_pixelIndices = region1.pixelIndices;
    combined_pixelIndices.insert(combined_pixelIndices.end(), region2.pixelIndices.begin(), region2.pixelIndices.end());

    // Создаем матрицу, представляющую все объединенные пиксели
    std::vector<uchar> combined_pixels;
    for (const cv::Point& coord : combined_pixelIndices)
    {
        combined_pixels.push_back(image.at<uchar>(coord.y, coord.x));
    }

    // Считаем среднее и дисперсию объединенного массива
    cv::Mat combined_mat(combined_pixels);
    cv::Scalar mean, dev;
    cv::meanStdDev(combined_mat, mean, dev);

    // Проверка по предикату
    if (dev[0] <= stddev) 
    {
        // Обновляем регион1
        region1.pixelIndices = combined_pixelIndices;
        region1.mean = mean;  // Обновляем среднее для region1

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
        std::vector<bool> active(regions.size(), true); // Флаги активности регионов

        for (size_t i = 0; i < regions.size(); ++i)
        {
            if (!active[i]) continue; // Пропускаем неактивные регионы

            for (size_t j = i + 1; j < regions.size(); ++j)
            {
                if (!active[j]) continue; // Пропускаем неактивные регионы

                if (!is_neighbour(regions[i], regions[j]))
                    continue;

                if (predicat(image, regions[i], regions[j], stddev))
                {
                    active[j] = false;  // Отмечаем j как неактивный
                    merged_flag = true;
                    break;  // Выходим из внутреннего цикла после объединения
                }
            }
        }

        // Удаляем все неактивные регионы
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

    // Обновляем цвета пикселей в результатирующем изображении
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