#include <cvlib.hpp>
#include <stdexcept>

namespace cvlib
{
Stitcher::Stitcher(float ratio)
    : matcher_(ratio)
{
    // Создаём детектор углов
    detector_ = cvlib::corner_detector_fast::create();
}

void Stitcher::initialize(cv::InputArray input)
{
    if (input.empty())
        throw std::runtime_error("Ошибка: входное изображение пустое.");

    // Копируем первое изображение как начальную панораму
    input.getMat().copyTo(pano_);

    // Ищем ключевые точки и дескрипторы
    detector_->detectAndCompute(pano_, cv::noArray(), prev_keypoints_, prev_descriptors_);
}

void Stitcher::stitch(cv::InputArray input, cv::OutputArray output)
{
    if (pano_.empty())
        throw std::runtime_error("Ошибка: панорама ещё не инициализирована.");

    if (input.empty())
        throw std::runtime_error("Ошибка: входное изображение пустое.");

    // Преобразуем входное изображение
    cv::Mat current;
    input.getMat().copyTo(current);

    // Ищем ключевые точки и дескрипторы нового изображения
    std::vector<cv::KeyPoint> current_keypoints;
    cv::Mat current_descriptors;
    detector_->detectAndCompute(current, cv::noArray(), current_keypoints, current_descriptors);

    if (current_keypoints.empty() || current_descriptors.empty())
        throw std::runtime_error("Ошибка: не удалось найти ключевые точки или дескрипторы.");

    // Сопоставляем дескрипторы
    std::vector<std::vector<cv::DMatch>> matches;
    matcher_.radiusMatch(current_descriptors, prev_descriptors_, matches, 50);

    // Извлекаем сопоставленные точки
    std::vector<cv::Point2f> src_pts, dst_pts;
    for (const auto& match : matches)
    {
        if (!match.empty())
        {
            src_pts.push_back(current_keypoints[match[0].queryIdx].pt);
            dst_pts.push_back(prev_keypoints_[match[0].trainIdx].pt);
        }
    }

    if (src_pts.size() < 4 || dst_pts.size() < 4)
        throw std::runtime_error("Ошибка: недостаточно совпадений для вычисления гомографии.");

    // Вычисляем гомографию
    cv::Mat homography = cv::findHomography(src_pts, dst_pts, cv::RANSAC);

    if (homography.empty())
        throw std::runtime_error("Ошибка: не удалось вычислить гомографию.");

    // Создаём новый размер для панорамы
    cv::Size new_size(pano_.cols + current.cols, std::max(pano_.rows, current.rows));

    // Проецируем новое изображение в панораму
    cv::Mat warped;
    cv::warpPerspective(current, warped, homography, new_size);

    // Копируем текущую панораму в новую область
    cv::Mat roi(warped, cv::Rect(0, 0, pano_.cols, pano_.rows));
    pano_.copyTo(roi);

    // Обновляем панораму
    pano_ = warped;
    pano_.copyTo(output);

    // Обновляем ключевые точки и дескрипторы
    prev_keypoints_ = current_keypoints;
    prev_descriptors_ = current_descriptors;
}
} // namespace cvlib