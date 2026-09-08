#pragma once

#include "settings.hpp"

#include <cstdint>

namespace cheeky::foveated_dlss {

struct DlssNrResourceBase {
    std::uint32_t x{};
    std::uint32_t y{};
};

// color_is_region means the texture already contains only the NR crop, whose
// resource origin is zero. Otherwise add the eye/output base to the crop offset.
[[nodiscard]] DlssNrResourceBase dlss_nr_resource_base(
    std::uint32_t local_x,
    std::uint32_t local_y,
    std::uint32_t color_base_x,
    std::uint32_t color_base_y,
    bool color_is_region
) noexcept;

[[nodiscard]] FoveationParameters dlss_nr_foveation_parameters(
    const Settings& settings,
    const FoveationGeometry* shared_sr_crop,
    std::uint32_t render_width,
    std::uint32_t render_height
) noexcept;

struct DlssNrViewCrop {
    std::uint32_t origin_x{};
    std::uint32_t origin_y{};
    std::uint32_t width{};
    std::uint32_t height{};
};

// Packed SBS/TAB keeps one-eye NGX size. Otherwise a larger color texture is
// the displayed view, so 50% and unfoveated cover that picture, not a
// top-left OutWidth box.
struct DlssNrDisplayedView {
    std::uint32_t width{};
    std::uint32_t height{};
    std::uint32_t travel_width{};
    std::uint32_t travel_height{};
};

[[nodiscard]] DlssNrDisplayedView dlss_nr_displayed_view(
    std::uint32_t view_width,
    std::uint32_t view_height,
    std::uint32_t travel_width = 0U,
    std::uint32_t travel_height = 0U
) noexcept;

// Gaze u/v is 0-1 in that eye's displayed view. Adds a span-normalized offset
// on top of the origin sliders so those sliders stay a bias / fallback.
void apply_nr_gaze_uv(
    Settings& settings,
    float gaze_u,
    float gaze_v,
    std::uint32_t view_width,
    std::uint32_t view_height,
    std::uint32_t travel_width = 0U,
    std::uint32_t travel_height = 0U
) noexcept;

[[nodiscard]] DlssNrViewCrop calculate_dlss_nr_view_crop(
    const Settings& settings,
    std::uint32_t view_width,
    std::uint32_t view_height,
    const FoveationGeometry* shared_sr_crop,
    std::uint32_t render_width,
    std::uint32_t render_height,
    std::uint32_t travel_width = 0U,
    std::uint32_t travel_height = 0U,
    std::uint32_t eye_base_x = 0U,
    std::uint32_t eye_base_y = 0U
) noexcept;

}  // namespace cheeky::foveated_dlss
