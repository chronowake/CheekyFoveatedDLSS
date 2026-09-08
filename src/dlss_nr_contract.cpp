#include "dlss_nr_contract.hpp"

#include "foveation.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace cheeky::foveated_dlss {

DlssNrResourceBase dlss_nr_resource_base(
    const std::uint32_t local_x,
    const std::uint32_t local_y,
    const std::uint32_t color_base_x,
    const std::uint32_t color_base_y,
    const bool color_is_region
) noexcept {
    return color_is_region
        ? DlssNrResourceBase{0U, 0U}
        : DlssNrResourceBase{
            color_base_x + local_x,
            color_base_y + local_y,
        };
}

FoveationParameters dlss_nr_foveation_parameters(
    const Settings& settings,
    const FoveationGeometry* const shared_sr_crop,
    const std::uint32_t render_width,
    const std::uint32_t render_height
) noexcept {
    const bool follow_sr = settings.nr_use_sr_foveation &&
        shared_sr_crop != nullptr && render_width != 0U && render_height != 0U;
    FoveationParameters parameters{
        follow_sr ? settings.width : settings.nr_width,
        follow_sr ? settings.height : settings.nr_height,
        0.0F,
        0.0F,
        follow_sr ? settings.roundness : settings.nr_roundness,
        follow_sr ? settings.transition_width : settings.nr_transition_width,
    };
    float center_x = settings.nr_center_x + settings.nr_x_offset;
    float center_y = settings.nr_height_offset;
    if (follow_sr) {
        center_x = (2.0F * shared_sr_crop->input_base_x +
            shared_sr_crop->input_width) / render_width - 1.0F;
        center_y = (2.0F * shared_sr_crop->input_base_y +
            shared_sr_crop->input_height) / render_height - 1.0F;
    }
    parameters.x_offset = parameters.width < 1.0F
        ? std::clamp(center_x / (1.0F - parameters.width), -1.0F, 1.0F) : 0.0F;
    parameters.y_offset = parameters.height < 1.0F
        ? std::clamp(center_y / (1.0F - parameters.height), -1.0F, 1.0F) : 0.0F;
    return parameters;
}

DlssNrDisplayedView dlss_nr_displayed_view(
    const std::uint32_t view_width,
    const std::uint32_t view_height,
    const std::uint32_t travel_width,
    const std::uint32_t travel_height
) noexcept {
    const auto span_width = travel_width == 0U ? view_width : travel_width;
    const auto span_height = travel_height == 0U ? view_height : travel_height;
    const bool packed_sbs =
        view_width != 0U && span_width >= view_width * 2U &&
        (view_height == 0U || span_height < view_height * 2U);
    const bool packed_tab =
        view_height != 0U && span_height >= view_height * 2U &&
        (view_width == 0U || span_width < view_width * 2U);
    return {
        packed_sbs ? view_width : span_width,
        packed_tab ? view_height : span_height,
        span_width,
        span_height,
    };
}

void apply_nr_gaze_uv(
    Settings& settings,
    const float gaze_u,
    const float gaze_v,
    const std::uint32_t view_width,
    const std::uint32_t view_height,
    const std::uint32_t travel_width,
    const std::uint32_t travel_height
) noexcept {
    const auto displayed = dlss_nr_displayed_view(
        view_width, view_height, travel_width, travel_height
    );
    if (displayed.travel_width != 0U) {
        settings.nr_center_x += (gaze_u - 0.5F) *
            static_cast<float>(displayed.width) /
            static_cast<float>(displayed.travel_width);
    }
    if (displayed.travel_height != 0U) {
        settings.nr_height_offset += (gaze_v - 0.5F) *
            static_cast<float>(displayed.height) /
            static_cast<float>(displayed.travel_height);
    }
}

DlssNrViewCrop calculate_dlss_nr_view_crop(
    const Settings& settings,
    const std::uint32_t view_width,
    const std::uint32_t view_height,
    const FoveationGeometry* const shared_sr_crop,
    const std::uint32_t render_width,
    const std::uint32_t render_height,
    const std::uint32_t travel_width,
    const std::uint32_t travel_height,
    const std::uint32_t eye_base_x,
    const std::uint32_t eye_base_y
) noexcept {
    DlssNrViewCrop crop{};
    if (view_width == 0U || view_height == 0U) return crop;
    const auto displayed = dlss_nr_displayed_view(
        view_width, view_height, travel_width, travel_height
    );
    const auto span_width = displayed.travel_width;
    const auto span_height = displayed.travel_height;
    const auto size_width = displayed.width;
    const auto size_height = displayed.height;
    const auto align_down = [](const std::int64_t value) noexcept {
        const auto quantized = value - value % 8;
        return quantized < 0 ? 0 : static_cast<std::uint32_t>(quantized);
    };
    const auto align_up = [](const std::uint32_t value) noexcept {
        return (std::max)(8U, (value + 7U) / 8U * 8U);
    };
    if (settings.nr_use_sr_foveation && shared_sr_crop != nullptr &&
        render_width != 0U && render_height != 0U) {
        const auto parameters = dlss_nr_foveation_parameters(
            settings, shared_sr_crop, render_width, render_height
        );
        FoveationGeometry geometry{};
        if (calculate_foveation_geometry(
                parameters, view_width, view_height, view_width, view_height,
                0U, 0U, geometry
            )) {
            crop.origin_x = align_down(
                static_cast<std::int64_t>(eye_base_x) + geometry.output_base_x
            );
            crop.origin_y = align_down(
                static_cast<std::int64_t>(eye_base_y) + geometry.output_base_y
            );
            crop.width = align_up(geometry.output_width);
            crop.height = align_up(geometry.output_height);
        }
    } else {
        const auto width = std::clamp(settings.nr_width, 0.20F, 1.0F);
        const auto height = std::clamp(settings.nr_height, 0.20F, 1.0F);
        crop.width = align_up(static_cast<std::uint32_t>(std::lround(
            static_cast<double>(size_width) * width
        )));
        crop.height = align_up(static_cast<std::uint32_t>(std::lround(
            static_cast<double>(size_height) * height
        )));
        if (crop.width > span_width) crop.width = align_down(span_width);
        if (crop.height > span_height) crop.height = align_down(span_height);
        const auto center_x =
            static_cast<double>(eye_base_x) +
            0.5 * static_cast<double>(size_width) +
            static_cast<double>(settings.nr_center_x) * span_width +
            static_cast<double>(settings.nr_x_offset) * size_width;
        const auto center_y =
            static_cast<double>(eye_base_y) +
            0.5 * static_cast<double>(size_height) +
            static_cast<double>(settings.nr_height_offset) * span_height;
        const auto origin_x = static_cast<std::int64_t>(std::llround(
            center_x - 0.5 * crop.width + settings.nr_source_x
        ));
        const auto origin_y = static_cast<std::int64_t>(std::llround(
            center_y - 0.5 * crop.height + settings.nr_source_y
        ));
        crop.origin_x = align_down(origin_x);
        crop.origin_y = align_down(origin_y);
    }
    if (crop.width == 0U || crop.width > span_width) {
        crop.width = align_down(span_width);
    }
    if (crop.height == 0U || crop.height > span_height) {
        crop.height = align_down(span_height);
    }
    if (crop.origin_x + crop.width > span_width) {
        crop.origin_x = align_down(
            static_cast<std::int64_t>(span_width) - crop.width
        );
    }
    if (crop.origin_y + crop.height > span_height) {
        crop.origin_y = align_down(
            static_cast<std::int64_t>(span_height) - crop.height
        );
    }
    return crop;
}

}  // namespace cheeky::foveated_dlss
