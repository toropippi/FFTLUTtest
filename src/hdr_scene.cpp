#include "hdr_scene.h"

#include "images.h"

namespace fftlut {

namespace {

inline size_t PixelIndex(int x, int y, int width) {
    return (static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)) * 3U;
}

constexpr float kBloomDebugSpotIntensityScale = 45.0f;

}  // namespace

BloomDebugScene GenerateBloomDebugScene(int width, int height, uint32_t seed) {
    const GeneratedImage game_like = GenerateImage("game_like_scene_simple", 0, width, height, seed);
    const GeneratedImage text_overlay = GenerateImage("text_small", 0, width, height, seed);
    const GeneratedImage stripes = GenerateImage("stripes_sine", 1, width, height, seed);
    const GeneratedImage noise = GenerateImage("random_noise_uniform", 0, width, height, seed);

    BloomDebugScene scene;
    scene.image.width = width;
    scene.image.height = height;
    scene.image.pixels.resize(static_cast<size_t>(width) * static_cast<size_t>(height) * 3U, 0.0f);

    for (int y = 0; y < height; ++y) {
        const float ny = height > 1 ? static_cast<float>(y) / static_cast<float>(height - 1) : 0.0f;
        const float sky_blend = 1.0f - ny;
        for (int x = 0; x < width; ++x) {
            const float nx = width > 1 ? static_cast<float>(x) / static_cast<float>(width - 1) : 0.0f;
            const size_t scalar_idx = static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
            const size_t rgb_idx = PixelIndex(x, y, width);
            const float base = game_like.pixels[scalar_idx];
            const float text = text_overlay.pixels[scalar_idx];
            const float stripe = stripes.pixels[scalar_idx];
            const float n = noise.pixels[scalar_idx];

            scene.image.pixels[rgb_idx + 0U] =
                0.03f + 0.24f * base + 0.06f * stripe + 0.03f * text + 0.05f * nx + 0.04f * n;
            scene.image.pixels[rgb_idx + 1U] =
                0.04f + 0.20f * base + 0.04f * stripe + 0.07f * text + 0.06f * sky_blend + 0.03f * n;
            scene.image.pixels[rgb_idx + 2U] =
                0.05f + 0.18f * base + 0.08f * stripe + 0.02f * text + 0.14f * sky_blend + 0.03f * n;
        }
    }

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> x_dist(96, width - 97);
    std::uniform_int_distribution<int> y_dist(96, height - 97);
    std::uniform_int_distribution<int> radius_dist(3, 10);
    std::uniform_real_distribution<float> peak_dist(20.0f, 100.0f);
    std::uniform_int_distribution<int> palette_dist(0, 3);

    const std::array<float3, 4> palette = {
        make_float3(1.0f, 1.0f, 1.0f),
        make_float3(1.2f, 1.0f, 0.6f),
        make_float3(0.7f, 1.2f, 1.4f),
        make_float3(1.3f, 0.8f, 1.2f),
    };

    constexpr int kSpotCount = 12;
    scene.spots.reserve(kSpotCount);
    for (int i = 0; i < kSpotCount; ++i) {
        const int cx = x_dist(rng);
        const int cy = y_dist(rng);
        const int radius = radius_dist(rng);
        const float peak = peak_dist(rng) * kBloomDebugSpotIntensityScale;
        const float3 color = palette[palette_dist(rng)];

        scene.spots.push_back({
            cx,
            cy,
            static_cast<float>(radius),
            peak,
            color.x,
            color.y,
            color.z,
        });

        for (int y = std::max(0, cy - radius); y <= std::min(height - 1, cy + radius); ++y) {
            for (int x = std::max(0, cx - radius); x <= std::min(width - 1, cx + radius); ++x) {
                const float dx = static_cast<float>(x - cx);
                const float dy = static_cast<float>(y - cy);
                const float distance = std::sqrt(dx * dx + dy * dy);
                if (distance > static_cast<float>(radius)) {
                    continue;
                }
                const float falloff = 1.0f - distance / static_cast<float>(radius);
                const float intensity = peak * falloff * falloff;
                const size_t idx = PixelIndex(x, y, width);
                scene.image.pixels[idx + 0U] += intensity * color.x;
                scene.image.pixels[idx + 1U] += intensity * color.y;
                scene.image.pixels[idx + 2U] += intensity * color.z;
            }
        }
    }

    return scene;
}

void WriteSpotMetadataJson(
    const std::filesystem::path& path,
    const std::vector<HdrSpotMetadata>& spots,
    int width,
    int height,
    uint32_t seed) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Failed to write spot metadata JSON: " + path.string());
    }

    output << "{\n"
           << "  \"width\": " << width << ",\n"
           << "  \"height\": " << height << ",\n"
           << "  \"seed\": " << seed << ",\n"
           << "  \"spots\": [\n";
    for (size_t i = 0; i < spots.size(); ++i) {
        const HdrSpotMetadata& spot = spots[i];
        output << "    {\n"
               << "      \"center_x\": " << spot.center_x << ",\n"
               << "      \"center_y\": " << spot.center_y << ",\n"
               << "      \"radius\": " << FormatDouble(spot.radius) << ",\n"
               << "      \"peak_intensity\": " << FormatDouble(spot.peak_intensity) << ",\n"
               << "      \"color\": ["
               << FormatDouble(spot.color_r) << ", "
               << FormatDouble(spot.color_g) << ", "
               << FormatDouble(spot.color_b) << "]\n"
               << "    }";
        if (i + 1 != spots.size()) {
            output << ",";
        }
        output << "\n";
    }
    output << "  ]\n"
           << "}\n";
}

}  // namespace fftlut
