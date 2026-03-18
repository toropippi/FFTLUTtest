#include "images.h"

#include "bitmap_font.h"

namespace fftlut {

namespace {

struct Canvas {
    int width;
    int height;
    std::vector<float> pixels;

    Canvas(int w, int h) : width(w), height(h), pixels(static_cast<size_t>(w) * static_cast<size_t>(h), 0.0f) {}

    float& At(int x, int y) {
        return pixels[static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)];
    }

    const float& At(int x, int y) const {
        return pixels[static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)];
    }

    void Fill(float value) {
        std::fill(pixels.begin(), pixels.end(), value);
    }

    void Set(int x, int y, float value) {
        if (x < 0 || x >= width || y < 0 || y >= height) {
            return;
        }
        At(x, y) = Clamp(value, 0.0f, 1.0f);
    }

    void SetMax(int x, int y, float value) {
        if (x < 0 || x >= width || y < 0 || y >= height) {
            return;
        }
        At(x, y) = std::max(At(x, y), Clamp(value, 0.0f, 1.0f));
    }

    void FillRect(int x0, int y0, int w, int h, float value) {
        const int x1 = std::min(width, x0 + w);
        const int y1 = std::min(height, y0 + h);
        for (int y = std::max(0, y0); y < y1; ++y) {
            for (int x = std::max(0, x0); x < x1; ++x) {
                Set(x, y, value);
            }
        }
    }

    void DrawRectOutline(int x0, int y0, int w, int h, float value, int thickness = 1) {
        for (int t = 0; t < thickness; ++t) {
            for (int x = x0; x < x0 + w; ++x) {
                SetMax(x, y0 + t, value);
                SetMax(x, y0 + h - 1 - t, value);
            }
            for (int y = y0; y < y0 + h; ++y) {
                SetMax(x0 + t, y, value);
                SetMax(x0 + w - 1 - t, y, value);
            }
        }
    }

    void DrawLine(float x0, float y0, float x1, float y1, float value, int thickness = 1) {
        const float dx = x1 - x0;
        const float dy = y1 - y0;
        const int steps = std::max(1, static_cast<int>(std::ceil(std::max(std::abs(dx), std::abs(dy)))));
        for (int i = 0; i <= steps; ++i) {
            const float t = static_cast<float>(i) / static_cast<float>(steps);
            const int x = static_cast<int>(std::round(x0 + dx * t));
            const int y = static_cast<int>(std::round(y0 + dy * t));
            for (int oy = -thickness / 2; oy <= thickness / 2; ++oy) {
                for (int ox = -thickness / 2; ox <= thickness / 2; ++ox) {
                    SetMax(x + ox, y + oy, value);
                }
            }
        }
    }

    void DrawCircle(int cx, int cy, int radius, float value, bool fill) {
        const int r2 = radius * radius;
        for (int y = cy - radius; y <= cy + radius; ++y) {
            for (int x = cx - radius; x <= cx + radius; ++x) {
                const int dx = x - cx;
                const int dy = y - cy;
                const int d2 = dx * dx + dy * dy;
                if (fill) {
                    if (d2 <= r2) {
                        SetMax(x, y, value);
                    }
                } else {
                    if (std::abs(d2 - r2) <= radius) {
                        SetMax(x, y, value);
                    }
                }
            }
        }
    }

    void DrawText(int x, int y, int scale, const std::string& text, float value) {
        int cursor_x = x;
        for (char c : text) {
            RasterizeGlyph(c, cursor_x, y, scale, [&](int px, int py) { SetMax(px, py, value); });
            cursor_x += 6 * scale;
        }
    }
};

float Smoothstep(float edge0, float edge1, float x) {
    const float t = Clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

std::vector<float> SobelMagnitude(const std::vector<float>& input, int width, int height) {
    static const int kx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1},
    };
    static const int ky[3][3] = {
        {-1, -2, -1},
        {0, 0, 0},
        {1, 2, 1},
    };
    std::vector<float> output(input.size(), 0.0f);
    float max_mag = 0.0f;
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            float gx = 0.0f;
            float gy = 0.0f;
            for (int ky_idx = -1; ky_idx <= 1; ++ky_idx) {
                for (int kx_idx = -1; kx_idx <= 1; ++kx_idx) {
                    const float sample = input[static_cast<size_t>(y + ky_idx) * static_cast<size_t>(width) +
                                               static_cast<size_t>(x + kx_idx)];
                    gx += static_cast<float>(kx[ky_idx + 1][kx_idx + 1]) * sample;
                    gy += static_cast<float>(ky[ky_idx + 1][kx_idx + 1]) * sample;
                }
            }
            const float magnitude = std::sqrt(gx * gx + gy * gy);
            output[static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)] = magnitude;
            max_mag = std::max(max_mag, magnitude);
        }
    }
    if (max_mag > 0.0f) {
        for (float& value : output) {
            value /= max_mag;
        }
    }
    return output;
}

std::mt19937 MakeRng(uint32_t seed, const std::string& image_type, int variant_id) {
    std::seed_seq seq = {
        seed,
        static_cast<uint32_t>(std::hash<std::string>{}(image_type)),
        static_cast<uint32_t>(variant_id * 2654435761U),
    };
    return std::mt19937(seq);
}

GeneratedImage GenerateHorizontalGradient(int width, int height) {
    Canvas canvas(width, height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            canvas.At(x, y) = width > 1 ? static_cast<float>(x) / static_cast<float>(width - 1) : 0.0f;
        }
    }
    return {std::move(canvas.pixels), "linear_x", "Left-to-right linear gradient"};
}

GeneratedImage GenerateVerticalGradient(int width, int height) {
    Canvas canvas(width, height);
    for (int y = 0; y < height; ++y) {
        const float value = height > 1 ? static_cast<float>(y) / static_cast<float>(height - 1) : 0.0f;
        for (int x = 0; x < width; ++x) {
            canvas.At(x, y) = value;
        }
    }
    return {std::move(canvas.pixels), "linear_y", "Top-to-bottom linear gradient"};
}

GeneratedImage GenerateRadialGradient(int width, int height) {
    Canvas canvas(width, height);
    const float cx = 0.5f * static_cast<float>(width - 1);
    const float cy = 0.5f * static_cast<float>(height - 1);
    const float max_r = std::sqrt(cx * cx + cy * cy);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const float dx = static_cast<float>(x) - cx;
            const float dy = static_cast<float>(y) - cy;
            canvas.At(x, y) = Clamp(std::sqrt(dx * dx + dy * dy) / max_r, 0.0f, 1.0f);
        }
    }
    return {std::move(canvas.pixels), "radial_center", "Radial distance gradient"};
}

GeneratedImage GenerateSharpEdgeVertical(int width, int height) {
    Canvas canvas(width, height);
    const int split = width / 2;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            canvas.At(x, y) = x < split ? 0.0f : 1.0f;
        }
    }
    return {std::move(canvas.pixels), "center_split", "Hard vertical step edge"};
}

GeneratedImage GenerateSharpEdgeHorizontal(int width, int height) {
    Canvas canvas(width, height);
    const int split = height / 2;
    for (int y = 0; y < height; ++y) {
        const float value = y < split ? 0.0f : 1.0f;
        for (int x = 0; x < width; ++x) {
            canvas.At(x, y) = value;
        }
    }
    return {std::move(canvas.pixels), "center_split", "Hard horizontal step edge"};
}

GeneratedImage GenerateCheckerboard(int width, int height, int variant_id) {
    static const int cell_sizes[] = {4, 8, 16};
    const int cell = cell_sizes[Clamp(variant_id, 0, 2)];
    Canvas canvas(width, height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const int cx = (x / cell) & 1;
            const int cy = (y / cell) & 1;
            canvas.At(x, y) = (cx ^ cy) ? 1.0f : 0.0f;
        }
    }
    std::ostringstream variant_name;
    variant_name << "cell_" << cell;
    return {std::move(canvas.pixels), variant_name.str(), "High-frequency checkerboard"};
}

GeneratedImage GenerateImpulse(int width, int height, int variant_id) {
    Canvas canvas(width, height);
    int x = width / 2;
    int y = height / 2;
    if (variant_id == 1) {
        x = width / 3;
        y = (height * 2) / 3;
    }
    canvas.Set(x, y, 1.0f);
    return {std::move(canvas.pixels), variant_id == 0 ? "center" : "offset", "Single bright impulse"};
}

GeneratedImage GenerateStripesSine(int width, int height, int variant_id) {
    struct Params {
        float cycles;
        float angle_deg;
        float phase;
        const char* name;
    };
    static const Params kParams[] = {
        {6.0f, 0.0f, 0.0f, "freq6_horiz"},
        {12.0f, 45.0f, static_cast<float>(kPi * 0.25), "freq12_diag"},
        {20.0f, 90.0f, static_cast<float>(kPi * 0.5), "freq20_vert"},
    };
    const Params& params = kParams[Clamp(variant_id, 0, 2)];
    Canvas canvas(width, height);
    const float angle = params.angle_deg * static_cast<float>(kPi / 180.0);
    const float dir_x = std::cos(angle);
    const float dir_y = std::sin(angle);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const float nx = width > 1 ? static_cast<float>(x) / static_cast<float>(width - 1) : 0.0f;
            const float ny = height > 1 ? static_cast<float>(y) / static_cast<float>(height - 1) : 0.0f;
            const float phase = 2.0f * static_cast<float>(kPi) * params.cycles * (nx * dir_x + ny * dir_y) + params.phase;
            canvas.At(x, y) = 0.5f + 0.5f * std::sin(phase);
        }
    }
    return {std::move(canvas.pixels), params.name, "Single sinusoidal stripe field"};
}

GeneratedImage GenerateSineSum(int width, int height, int variant_id) {
    Canvas canvas(width, height);
    for (int y = 0; y < height; ++y) {
        const float ny = height > 1 ? static_cast<float>(y) / static_cast<float>(height - 1) : 0.0f;
        for (int x = 0; x < width; ++x) {
            const float nx = width > 1 ? static_cast<float>(x) / static_cast<float>(width - 1) : 0.0f;
            float value = 0.0f;
            if (variant_id == 0) {
                value += 0.40f * std::sin(2.0f * static_cast<float>(kPi) * 3.0f * nx);
                value += 0.25f * std::sin(2.0f * static_cast<float>(kPi) * 9.0f * ny + 0.35f);
                value += 0.18f * std::sin(2.0f * static_cast<float>(kPi) * 23.0f * (nx + ny));
            } else {
                value += 0.38f * std::sin(2.0f * static_cast<float>(kPi) * 5.0f * (0.8f * nx + 0.1f * ny));
                value += 0.20f * std::sin(2.0f * static_cast<float>(kPi) * 17.0f * (ny - 0.3f * nx) + 0.6f);
                value += 0.15f * std::sin(2.0f * static_cast<float>(kPi) * 31.0f * (nx - ny) + 1.2f);
            }
            canvas.At(x, y) = Clamp(0.5f + value, 0.0f, 1.0f);
        }
    }
    return {std::move(canvas.pixels), variant_id == 0 ? "mix_low_high_a" : "mix_low_high_b", "Sum of low and high-frequency sinusoids"};
}

GeneratedImage GenerateTextLarge(int width, int height) {
    Canvas canvas(width, height);
    canvas.Fill(0.0f);
    const std::string text = "FFT TEST";
    const int scale = std::max(2, std::min(width / 60, height / 20));
    const int text_width = static_cast<int>(text.size()) * 6 * scale - scale;
    const int text_height = 7 * scale;
    const int x = std::max(0, (width - text_width) / 2);
    const int y = std::max(0, (height - text_height) / 2);
    canvas.DrawText(x, y, scale, text, 1.0f);
    return {std::move(canvas.pixels), "fft_test", "Large block text pattern"};
}

GeneratedImage GenerateTextSmall(int width, int height, int variant_id) {
    Canvas canvas(width, height);
    canvas.Fill(0.0f);
    if (variant_id == 0) {
        canvas.DrawText(width / 10, height / 4, 2, "FFT LUT FAST", 1.0f);
        canvas.DrawText(width / 8, height / 2, 1, "EDGE DETAIL 01", 0.85f);
        canvas.DrawText(width / 6, (height * 3) / 4, 1, "SAMPLE 256X256", 0.75f);
        canvas.DrawLine(width / 8.0f, height / 2.0f + 16.0f, (width * 7) / 8.0f, height / 2.0f + 16.0f, 0.8f, 1);
    } else {
        canvas.DrawText(width / 12, height / 5, 1, "HUD FFT", 0.9f);
        canvas.DrawText(width / 7, height / 2, 2, "LUT", 1.0f);
        canvas.DrawText(width / 2, height / 2, 1, "FAST", 1.0f);
        canvas.DrawRectOutline(width / 8, (height * 3) / 5, width / 3, height / 8, 0.7f, 1);
        canvas.DrawText(width / 8 + 6, (height * 3) / 5 + 6, 1, "NO AA TEXT", 0.95f);
    }
    return {std::move(canvas.pixels), variant_id == 0 ? "mixed_small_text_a" : "mixed_small_text_b", "Small text and thin line pattern"};
}

GeneratedImage GenerateRandomNoiseUniform(int width, int height, uint32_t seed) {
    Canvas canvas(width, height);
    auto rng = MakeRng(seed, "random_noise_uniform", 0);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (float& value : canvas.pixels) {
        value = dist(rng);
    }
    return {std::move(canvas.pixels), "uniform01", "Uniform random noise"};
}

GeneratedImage GenerateRandomNoiseGaussianLike(int width, int height, uint32_t seed) {
    Canvas canvas(width, height);
    auto rng = MakeRng(seed, "random_noise_gaussian_like", 0);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (float& value : canvas.pixels) {
        float sum = 0.0f;
        for (int i = 0; i < 12; ++i) {
            sum += dist(rng);
        }
        const float gaussian_like = (sum - 6.0f) / 6.0f;
        value = Clamp(0.5f + gaussian_like * 0.18f, 0.0f, 1.0f);
    }
    return {std::move(canvas.pixels), "gaussian_like", "Gaussian-like random noise from uniform sums"};
}

GeneratedImage GenerateGameLikeScene(int width, int height, int variant_id) {
    Canvas canvas(width, height);
    const int horizon = height / (variant_id == 0 ? 3 : 2);
    for (int y = 0; y < height; ++y) {
        const float t = static_cast<float>(y) / static_cast<float>(std::max(1, height - 1));
        float value = 0.0f;
        if (y < horizon) {
            const float sky_t = static_cast<float>(y) / static_cast<float>(std::max(1, horizon));
            value = 0.85f - 0.45f * sky_t;
        } else {
            const float floor_t = static_cast<float>(y - horizon) / static_cast<float>(std::max(1, height - horizon - 1));
            value = 0.12f + 0.25f * (1.0f - floor_t);
        }
        for (int x = 0; x < width; ++x) {
            canvas.At(x, y) = value;
        }
    }

    const float vanish_x = variant_id == 0 ? width * 0.52f : width * 0.42f;
    const float vanish_y = static_cast<float>(horizon);
    for (int i = 1; i <= 18; ++i) {
        const float t = static_cast<float>(i) / 18.0f;
        const float curve = std::pow(t, 1.8f);
        const float y = vanish_y + curve * static_cast<float>(height - horizon - 1);
        canvas.DrawLine(0.0f, y, static_cast<float>(width - 1), y, 0.55f, 1);
    }
    for (int i = -8; i <= 8; ++i) {
        const float bottom_x = width * 0.5f + static_cast<float>(i) * width * 0.08f;
        canvas.DrawLine(vanish_x, vanish_y, bottom_x, static_cast<float>(height - 1), 0.58f, 1);
    }

    auto draw_box = [&](int x, int y, int w, int h, int depth, float fill, float outline) {
        canvas.FillRect(x, y, w, h, fill);
        canvas.DrawRectOutline(x, y, w, h, outline, 2);
        canvas.DrawRectOutline(x + depth, y - depth, w, h, outline * 0.9f, 1);
        canvas.DrawLine(static_cast<float>(x), static_cast<float>(y), static_cast<float>(x + depth), static_cast<float>(y - depth), outline, 1);
        canvas.DrawLine(static_cast<float>(x + w), static_cast<float>(y), static_cast<float>(x + w + depth), static_cast<float>(y - depth), outline, 1);
        canvas.DrawLine(static_cast<float>(x), static_cast<float>(y + h), static_cast<float>(x + depth), static_cast<float>(y + h - depth), outline, 1);
        canvas.DrawLine(static_cast<float>(x + w), static_cast<float>(y + h), static_cast<float>(x + w + depth), static_cast<float>(y + h - depth), outline, 1);
    };

    draw_box(width / 8, height / 3, width / 8, height / 5, 14, 0.18f, 0.92f);
    draw_box(width / 3, height / 4, width / 6, height / 3, 18, 0.20f, 0.95f);
    draw_box((width * 5) / 8, height / 3, width / 7, height / 4, 12, 0.24f, 0.88f);

    canvas.FillRect(width / 20, height / 20, width / 4, height / 8, 0.10f);
    canvas.DrawRectOutline(width / 20, height / 20, width / 4, height / 8, 0.92f, 2);
    canvas.DrawText(width / 20 + 8, height / 20 + 8, 1, "FFT HUD", 0.98f);
    canvas.DrawText(width / 20 + 8, height / 20 + 24, 1, "FPS 120", 0.86f);

    canvas.FillRect((width * 3) / 4, height / 18, width / 5, height / 12, 0.90f);
    canvas.DrawRectOutline((width * 3) / 4, height / 18, width / 5, height / 12, 1.0f, 1);
    canvas.DrawText((width * 3) / 4 + 6, height / 18 + 6, 1, "UI BOX", 0.15f);

    const int cross_x = width / 2;
    const int cross_y = (height * 11) / 16;
    canvas.DrawLine(static_cast<float>(cross_x - 10), static_cast<float>(cross_y), static_cast<float>(cross_x + 10), static_cast<float>(cross_y), 0.97f, 1);
    canvas.DrawLine(static_cast<float>(cross_x), static_cast<float>(cross_y - 10), static_cast<float>(cross_x), static_cast<float>(cross_y + 10), 0.97f, 1);

    return {std::move(canvas.pixels), variant_id == 0 ? "city_block_a" : "city_block_b", "Synthetic game-like scene with gradients, grid, UI, and text"};
}

GeneratedImage GenerateEdgeEnhancedScene(int width, int height) {
    Canvas base(width, height);
    for (int y = 0; y < height; ++y) {
        const float t = static_cast<float>(y) / static_cast<float>(std::max(1, height - 1));
        for (int x = 0; x < width; ++x) {
            base.At(x, y) = 0.12f + 0.15f * t;
        }
    }
    base.FillRect(width / 8, height / 5, width / 4, height / 3, 0.8f);
    base.DrawCircle((width * 3) / 4, height / 3, width / 10, 0.95f, true);
    base.DrawLine(width / 10.0f, height * 0.8f, width * 0.9f, height * 0.55f, 0.85f, 4);
    base.DrawText(width / 5, (height * 3) / 4, 2, "EDGE", 1.0f);
    std::vector<float> sobel = SobelMagnitude(base.pixels, width, height);
    return {std::move(sobel), "sobel_scene", "Sobel-like edge-enhanced synthetic scene"};
}

ImagePreset MakePreset(const char* image_type, int variant_id, const char* variant_name) {
    return {image_type, variant_id, variant_name};
}

}  // namespace

std::vector<ImagePreset> EnumerateImagePresets() {
    return {
        MakePreset("horizontal_gradient", 0, "linear_x"),
        MakePreset("vertical_gradient", 0, "linear_y"),
        MakePreset("radial_gradient", 0, "radial_center"),
        MakePreset("sharp_edge_vertical", 0, "center_split"),
        MakePreset("sharp_edge_horizontal", 0, "center_split"),
        MakePreset("checkerboard", 0, "cell_4"),
        MakePreset("checkerboard", 1, "cell_8"),
        MakePreset("checkerboard", 2, "cell_16"),
        MakePreset("impulse", 0, "center"),
        MakePreset("impulse", 1, "offset"),
        MakePreset("stripes_sine", 0, "freq6_horiz"),
        MakePreset("stripes_sine", 1, "freq12_diag"),
        MakePreset("stripes_sine", 2, "freq20_vert"),
        MakePreset("sine_sum", 0, "mix_low_high_a"),
        MakePreset("sine_sum", 1, "mix_low_high_b"),
        MakePreset("text_large", 0, "fft_test"),
        MakePreset("text_small", 0, "mixed_small_text_a"),
        MakePreset("text_small", 1, "mixed_small_text_b"),
        MakePreset("random_noise_uniform", 0, "uniform01"),
        MakePreset("random_noise_gaussian_like", 0, "gaussian_like"),
        MakePreset("game_like_scene_simple", 0, "city_block_a"),
        MakePreset("game_like_scene_simple", 1, "city_block_b"),
        MakePreset("edge_enhanced_scene", 0, "sobel_scene"),
    };
}

std::vector<ImagePreset> EnumerateRepresentativeLargePresets() {
    return {
        MakePreset("horizontal_gradient", 0, "linear_x"),
        MakePreset("sharp_edge_vertical", 0, "center_split"),
        MakePreset("text_small", 0, "mixed_small_text_a"),
        MakePreset("game_like_scene_simple", 0, "city_block_a"),
    };
}

GeneratedImage GenerateImage(
    const std::string& image_type,
    int variant_id,
    int width,
    int height,
    uint32_t seed) {
    const std::string key = ToLower(image_type);
    if (key == "horizontal_gradient") {
        return GenerateHorizontalGradient(width, height);
    }
    if (key == "vertical_gradient") {
        return GenerateVerticalGradient(width, height);
    }
    if (key == "radial_gradient") {
        return GenerateRadialGradient(width, height);
    }
    if (key == "sharp_edge_vertical") {
        return GenerateSharpEdgeVertical(width, height);
    }
    if (key == "sharp_edge_horizontal") {
        return GenerateSharpEdgeHorizontal(width, height);
    }
    if (key == "checkerboard") {
        return GenerateCheckerboard(width, height, variant_id);
    }
    if (key == "impulse") {
        return GenerateImpulse(width, height, variant_id);
    }
    if (key == "stripes_sine") {
        return GenerateStripesSine(width, height, variant_id);
    }
    if (key == "sine_sum") {
        return GenerateSineSum(width, height, variant_id);
    }
    if (key == "text_large") {
        return GenerateTextLarge(width, height);
    }
    if (key == "text_small") {
        return GenerateTextSmall(width, height, variant_id);
    }
    if (key == "random_noise_uniform") {
        return GenerateRandomNoiseUniform(width, height, seed);
    }
    if (key == "random_noise_gaussian_like") {
        return GenerateRandomNoiseGaussianLike(width, height, seed);
    }
    if (key == "game_like_scene_simple") {
        return GenerateGameLikeScene(width, height, variant_id);
    }
    if (key == "edge_enhanced_scene") {
        return GenerateEdgeEnhancedScene(width, height);
    }
    throw std::invalid_argument("Unknown image type: " + image_type);
}

}  // namespace fftlut
