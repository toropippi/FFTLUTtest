#include "exr_io.h"

#include <cstdlib>

#define NOMINMAX
#define TINYEXR_USE_MINIZ 1
#define TINYEXR_IMPLEMENTATION
#include "../third_party/tinyexr.h"

namespace fftlut {

namespace {

[[noreturn]] void ThrowExrError(const std::string& prefix, const char* err) {
    std::string message = prefix;
    if (err != nullptr) {
        message += ": ";
        message += err;
        FreeEXRErrorMessage(err);
    }
    throw std::runtime_error(message);
}

void ValidateStandardRgbChannels(const std::filesystem::path& path) {
    EXRVersion version = {};
    EXRHeader header;
    InitEXRHeader(&header);

    const char* err = nullptr;
    int ret = ParseEXRVersionFromFile(&version, path.string().c_str());
    if (ret != TINYEXR_SUCCESS) {
        ThrowExrError("Failed to parse EXR version for " + path.string(), err);
    }

    ret = ParseEXRHeaderFromFile(&header, &version, path.string().c_str(), &err);
    if (ret != TINYEXR_SUCCESS) {
        ThrowExrError("Failed to parse EXR header for " + path.string(), err);
    }

    bool has_r = false;
    bool has_g = false;
    bool has_b = false;
    for (int i = 0; i < header.num_channels; ++i) {
        const std::string name = header.channels[i].name;
        has_r = has_r || (name == "R");
        has_g = has_g || (name == "G");
        has_b = has_b || (name == "B");
    }

    FreeEXRHeader(&header);

    if (!has_r || !has_g || !has_b) {
        throw std::runtime_error(
            "EXR must contain standard unlayered R, G, and B channels: " + path.string());
    }
}

int InferSquareDimensionFromRgbFloatBinSize(uintmax_t byte_size, const std::filesystem::path& path) {
    constexpr uintmax_t kBytesPerPixel = 3U * sizeof(float);
    if (byte_size == 0 || (byte_size % kBytesPerPixel) != 0) {
        throw std::runtime_error(
            "RGB float bin size is not divisible by 3*sizeof(float): " + path.string());
    }

    const uintmax_t pixel_count = byte_size / kBytesPerPixel;
    const double side_f = std::sqrt(static_cast<double>(pixel_count));
    const int side = static_cast<int>(std::llround(side_f));
    if (static_cast<uintmax_t>(side) * static_cast<uintmax_t>(side) != pixel_count) {
        throw std::runtime_error(
            "RGB float bin does not describe a square image: " + path.string());
    }
    if (!IsPowerOfTwo(side)) {
        throw std::runtime_error(
            "RGB float bin side length must be a power of two: " + path.string());
    }
    return side;
}

}  // namespace

RgbImageF LoadRgbExr(const std::filesystem::path& path) {
    ValidateStandardRgbChannels(path);

    float* rgba = nullptr;
    int width = 0;
    int height = 0;
    const char* err = nullptr;
    const int ret = LoadEXR(&rgba, &width, &height, path.string().c_str(), &err);
    if (ret != TINYEXR_SUCCESS) {
        ThrowExrError("Failed to load EXR image " + path.string(), err);
    }

    RgbImageF image;
    image.width = width;
    image.height = height;
    image.pixels.resize(static_cast<size_t>(width) * static_cast<size_t>(height) * 3U);
    for (size_t i = 0; i < static_cast<size_t>(width) * static_cast<size_t>(height); ++i) {
        image.pixels[i * 3U + 0U] = rgba[i * 4U + 0U];
        image.pixels[i * 3U + 1U] = rgba[i * 4U + 1U];
        image.pixels[i * 3U + 2U] = rgba[i * 4U + 2U];
    }

    std::free(rgba);
    return image;
}

RgbImageF LoadRgbFloatBin(const std::filesystem::path& path) {
    const uintmax_t byte_size = std::filesystem::file_size(path);
    const int side = InferSquareDimensionFromRgbFloatBinSize(byte_size, path);

    RgbImageF image;
    image.width = side;
    image.height = side;
    image.pixels.resize(static_cast<size_t>(side) * static_cast<size_t>(side) * 3U);

    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Failed to open RGB float bin: " + path.string());
    }

    input.read(
        reinterpret_cast<char*>(image.pixels.data()),
        static_cast<std::streamsize>(image.pixels.size() * sizeof(float)));
    if (!input || input.gcount() != static_cast<std::streamsize>(image.pixels.size() * sizeof(float))) {
        throw std::runtime_error("Failed to read RGB float bin payload: " + path.string());
    }

    for (float value : image.pixels) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("RGB float bin contains non-finite values: " + path.string());
        }
    }

    return image;
}

RgbImageF LoadRgbImageAuto(const std::filesystem::path& path) {
    const std::string extension = ToLower(path.extension().string());
    if (extension == ".exr") {
        return LoadRgbExr(path);
    }
    if (extension == ".bin") {
        return LoadRgbFloatBin(path);
    }
    throw std::runtime_error(
        "Unsupported kernel image format for " + path.string() + ". Expected .exr or .bin.");
}

}  // namespace fftlut
