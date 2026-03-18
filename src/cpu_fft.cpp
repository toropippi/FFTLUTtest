#include "cpu_fft.h"

namespace fftlut {

namespace {

void Fft1DInplace(std::vector<std::complex<double>>& values, bool inverse) {
    const int n = static_cast<int>(values.size());
    const int log2_n = IntegerLog2(n);
    for (int i = 0; i < n; ++i) {
        const int reversed = static_cast<int>(ReverseBits(static_cast<uint32_t>(i), log2_n));
        if (reversed > i) {
            std::swap(values[i], values[reversed]);
        }
    }

    for (int stage = 0; stage < log2_n; ++stage) {
        const int m = 1 << (stage + 1);
        const int half_m = m >> 1;
        const double sign = inverse ? 1.0 : -1.0;
        for (int base = 0; base < n; base += m) {
            for (int j = 0; j < half_m; ++j) {
                const double angle = sign * 2.0 * kPi * static_cast<double>(j) / static_cast<double>(m);
                const std::complex<double> twiddle(std::cos(angle), std::sin(angle));
                const std::complex<double> u = values[base + j];
                const std::complex<double> v = twiddle * values[base + j + half_m];
                values[base + j] = u + v;
                values[base + j + half_m] = u - v;
            }
        }
    }
}

void TransformRows(std::vector<std::complex<double>>& data, int row_length, int row_count, bool inverse) {
    std::vector<std::complex<double>> row(static_cast<size_t>(row_length));
    for (int row_idx = 0; row_idx < row_count; ++row_idx) {
        const size_t offset = static_cast<size_t>(row_idx) * static_cast<size_t>(row_length);
        std::copy(data.begin() + static_cast<std::ptrdiff_t>(offset),
                  data.begin() + static_cast<std::ptrdiff_t>(offset + row_length),
                  row.begin());
        Fft1DInplace(row, inverse);
        std::copy(row.begin(), row.end(), data.begin() + static_cast<std::ptrdiff_t>(offset));
    }
}

std::vector<std::complex<double>> Transpose(
    const std::vector<std::complex<double>>& input,
    int width,
    int height) {
    std::vector<std::complex<double>> output(input.size());
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            output[static_cast<size_t>(x) * static_cast<size_t>(height) + static_cast<size_t>(y)] =
                input[static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)];
        }
    }
    return output;
}

std::vector<std::complex<double>> Run2DComplex(
    const std::vector<std::complex<double>>& input,
    int width,
    int height,
    bool inverse) {
    std::vector<std::complex<double>> data = input;
    TransformRows(data, width, height, inverse);
    data = Transpose(data, width, height);
    TransformRows(data, height, width, inverse);
    data = Transpose(data, height, width);
    if (inverse) {
        const double scale = 1.0 / static_cast<double>(width * height);
        for (std::complex<double>& value : data) {
            value *= scale;
        }
    }
    return data;
}

}  // namespace

CpuFftResult RunCpuReference2D(const std::vector<float>& image, int width, int height) {
    CpuFftResult result;
    result.spectrum = Forward2DRealToComplex(image, width, height);
    result.reconstruction = Inverse2DComplexToReal(result.spectrum, width, height);
    return result;
}

std::vector<std::complex<double>> Forward2DRealToComplex(
    const std::vector<float>& image,
    int width,
    int height) {
    std::vector<std::complex<double>> complex_input(image.size());
    for (size_t i = 0; i < image.size(); ++i) {
        complex_input[i] = std::complex<double>(static_cast<double>(image[i]), 0.0);
    }
    return Run2DComplex(complex_input, width, height, false);
}

std::vector<double> Inverse2DComplexToReal(
    const std::vector<std::complex<double>>& spectrum,
    int width,
    int height) {
    const std::vector<std::complex<double>> reconstruction_complex =
        Run2DComplex(spectrum, width, height, true);
    std::vector<double> reconstruction(reconstruction_complex.size());
    for (size_t i = 0; i < reconstruction_complex.size(); ++i) {
        reconstruction[i] = reconstruction_complex[i].real();
    }
    return reconstruction;
}

}  // namespace fftlut
