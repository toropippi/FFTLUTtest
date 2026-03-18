#include "gpu_fft.h"

namespace fftlut {

namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kTransposeTileDim = 32;
constexpr int kTransposeBlockRows = 8;

__device__ inline uint32_t ReverseBitsDevice(uint32_t value, int bit_count) {
    uint32_t reversed = 0;
    for (int i = 0; i < bit_count; ++i) {
        reversed = (reversed << 1U) | (value & 1U);
        value >>= 1U;
    }
    return reversed;
}

__device__ inline float2 ComplexMul(const float2 a, const float2 b) {
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

struct LutTwiddleProvider {
    __device__ static float2 Load(int table_index, int /*row_length*/, bool inverse, const float2* lut) {
        const float2 base = lut[table_index];
        return make_float2(base.x, inverse ? base.y : -base.y);
    }
};

struct FastTwiddleProvider {
    __device__ static float2 Load(int table_index, int row_length, bool inverse, const float2* /*lut*/) {
        float sine = 0.0f;
        float cosine = 0.0f;
        const float angle = 2.0f * 3.14159265358979323846f *
                            static_cast<float>(table_index) /
                            static_cast<float>(row_length);
        __sincosf(angle, &sine, &cosine);
        return make_float2(cosine, inverse ? sine : -sine);
    }
};

__global__ void RealToComplexKernel(const float* input, float2* output, int element_count) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= element_count) {
        return;
    }
    output[idx] = make_float2(input[idx], 0.0f);
}

__global__ void BitReverseRowsKernel(
    const float2* src,
    float2* dst,
    int row_length,
    int row_count,
    int log2_length) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = row_length * row_count;
    if (idx >= total) {
        return;
    }
    const int row = idx / row_length;
    const int col = idx - row * row_length;
    const int reversed = static_cast<int>(ReverseBitsDevice(static_cast<uint32_t>(col), log2_length));
    dst[row * row_length + reversed] = src[idx];
}

template <typename TwiddleProvider>
__global__ void FftStageRowsKernel(
    float2* data,
    int row_length,
    int row_count,
    int stage,
    bool inverse,
    const float2* lut) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int butterflies_per_row = row_length / 2;
    const int total = butterflies_per_row * row_count;
    if (idx >= total) {
        return;
    }

    const int row = idx / butterflies_per_row;
    const int butterfly = idx - row * butterflies_per_row;
    const int m = 1 << (stage + 1);
    const int half_m = m >> 1;
    const int group = butterfly / half_m;
    const int j = butterfly - group * half_m;
    const int index0 = group * m + j;
    const int index1 = index0 + half_m;
    const int twiddle_stride = row_length / m;
    const int twiddle_index = j * twiddle_stride;

    float2* row_ptr = data + row * row_length;
    const float2 u = row_ptr[index0];
    const float2 v = row_ptr[index1];
    const float2 twiddle = TwiddleProvider::Load(twiddle_index, row_length, inverse, lut);
    const float2 t = ComplexMul(twiddle, v);
    row_ptr[index0] = make_float2(u.x + t.x, u.y + t.y);
    row_ptr[index1] = make_float2(u.x - t.x, u.y - t.y);
}

__global__ void ScaleComplexKernel(float2* data, float scale, int element_count) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= element_count) {
        return;
    }
    data[idx].x *= scale;
    data[idx].y *= scale;
}

__global__ void HadamardMultiplyKernel(float2* lhs, const float2* rhs, int element_count) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= element_count) {
        return;
    }
    lhs[idx] = ComplexMul(lhs[idx], rhs[idx]);
}

__global__ void TransposeKernel(const float2* src, float2* dst, int width, int height) {
    __shared__ float2 tile[kTransposeTileDim][kTransposeTileDim + 1];

    int x = blockIdx.x * kTransposeTileDim + threadIdx.x;
    int y = blockIdx.y * kTransposeTileDim + threadIdx.y;
    for (int j = 0; j < kTransposeTileDim; j += kTransposeBlockRows) {
        if (x < width && (y + j) < height) {
            tile[threadIdx.y + j][threadIdx.x] = src[(y + j) * width + x];
        }
    }

    __syncthreads();

    x = blockIdx.y * kTransposeTileDim + threadIdx.x;
    y = blockIdx.x * kTransposeTileDim + threadIdx.y;
    for (int j = 0; j < kTransposeTileDim; j += kTransposeBlockRows) {
        if (x < height && (y + j) < width) {
            dst[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

std::vector<float2> CreateTwiddleLut(int row_length) {
    std::vector<float2> lut(static_cast<size_t>(row_length / 2));
    for (int i = 0; i < row_length / 2; ++i) {
        const double angle = 2.0 * kPi * static_cast<double>(i) / static_cast<double>(row_length);
        lut[static_cast<size_t>(i)] = make_float2(
            static_cast<float>(std::cos(angle)),
            static_cast<float>(std::sin(angle)));
    }
    return lut;
}

void UploadTwiddleLut(const std::vector<float2>& lut, float2** device_ptr) {
    CUDA_CHECK(cudaMalloc(device_ptr, lut.size() * sizeof(float2)));
    CUDA_CHECK(cudaMemcpy(*device_ptr, lut.data(), lut.size() * sizeof(float2), cudaMemcpyHostToDevice));
}

void LaunchTranspose(const float2* src, float2* dst, int width, int height) {
    const dim3 block(kTransposeTileDim, kTransposeBlockRows);
    const dim3 grid(
        static_cast<unsigned int>((width + kTransposeTileDim - 1) / kTransposeTileDim),
        static_cast<unsigned int>((height + kTransposeTileDim - 1) / kTransposeTileDim));
    TransposeKernel<<<grid, block>>>(src, dst, width, height);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace

GpuFftPlan::GpuFftPlan(int width, int height)
    : width_(width),
      height_(height),
      log2_width_(IntegerLog2(width)),
      log2_height_(IntegerLog2(height)) {
    const size_t element_count = static_cast<size_t>(width_) * static_cast<size_t>(height_);
    CUDA_CHECK(cudaMalloc(&d_real_input_, element_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_buffer_a_, element_count * sizeof(float2)));
    CUDA_CHECK(cudaMalloc(&d_buffer_b_, element_count * sizeof(float2)));
    CUDA_CHECK(cudaMalloc(&d_multiply_buffer_, element_count * sizeof(float2)));

    const std::vector<float2> width_lut = CreateTwiddleLut(width_);
    const std::vector<float2> height_lut = CreateTwiddleLut(height_);
    UploadTwiddleLut(width_lut, &d_lut_width_);
    UploadTwiddleLut(height_lut, &d_lut_height_);
}

GpuFftPlan::~GpuFftPlan() {
    cudaFree(d_lut_height_);
    cudaFree(d_lut_width_);
    cudaFree(d_multiply_buffer_);
    cudaFree(d_buffer_b_);
    cudaFree(d_buffer_a_);
    cudaFree(d_real_input_);
}

void GpuFftPlan::UploadRealInput(const std::vector<float>& image) {
    const size_t element_count = static_cast<size_t>(width_) * static_cast<size_t>(height_);
    if (image.size() != element_count) {
        throw std::runtime_error("Image size does not match GPU FFT dimensions");
    }
    CUDA_CHECK(cudaMemcpy(d_real_input_, image.data(), element_count * sizeof(float), cudaMemcpyHostToDevice));
}

void GpuFftPlan::UploadSpectrumToPrimary(const std::vector<float2>& spectrum) {
    const size_t element_count = static_cast<size_t>(width_) * static_cast<size_t>(height_);
    if (spectrum.size() != element_count) {
        throw std::runtime_error("Spectrum size does not match GPU FFT dimensions");
    }
    CUDA_CHECK(cudaMemcpy(d_buffer_a_, spectrum.data(), element_count * sizeof(float2), cudaMemcpyHostToDevice));
}

std::vector<float2> GpuFftPlan::DownloadPrimarySpectrum() const {
    const size_t element_count = static_cast<size_t>(width_) * static_cast<size_t>(height_);
    std::vector<float2> spectrum(element_count);
    CUDA_CHECK(cudaMemcpy(spectrum.data(), d_buffer_a_, element_count * sizeof(float2), cudaMemcpyDeviceToHost));
    return spectrum;
}

std::vector<float> GpuFftPlan::DownloadPrimaryReal() const {
    const size_t element_count = static_cast<size_t>(width_) * static_cast<size_t>(height_);
    std::vector<float2> reconstructed_complex(element_count);
    CUDA_CHECK(cudaMemcpy(
        reconstructed_complex.data(),
        d_buffer_a_,
        element_count * sizeof(float2),
        cudaMemcpyDeviceToHost));
    return RealFromComplex(reconstructed_complex);
}

void GpuFftPlan::MultiplyPrimaryByHostSpectrum(const std::vector<float2>& other) {
    const size_t element_count = static_cast<size_t>(width_) * static_cast<size_t>(height_);
    if (other.size() != element_count) {
        throw std::runtime_error("Spectrum size does not match GPU FFT dimensions");
    }
    CUDA_CHECK(cudaMemcpy(
        d_multiply_buffer_,
        other.data(),
        element_count * sizeof(float2),
        cudaMemcpyHostToDevice));
    const int blocks = (static_cast<int>(element_count) + kThreadsPerBlock - 1) / kThreadsPerBlock;
    HadamardMultiplyKernel<<<blocks, kThreadsPerBlock>>>(
        d_buffer_a_,
        d_multiply_buffer_,
        static_cast<int>(element_count));
    CUDA_CHECK(cudaGetLastError());
}

void GpuFftPlan::RunRowPass(
    const float2* src,
    float2* dst,
    int row_length,
    int row_count,
    bool inverse,
    GpuTwiddleMode mode,
    const float2* lut) {
    const int element_count = row_length * row_count;
    const int blocks = (element_count + kThreadsPerBlock - 1) / kThreadsPerBlock;
    const int log2_length = IntegerLog2(row_length);
    BitReverseRowsKernel<<<blocks, kThreadsPerBlock>>>(src, dst, row_length, row_count, log2_length);
    CUDA_CHECK(cudaGetLastError());
    LaunchStages(dst, row_length, row_count, inverse, mode, lut);
}

void GpuFftPlan::LaunchStages(
    float2* buffer,
    int row_length,
    int row_count,
    bool inverse,
    GpuTwiddleMode mode,
    const float2* lut) {
    const int total_butterflies = row_count * (row_length / 2);
    const int blocks = (total_butterflies + kThreadsPerBlock - 1) / kThreadsPerBlock;
    const int stage_count = IntegerLog2(row_length);
    for (int stage = 0; stage < stage_count; ++stage) {
        if (mode == GpuTwiddleMode::Lut) {
            FftStageRowsKernel<LutTwiddleProvider><<<blocks, kThreadsPerBlock>>>(
                buffer,
                row_length,
                row_count,
                stage,
                inverse,
                lut);
        } else {
            FftStageRowsKernel<FastTwiddleProvider><<<blocks, kThreadsPerBlock>>>(
                buffer,
                row_length,
                row_count,
                stage,
                inverse,
                lut);
        }
        CUDA_CHECK(cudaGetLastError());
    }
}

void GpuFftPlan::RunForward(GpuTwiddleMode mode) {
    const int element_count = width_ * height_;
    const int blocks = (element_count + kThreadsPerBlock - 1) / kThreadsPerBlock;
    RealToComplexKernel<<<blocks, kThreadsPerBlock>>>(d_real_input_, d_buffer_a_, element_count);
    CUDA_CHECK(cudaGetLastError());

    RunRowPass(d_buffer_a_, d_buffer_b_, width_, height_, false, mode, d_lut_width_);
    LaunchTranspose(d_buffer_b_, d_buffer_a_, width_, height_);
    RunRowPass(d_buffer_a_, d_buffer_b_, height_, width_, false, mode, d_lut_height_);
    LaunchTranspose(d_buffer_b_, d_buffer_a_, height_, width_);
}

void GpuFftPlan::RunInverse(GpuTwiddleMode mode) {
    RunRowPass(d_buffer_a_, d_buffer_b_, width_, height_, true, mode, d_lut_width_);
    LaunchTranspose(d_buffer_b_, d_buffer_a_, width_, height_);
    RunRowPass(d_buffer_a_, d_buffer_b_, height_, width_, true, mode, d_lut_height_);
    LaunchTranspose(d_buffer_b_, d_buffer_a_, height_, width_);

    const int element_count = width_ * height_;
    const int blocks = (element_count + kThreadsPerBlock - 1) / kThreadsPerBlock;
    const float scale = 1.0f / static_cast<float>(element_count);
    ScaleComplexKernel<<<blocks, kThreadsPerBlock>>>(d_buffer_a_, scale, element_count);
    CUDA_CHECK(cudaGetLastError());
}

std::vector<float2> GpuFftPlan::Forward(const std::vector<float>& image, GpuTwiddleMode mode) {
    UploadRealInput(image);
    RunForward(mode);
    CUDA_CHECK(cudaDeviceSynchronize());
    return DownloadPrimarySpectrum();
}

std::vector<float> GpuFftPlan::Inverse(const std::vector<float2>& spectrum, GpuTwiddleMode mode) {
    UploadSpectrumToPrimary(spectrum);
    RunInverse(mode);
    CUDA_CHECK(cudaDeviceSynchronize());
    return DownloadPrimaryReal();
}

std::vector<float> GpuFftPlan::Convolve(
    const std::vector<float>& image,
    const std::vector<float2>& kernel_spectrum,
    GpuTwiddleMode mode) {
    UploadRealInput(image);
    RunForward(mode);
    MultiplyPrimaryByHostSpectrum(kernel_spectrum);
    RunInverse(mode);
    CUDA_CHECK(cudaDeviceSynchronize());
    return DownloadPrimaryReal();
}

GpuFftResult GpuFftPlan::Run(const std::vector<float>& image, GpuTwiddleMode mode) {
    GpuFftResult result;
    result.spectrum = Forward(image, mode);
    result.reconstruction = Inverse(result.spectrum, mode);
    return result;
}

}  // namespace fftlut
