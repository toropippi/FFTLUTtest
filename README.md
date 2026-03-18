# FFT LUT vs Approximate Trig Comparison

This project generates synthetic grayscale test images, runs `2D FFT -> inverse FFT` in three modes, and measures how the reconstructed image and complex spectrum differ:

- `cpu_ref`: CPU double-precision reference FFT/IFFT
- `gpu_lut`: CUDA FFT/IFFT using GPU twiddle factors from a CPU-generated LUT
- `gpu_fast`: CUDA FFT/IFFT using GPU fast approximate trig via `__sincosf`

The experiment is designed so that **the only intentional difference between `gpu_lut` and `gpu_fast` is the twiddle-factor implementation**. FFT stage order, radix-2 butterfly structure, memory layout, transpose path, normalization, image generation, and output handling are otherwise shared.

## Repository Layout

This repo is intentionally kept small so it can be rebuilt and re-run later with AI assistance.

- `src/`
  - Main C++ / CUDA sources.
  - `main.cpp`: grayscale FFT round-trip experiment entrypoint.
  - `bloom_debug_main.cpp`: bloom debug runner for inspecting one kernel/image case.
  - `bloom_measure_main.cpp`: production-style bloom measurement runner for one kernel/image case.
  - `images.*`: synthetic grayscale test-pattern generators.
  - `hdr_scene.*`: deterministic RGB HDR source generation with bright spots.
  - `cpu_fft.*`: CPU double-precision FFT reference.
  - `gpu_fft.*`: CUDA radix-2 FFT implementation with LUT and fast trig modes.
  - `bloom.*`: RGB bloom convolution helpers, kernel `ifftshift`, and self-test.
  - `metrics.*`: numerical error metrics.
  - `output.*`: PNG / NPY / JSON / CSV output helpers.
  - `exr_io.*`: EXR loading plus raw float RGB `.bin` loading.
- `scripts/`
  - Utility scripts used after experiments.
  - `plot_figures.py`: paper-support figure generation from experiment outputs.
- `third_party/`
  - Vendored single-header or small dependency sources used by the native tools.
  - Includes `TinyEXR`, `stb_image_write`, and `miniz`.
- `build.ps1`
  - Rebuilds the native executables from source.
- `README.md`
  - Project overview, build/run instructions, and folder descriptions.

Generated folders such as `build/` and `output*/` are intentionally not tracked in git. Recreate them locally when needed.

Kernel binaries are expected to come from the separate diffraction project, for example under `C:\Users\5080\Documents\GitHub\DiffractionGPU\source`, and are also not tracked in this repo.

## Build

Requirements:

- Windows
- CUDA 12.9 with `nvcc`
- Visual Studio Build Tools with MSVC host compiler
- Python 3 with `numpy`, `pandas`, `matplotlib`, and `Pillow` for figure generation

Build the native experiment runner:

```powershell
.\build.ps1
```

This produces:

```text
build/fftlut_experiment.exe
build/bloom_debug.exe
build/bloom_measure.exe
```

## Run

Single case:

```powershell
.\build\fftlut_experiment.exe `
  --width 512 `
  --height 512 `
  --image-type checkerboard `
  --variant 1 `
  --seed 1234 `
  --output-dir output_single `
  --save-images 1 `
  --save-spectrum 1
```

Batch mode:

```powershell
.\build\fftlut_experiment.exe --run-all --output-dir output
```

`--run-all` executes:

- all registered presets at `256x256`
- representative larger cases at `512x512` and `1024x1024` for:
  - `horizontal_gradient`
  - `sharp_edge_vertical`
  - `text_small`
  - `game_like_scene_simple`

## FFT Implementation Notes

- Data type: input images are `float32` grayscale in `[0, 1]`
- Supported sizes: power-of-two widths and heights only
- 2D FFT structure: row FFT -> transpose -> row FFT -> transpose
- Inverse uses the same structure
- Normalization rule: only inverse applies scaling, with factor `1 / (width * height)`
- CPU reference uses double precision for both forward and inverse transforms
- GPU LUT twiddles are computed on CPU in double precision, cast to float, then uploaded once per dimension
- GPU fast twiddles are computed in-kernel with `__sincosf`

## Synthetic Image Set

Implemented generators:

1. `horizontal_gradient`
2. `vertical_gradient`
3. `radial_gradient`
4. `sharp_edge_vertical`
5. `sharp_edge_horizontal`
6. `checkerboard`
7. `impulse`
8. `stripes_sine`
9. `sine_sum`
10. `text_large`
11. `text_small`
12. `random_noise_uniform`
13. `random_noise_gaussian_like`
14. `game_like_scene_simple`
15. `edge_enhanced_scene`

Multi-variant presets are included for checkerboards, impulses, sine stripes, sine sums, small text, and game-like scenes. Text uses a built-in bitmap font with no antialiasing for deterministic edge behavior.

## Output Layout

Example output tree:

```text
output/
  summary.csv
  summary.json
  case_<image>_<variant>_<w>x<h>/
    original.png
    recon_cpu_double.png
    recon_gpu_lut.png
    recon_gpu_fast.png
    absdiff_lut_vs_ref.png
    absdiff_lut_vs_ref_log.png
    absdiff_lut_vs_ref.npy
    absdiff_fast_vs_ref.png
    absdiff_fast_vs_ref_log.png
    absdiff_fast_vs_ref.npy
    absdiff_fast_vs_lut.png
    absdiff_fast_vs_lut_log.png
    absdiff_fast_vs_lut.npy
    spectrum_ref_log.png
    spectrum_lut_log.png
    spectrum_fast_log.png
    spectrum_absdiff_fast_vs_ref_log.png
    spectrum_absdiff_fast_vs_ref.npy
    spectrum_absdiff_lut_vs_ref.npy
    metrics.json
```

Notes:

- `absdiff_*.png` are linearly normalized per image for visibility
- `absdiff_*_log.png` use log emphasis for small differences
- `.npy` files store raw `float32` values and should be used for numerical analysis
- FFT spectrum visualizations are FFT-shifted `log1p(|F|)` images

## Metrics

Each case records:

- `mse_vs_ref`
- `rmse_vs_ref`
- `mae_vs_ref`
- `max_abs_error`
- `psnr_vs_ref`
- `relative_l2_error`
- `mean_abs_error_in_spectrum`
- `max_abs_error_in_spectrum`

It also records direct reconstruction comparison between `gpu_fast` and `gpu_lut`:

- `mse_fast_vs_lut`
- `mae_fast_vs_lut`
- `max_abs_fast_vs_lut`
- `relative_l2_fast_vs_lut`

Per-case JSON additionally stores sanity metrics against the original input image so forward+inverse correctness is explicit.

## Figure Generation

Generate the paper-support figures after a batch run:

```powershell
python .\scripts\plot_figures.py --input-dir output
```

This writes:

- `figure_a_comparison_grid.png`
- `figure_b_metrics.png`
- `figure_c_spectrum_example.png`
- `figure_d_error_heatmaps.png`

Figure conventions:

- Figure A uses one representative preset per image type
- Figure B compares `gpu_lut` and `gpu_fast` bars on representative presets
- Figure C defaults to the case with the largest spectrum max error, unless overridden
- Figure D shows `absdiff_fast_vs_lut_log` for representative gradient / edge / text / game-like cases

## Debugging Recommendation

To validate correctness first, start with:

- `horizontal_gradient` at `256x256`
- `impulse` at `256x256`

Those cases make normalization mistakes and indexing mistakes easy to see before running the full batch.

## Fixed Bloom Debug Tool

Stage 1 also includes a separate FFT convolution bloom debugger for square power-of-two RGB HDR scenes:

```powershell
.\build\bloom_debug.exe `
  --kernel path\to\kernel_1024_or_2048.bin `
  --output-dir output_bloom_debug `
  --seed 1337
```

Behavior:

- loads a user-supplied square power-of-two RGB kernel from either:
  - raw `float32` RGB `.bin` with square power-of-two dimensions inferred from file size
  - RGB `.exr`
- applies `ifftshift` to the kernel before FFT so the center pixel is treated as the convolution origin
- generates a deterministic RGB HDR source with bright point-like highlights
- computes raw circular-convolution bloom outputs for:
  - `cpu_ref`
  - `gpu_lut`
  - `gpu_fast`
- writes raw HDR `.npy` arrays, tone-mapped PNG previews, luminance diff heatmaps, and spot metadata JSON

## Bloom Measurement Runner

The repo also includes a production-style single-case bloom measurement runner:

```powershell
.\build\bloom_measure.exe `
  --kernel path\to\kernel_1024.bin `
  --output-dir output_bloom_measure `
  --seed 1337
```

It uses the same RGB HDR source generator and convolution path as `bloom_debug.exe`, but writes:

- raw HDR source/kernel/output arrays
- tone-mapped PNGs
- luminance diff maps as `.npy` and PNG
- `metrics.json`, `summary.csv`, and `summary.json` for CPU/LUT/Fast comparisons
