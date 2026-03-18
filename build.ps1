param(
    [string]$Configuration = "Release"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$buildDir = Join-Path $root "build"
New-Item -ItemType Directory -Force -Path $buildDir | Out-Null

$nvcc = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe"
if (-not (Test-Path $nvcc)) {
    throw "nvcc not found at $nvcc"
}

$hostFlags = "/EHsc /W3 /nologo"
$commonFlags = @(
    "-std=c++17",
    "-O3",
    "-arch=native",
    "-Xcompiler", $hostFlags,
    "-I$root\src",
    "-I$root\third_party"
)

function Build-Target {
    param(
        [string]$Output,
        [string[]]$Sources
    )

    Write-Host "Building $Output"
    & $nvcc @commonFlags "-o" $Output @Sources
    if ($LASTEXITCODE -ne 0) {
        throw "nvcc build failed with exit code $LASTEXITCODE"
    }
}

$sharedSources = @(
    (Join-Path $root "src\images.cpp"),
    (Join-Path $root "src\cpu_fft.cpp"),
    (Join-Path $root "src\metrics.cpp"),
    (Join-Path $root "src\output.cpp"),
    (Join-Path $root "src\gpu_fft.cu"),
    (Join-Path $root "src\exr_io.cpp"),
    (Join-Path $root "src\hdr_scene.cpp"),
    (Join-Path $root "src\bloom.cpp"),
    (Join-Path $root "third_party\miniz.c"),
    (Join-Path $root "third_party\miniz_tdef.c"),
    (Join-Path $root "third_party\miniz_tinfl.c")
)

$roundTripSources = @(
    (Join-Path $root "src\main.cpp")
) + $sharedSources

$bloomDebugSources = @(
    (Join-Path $root "src\bloom_debug_main.cpp")
) + $sharedSources

$bloomMeasureSources = @(
    (Join-Path $root "src\bloom_measure_main.cpp")
) + $sharedSources

$roundTripOutput = Join-Path $buildDir "fftlut_experiment.exe"
$bloomDebugOutput = Join-Path $buildDir "bloom_debug.exe"
$bloomMeasureOutput = Join-Path $buildDir "bloom_measure.exe"

Build-Target -Output $roundTripOutput -Sources $roundTripSources
Build-Target -Output $bloomDebugOutput -Sources $bloomDebugSources
Build-Target -Output $bloomMeasureOutput -Sources $bloomMeasureSources

Write-Host "Build complete: $roundTripOutput"
Write-Host "Build complete: $bloomDebugOutput"
Write-Host "Build complete: $bloomMeasureOutput"
