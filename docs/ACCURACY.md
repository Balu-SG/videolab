# Accuracy benchmark

20 synthetic 240 fps clips with exact ground truth, run through the same pipeline as the app: SAM 3.1 tracking (concept prompt, then a box re-prompt), A4 sheet detection and homography, then the physics fit. Conditions vary the camera angle and distance, lighting, sensor noise and motion blur. Reproduce with `scripts/benchmark.py`; the clips come from `scripts/synth.py`.

SAM prompt: `text` is the colour-and-shape concept (e.g. "red ball") followed by a box re-prompt; `tap` is the fallback when the concept finds nothing, a point prompt on frame 0 as a user's tap in the app.

These are rendered scenes, not real footage. They test the measurement chain against known answers; real-phone results are reported separately.

## Summary

| Experiment | Runs | Mean abs. error | Worst error | Within 2σ |
|---|---|---|---|---|
| pendulum (g) | 8 | 1.59% | 3.74% | 8/8 |
| free fall (g) | 6 | 0.75% | 1.31% | 6/6 |
| spring (k) | 6 | 0.02% | 0.03% | 6/6 |
| **all** | 20 | 0.87% | 3.74% | 20/20 |

## Runs

| Run | Setup | Condition | SAM prompt | True | Measured | Error | z | Tracked frames |
|---|---|---|---|---|---|---|---|---|
| p1 | L = 0.3 m, 10° | baseline | text | g = 9.81 m/s² | 10.170 ± 0.737 | +3.67% | +0.5 | 1440/1440 |
| p2 | L = 0.45 m, 15° | baseline | text | g = 9.81 m/s² | 9.751 ± 0.213 | -0.61% | -0.3 | 1440/1440 |
| p3 | L = 0.6 m, 8° | tilted camera | text | g = 9.81 m/s² | 9.443 ± 0.470 | -3.74% | -0.8 | 1440/1440 |
| p4 | L = 0.75 m, 20° | baseline | text | g = 9.81 m/s² | 9.858 ± 0.095 | +0.49% | +0.5 | 1440/1440 |
| p5 | L = 0.9 m, 12° | far camera | text | g = 9.81 m/s² | 9.651 ± 0.259 | -1.62% | -0.6 | 1440/1440 |
| p6 | L = 0.5 m, 12° | dim + noisy | text | g = 9.81 m/s² | 9.717 ± 0.289 | -0.94% | -0.3 | 1440/1440 |
| p7 | L = 0.5 m, 15° | motion blur | text | g = 9.81 m/s² | 9.762 ± 0.194 | -0.49% | -0.2 | 1440/1440 |
| p8 | L = 0.6 m, 10° | tilted camera, dim + noisy | text | g = 9.81 m/s² | 9.697 ± 0.332 | -1.15% | -0.3 | 1440/1440 |
| f1 | drop 0.6 m | baseline | text | g = 9.81 m/s² | 9.755 ± 0.064 | -0.56% | -0.8 | 202/202 |
| f2 | drop 0.8 m | tilted camera | text | g = 9.81 m/s² | 9.755 ± 0.063 | -0.57% | -0.9 | 215/215 |
| f3 | drop 1.0 m | far camera | tap | g = 9.81 m/s² | 9.682 ± 0.100 | -1.31% | -1.3 | 227/227 |
| f4 | drop 1.05 m | motion blur | tap | g = 9.81 m/s² | 9.741 ± 0.064 | -0.70% | -1.1 | 229/229 |
| f5 | drop 0.9 m | dim + noisy | tap | g = 9.81 m/s² | 9.737 ± 0.064 | -0.74% | -1.1 | 221/221 |
| f6 | drop 0.7 m | baseline | text | g = 9.81 m/s² | 9.750 ± 0.064 | -0.62% | -0.9 | 209/209 |
| s1 | m = 0.2 kg, k = 10 N/m | baseline | text | k = 10 N/m | 9.997 ± 0.010 | -0.03% | -0.3 | 960/960 |
| s2 | m = 0.2 kg, k = 15 N/m | tilted camera | text | k = 15 N/m | 14.997 ± 0.015 | -0.02% | -0.2 | 960/960 |
| s3 | m = 0.2 kg, k = 20 N/m | far camera | text | k = 20 N/m | 19.998 ± 0.020 | -0.01% | -0.1 | 960/960 |
| s4 | m = 0.3 kg, k = 30 N/m | motion blur | text | k = 30 N/m | 29.996 ± 0.030 | -0.01% | -0.1 | 960/960 |
| s5 | m = 0.25 kg, k = 40 N/m | dim + noisy | text | k = 40 N/m | 39.999 ± 0.040 | -0.00% | -0.0 | 960/960 |
| s6 | m = 0.15 kg, k = 25 N/m | baseline | text | k = 25 N/m | 24.993 ± 0.025 | -0.03% | -0.3 | 960/960 |

## Pendulum with the string length entered

From the swing alone, the length comes from the arc's curvature, which is weak for short or narrow swings (p1, p3), and the error bar says so. The period is always precise, so typing in the measured string length fixes it. Below, the same tracks with the true length entered; a real tape measure adds about ±2 mm, which the error bar includes.

| Run | Measured g | Error |
|---|---|---|
| p1 | 9.808 ± 0.066 | -0.02% |
| p2 | 9.810 ± 0.045 | +0.00% |
| p3 | 9.811 ± 0.034 | +0.01% |
| p4 | 9.809 ± 0.028 | -0.01% |
| p5 | 9.809 ± 0.024 | -0.01% |
| p6 | 9.811 ± 0.040 | +0.01% |
| p7 | 9.810 ± 0.040 | +0.00% |
| p8 | 9.810 ± 0.034 | +0.00% |
