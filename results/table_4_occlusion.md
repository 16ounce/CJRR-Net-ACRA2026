# Table 4 — Error by Current-View Occlusion

Values are selected-view 2D MPJPE in pixels. Five rows are checked against archived results. The shared-input MLP row is transcribed from the Camera Ready PDF because its per-decision occlusion results are unavailable here.
Uniform random uses the exact within-bin expectation as its centre and the sample standard deviation of ten seeded draws.

| Policy | Low (0–2) | Medium (3–5) | High (≥6) | Evidence |
|---|---:|---:|---:|---|
| Uniform random | 247.22 ± 6.69 | 275.54 ± 9.99 | 401.93 ± 11.23 | archived result JSON |
| Geometric utility | 162.45 ± 0.00 | 193.74 ± 0.00 | 253.63 ± 0.00 | archived result JSON |
| ActiveMoCap-inspired planner | 152.13 ± 18.48 | 163.85 ± 12.90 | 242.83 ± 8.80 | archived result JSON |
| PoseErrNet (adapted) | 97.87 ± 2.89 | 109.73 ± 2.91 | 154.37 ± 11.65 | archived result JSON |
| MLP with Shared Perception Inputs | 98.51 ± 3.31 | 102.82 ± 2.18 | 151.29 ± 6.79 | paper reported; raw decisions unavailable |
| CJRR-Net | 92.29 ± 2.74 | 99.24 ± 2.16 | 130.24 ± 5.51 | archived result JSON |
