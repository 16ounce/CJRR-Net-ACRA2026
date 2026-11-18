# Table 1 — Overall Future-View Selection Performance

Values are reconstructed from the archived result JSON files and checked against the Camera Ready PDF.
For uniform random, the centre is the exact expectation; the standard deviation comes from ten seeded draws.
Relative reduction = (299.7563 − selected-view MPJPE) / 299.7563 × 100%.

| Policy | MPJPE (px) ↓ | Relative reduction (%) ↑ | Tol@15 (%) ↑ | 0-Recall ↓ |
|---|---:|---:|---:|---:|
| Uniform random | 299.76 ± 5.27 | 0.00 ± 1.76 | 26.25 ± 0.67 | 0.00 ± 0.00 |
| Geometric utility | 198.30 ± 0.00 | 33.85 ± 0.00 | 33.35 ± 0.00 | 0.00 ± 0.00 |
| ActiveMoCap-inspired planner | 181.29 ± 13.55 | 39.52 ± 4.52 | 40.37 ± 4.02 | 0.00 ± 0.00 |
| PoseErrNet (adapted) | 117.56 ± 5.02 | 60.78 ± 1.68 | 53.81 ± 0.45 | 6.30 ± 0.48 |
| MLP with shared perception inputs | 114.64 ± 3.42 | 61.76 ± 1.14 | 55.15 ± 0.80 | 2.50 ± 1.18 |
| CJRR-Net (ours) | 105.18 ± 2.61 | 64.91 ± 0.87 | 58.17 ± 0.59 | 0.00 ± 0.00 |
