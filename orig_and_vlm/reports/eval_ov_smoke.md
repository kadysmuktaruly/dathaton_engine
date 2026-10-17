# Eval `ov_smoke`

## Summary

| metric | value |
|---|---|
| videos | 8 |
| tp | 3 |
| fp | 1 |
| fn | 2 |
| tn | 2 |
| precision | 0.75 |
| recall | 0.6 |
| f1 | 0.667 |
| timecodes_total | 6 |
| timecodes_matched | 3 |
| err_median | 1.0 |
| err_max | 1.0 |
| within_1s | 1.0 |
| within_2s | 1.0 |
| win_err_median | 0.0 |
| type_acc | 1.0 |
| time_mean_s | 37.2 |
| time_max_s | 56.47 |
| peak_rss_gb | 1.74 |

## Per confirmed code

| code | labels | video hit | right type | errors (s) |
|---|---|---|---|---|
| A2 | 1 | 0 | 0 | [] |
| B1 | 3 | 1 | 1 | [1.0] |
| B5 | 2 | 2 | 2 | [1.0, 1.0] |

## Misses (violation clips predicted clean)

- X-4 labels [57.0, 68.0] want ['B1', 'A2']; overlay: runs/overlays/X-4.mp4
- X-38 labels [33.0] want ['B1']; overlay: runs/overlays/X-38.mp4

## False positives (clean clips predicted violation)

- X-29 preds [(6.3, 'B5')] events [('B5', 'cashier_left_room')]; overlay: runs/overlays/X-29.mp4

## Per video

| clip | label | pred | preds (t, code) | label t | want | time s | perception | rules | vlm |
|---|---|---|---|---|---|---|---|---|---|
| X-4 | V | - | [] | [57.0, 68.0] | ['B1', 'A2'] | 56.47 | 28.91 | 0.86 | 26.06 |
| X-12 | - | - | [] | [] | [] | 41.75 | 11.64 | 0.45 | 29.66 |
| X-23 | - | - | [] | [] | [] | 23.9 | 3.82 | 0.23 | 19.84 |
| X-24 | V | V | [(2.5, 'X0'), (9.0, 'B5')] | [8.0] | ['B5'] | 32.85 | 3.25 | 0.2 | 29.19 |
| X-26 | V | V | [(8.0, 'B1')] | [9.0] | ['B1'] | 31.48 | 7.6 | 0.27 | 23.51 |
| X-28 | V | V | [(6.0, 'B5'), (12.7, 'B5')] | [7.0] | ['B5'] | 35.59 | 5.7 | 0.22 | 29.42 |
| X-29 | - | V | [(6.3, 'B5')] | [] | [] | 30.11 | 4.23 | 0.23 | 25.53 |
| X-38 | V | - | [] | [33.0] | ['B1'] | 45.41 | 19.04 | 0.65 | 25.72 |
