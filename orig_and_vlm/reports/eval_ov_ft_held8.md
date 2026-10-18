# Eval `ov_ft_held8`

## Summary

| metric | value |
|---|---|
| videos | 8 |
| tp | 4 |
| fp | 2 |
| fn | 1 |
| tn | 1 |
| precision | 0.667 |
| recall | 0.8 |
| f1 | 0.727 |
| timecodes_total | 5 |
| timecodes_matched | 4 |
| err_median | 2.0 |
| err_max | 3.3 |
| within_1s | 0.25 |
| within_2s | 0.5 |
| win_err_median | 1.0 |
| type_acc | 1.0 |
| time_mean_s | 33.17 |
| time_max_s | 49.78 |
| peak_rss_gb | 3.0 |

## Per confirmed code

| code | labels | video hit | right type | errors (s) |
|---|---|---|---|---|
| B1 | 1 | 1 | 1 | [3.3] |
| B5 | 1 | 1 | 1 | [1.0] |
| B7 | 1 | 0 | 0 | [] |
| C1 | 1 | 1 | 1 | [2.3] |
| C2 | 1 | 1 | 1 | [1.7] |

## Misses (violation clips predicted clean)

- X-13 labels [47.0] want ['B7']; overlay: runs/overlays/X-13.mp4

## False positives (clean clips predicted violation)

- X-19 preds [(13.7, 'B1')] events [('B1', 'cash_open_new_client')]; overlay: runs/overlays/X-19.mp4
- X-29 preds [(6.3, 'B5')] events [('B5', 'cashier_left_room')]; overlay: runs/overlays/X-29.mp4

## Per video

| clip | label | pred | preds (t, code) | label t | want | time s | perception | rules | vlm |
|---|---|---|---|---|---|---|---|---|---|
| X-13 | V | - | [] | [47.0] | ['B7'] | 49.78 | 4.24 | 0.69 | 44.04 |
| X-18 | V | V | [(13.7, 'B1'), (17.7, 'C1')] | [20.0] | ['C1'] | 38.58 | 0.15 | 0.97 | 37.08 |
| X-19 | - | V | [(13.7, 'B1')] | [] | [] | 37.96 | 0.18 | 0.75 | 36.86 |
| X-24 | V | V | [(9.0, 'B5')] | [8.0] | ['B5'] | 27.11 | 0.13 | 0.2 | 26.59 |
| X-29 | - | V | [(6.3, 'B5')] | [] | [] | 26.36 | 0.17 | 0.28 | 25.81 |
| X-33 | V | V | [(4.3, 'C2')] | [6.0] | ['C2'] | 16.75 | 0.17 | 0.49 | 16.0 |
| X-38 | V | V | [(36.3, 'B1')] | [33.0] | ['B1'] | 40.39 | 0.17 | 0.63 | 39.24 |
| X-45 | - | - | [] | [] | [] | 28.44 | 0.14 | 0.36 | 27.94 |
