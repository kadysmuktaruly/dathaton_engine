# Eval `ft_held8_main`

## Summary

| metric | value |
|---|---|
| videos | 8 |
| tp | 3 |
| fp | 0 |
| fn | 2 |
| tn | 3 |
| precision | 1.0 |
| recall | 0.6 |
| f1 | 0.75 |
| timecodes_total | 5 |
| timecodes_matched | 3 |
| err_median | 2.3 |
| err_max | 5.5 |
| within_1s | 0.0 |
| within_2s | 0.333 |
| win_err_median | 1.3 |
| type_acc | 0.667 |
| time_mean_s | 20.18 |
| time_max_s | 23.96 |
| peak_rss_gb | 2.38 |

## Per confirmed code

| code | labels | video hit | right type | errors (s) |
|---|---|---|---|---|
| B1 | 1 | 0 | 0 | [] |
| B5 | 1 | 1 | 0 | [5.5] |
| B7 | 1 | 0 | 0 | [] |
| C1 | 1 | 1 | 1 | [2.3] |
| C2 | 1 | 1 | 1 | [1.7] |

## Misses (violation clips predicted clean)

- X-13 labels [47.0] want ['B7']; overlay: runs/overlays/X-13.mp4
- X-38 labels [33.0] want ['B1']; overlay: runs/overlays/X-38.mp4

## False positives (clean clips predicted violation)


## Per video

| clip | label | pred | preds (t, code) | label t | want | time s | perception | rules | vlm |
|---|---|---|---|---|---|---|---|---|---|
| X-13 | V | - | [] | [47.0] | ['B7'] | 23.96 | 4.37 | 0.71 | 18.08 |
| X-18 | V | V | [(17.7, 'C1')] | [20.0] | ['C1'] | 23.21 | 0.14 | 0.96 | 21.9 |
| X-19 | - | - | [] | [] | [] | 18.29 | 0.15 | 0.72 | 17.42 |
| X-24 | V | V | [(2.5, 'X0')] | [8.0] | ['B5'] | 22.92 | 0.14 | 0.21 | 22.52 |
| X-29 | - | - | [] | [] | [] | 18.51 | 0.13 | 0.23 | 18.15 |
| X-33 | V | V | [(4.3, 'C2')] | [6.0] | ['C2'] | 17.45 | 0.14 | 0.4 | 16.83 |
| X-38 | V | - | [] | [33.0] | ['B1'] | 19.07 | 0.14 | 0.66 | 18.27 |
| X-45 | - | - | [] | [] | [] | 18.06 | 0.15 | 0.38 | 17.53 |
