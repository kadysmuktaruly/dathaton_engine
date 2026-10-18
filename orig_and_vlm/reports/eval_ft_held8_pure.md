# Eval `ft_held8_pure`

## Summary

| metric | value |
|---|---|
| videos | 8 |
| tp | 2 |
| fp | 0 |
| fn | 3 |
| tn | 3 |
| precision | 1.0 |
| recall | 0.4 |
| f1 | 0.571 |
| timecodes_total | 5 |
| timecodes_matched | 2 |
| err_median | 2.0 |
| err_max | 2.3 |
| within_1s | 0.0 |
| within_2s | 0.5 |
| win_err_median | 1.0 |
| type_acc | 1.0 |
| time_mean_s | 8.73 |
| time_max_s | 18.74 |
| peak_rss_gb | 0.73 |

## Per confirmed code

| code | labels | video hit | right type | errors (s) |
|---|---|---|---|---|
| B1 | 1 | 0 | 0 | [] |
| B5 | 1 | 0 | 0 | [] |
| B7 | 1 | 0 | 0 | [] |
| C1 | 1 | 1 | 1 | [2.3] |
| C2 | 1 | 1 | 1 | [1.7] |

## Misses (violation clips predicted clean)

- X-13 labels [47.0] want ['B7']; overlay: runs/overlays/X-13.mp4
- X-24 labels [8.0] want ['B5']; overlay: runs/overlays/X-24.mp4
- X-38 labels [33.0] want ['B1']; overlay: runs/overlays/X-38.mp4

## False positives (clean clips predicted violation)


## Per video

| clip | label | pred | preds (t, code) | label t | want | time s | perception | rules | vlm |
|---|---|---|---|---|---|---|---|---|---|
| X-13 | V | - | [] | [47.0] | ['B7'] | 18.74 | 17.31 | 0.69 | 0.0 |
| X-18 | V | V | [(17.7, 'C1')] | [20.0] | ['C1'] | 6.17 | 5.14 | 0.84 | 0.0 |
| X-19 | - | - | [] | [] | [] | 14.33 | 13.44 | 0.89 | 0.0 |
| X-24 | V | - | [] | [8.0] | ['B5'] | 3.19 | 2.99 | 0.19 | 0.0 |
| X-29 | - | - | [] | [] | [] | 3.33 | 3.1 | 0.23 | 0.0 |
| X-33 | V | V | [(4.3, 'C2')] | [6.0] | ['C2'] | 3.17 | 2.69 | 0.41 | 0.0 |
| X-38 | V | - | [] | [33.0] | ['B1'] | 14.21 | 13.54 | 0.67 | 0.0 |
| X-45 | - | - | [] | [] | [] | 6.71 | 6.34 | 0.38 | 0.0 |
