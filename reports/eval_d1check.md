# Eval `d1check`

## Summary

| metric | value |
|---|---|
| videos | 4 |
| tp | 2 |
| fp | 0 |
| fn | 0 |
| tn | 2 |
| precision | 1.0 |
| recall | 1.0 |
| f1 | 1.0 |
| timecodes_total | 2 |
| timecodes_matched | 2 |
| err_median | 0.65 |
| err_max | 1.0 |
| within_1s | 1.0 |
| within_2s | 1.0 |
| win_err_median | 0.0 |
| type_acc | 1.0 |
| time_mean_s | 2.51 |
| time_max_s | 4.63 |
| peak_rss_gb | 2.88 |

## Per confirmed code

| code | labels | video hit | right type | errors (s) |
|---|---|---|---|---|
| B1 | 1 | 1 | 1 | [0.3] |
| D1 | 1 | 1 | 1 | [1.0] |

## Misses (violation clips predicted clean)


## False positives (clean clips predicted violation)


## Per video

| clip | label | pred | preds (t, code) | label t | want | time s | perception | rules | vlm |
|---|---|---|---|---|---|---|---|---|---|
| X-14 | - | - | [] | [] | [] | 4.56 | 2.94 | 1.26 | 0.0 |
| X-15 | V | V | [(15.3, 'B1')] | [15.0] | ['B1'] | 0.48 | 0.11 | 0.22 | 0.01 |
| X-16 | - | - | [] | [] | [] | 0.37 | 0.11 | 0.26 | 0.001 |
| X-17 | V | V | [(6.0, 'D1')] | [7.0] | ['D1'] | 4.63 | 0.1 | 4.43 | 0.0 |
