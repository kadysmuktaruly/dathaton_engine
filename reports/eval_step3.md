# Eval `step3`

## Summary

| metric | value |
|---|---|
| videos | 40 |
| tp | 16 |
| fp | 12 |
| fn | 5 |
| tn | 7 |
| precision | 0.571 |
| recall | 0.762 |
| f1 | 0.653 |
| timecodes_total | 22 |
| timecodes_matched | 17 |
| err_median | 4.3 |
| err_max | 15.3 |
| within_1s | 0.118 |
| within_2s | 0.294 |
| win_err_median | 3.3 |
| type_acc | 0.353 |
| time_mean_s | 0.38 |
| time_max_s | 3.39 |
| peak_rss_gb | 0.75 |

## Per confirmed code

| code | labels | video hit | right type | errors (s) |
|---|---|---|---|---|
| A1 | 1 | 1 | 0 | [4.3] |
| A2 | 1 | 1 | 0 | [12.7] |
| A3 | 1 | 0 | 0 | [] |
| A5 | 1 | 1 | 0 | [4.7] |
| B1 | 4 | 4 | 0 | [1.7, 1.0, 6.7, 0.3] |
| B3 | 1 | 1 | 0 | [13.0] |
| B5 | 5 | 4 | 4 | [1.7, 4.7, 4.3, 1.3] |
| B7 | 2 | 1 | 0 | [15.3] |
| C1 | 2 | 2 | 2 | [3.0, 9.3] |
| C2 | 2 | 2 | 0 | [2.7, 2.7] |
| D1 | 1 | 0 | 0 | [] |
| X0 | 1 | 0 | 0 | [] |

## Misses (violation clips predicted clean)

- X-1 labels [22.0] want ['A3']; overlay: runs/overlays/X-1.mp4
- X-17 labels [7.0] want ['D1']; overlay: runs/overlays/X-17.mp4
- X-20 labels [5.0] want ['B5']; overlay: runs/overlays/X-20.mp4
- X-44 labels [16.0] want ['B7']; overlay: runs/overlays/X-44.mp4
- X-46 labels [11.0] want ['X0']; overlay: runs/overlays/X-46.mp4

## False positives (clean clips predicted violation)

- X-5 preds [(1.0, 'C1'), (62.0, 'C1')] events [('C1', 'to_staff_side'), ('C1', 'to_staff_side')]; overlay: runs/overlays/X-5.mp4
- X-12 preds [(37.0, 'B5'), (40.3, 'B5'), (40.3, 'C1')] events [('B5', 'cashier_left_room'), ('B5', 'cashier_left_room'), ('C1', 'entered_empty_room')]; overlay: runs/overlays/X-12.mp4
- X-16 preds [(2.3, 'C1')] events [('C1', 'to_staff_side')]; overlay: runs/overlays/X-16.mp4
- X-19 preds [(1.7, 'C1'), (49.7, 'C1')] events [('C1', 'to_staff_side'), ('C1', 'to_staff_side')]; overlay: runs/overlays/X-19.mp4
- X-23 preds [(5.7, 'B5'), (14.7, 'B5')] events [('B5', 'cashier_left_room'), ('B5', 'cashier_left_room')]; overlay: runs/overlays/X-23.mp4
- X-25 preds [(28.3, 'B5')] events [('B5', 'cashier_left_room')]; overlay: runs/overlays/X-25.mp4
- X-27 preds [(21.7, 'B5')] events [('B5', 'cashier_left_room')]; overlay: runs/overlays/X-27.mp4
- X-29 preds [(15.7, 'B5')] events [('B5', 'cashier_left_room')]; overlay: runs/overlays/X-29.mp4
- X-31 preds [(3.3, 'C1')] events [('C1', 'entered_empty_room')]; overlay: runs/overlays/X-31.mp4
- X-35 preds [(3.0, 'C1')] events [('C1', 'to_staff_side')]; overlay: runs/overlays/X-35.mp4
- X-37 preds [(2.7, 'C1')] events [('C1', 'to_staff_side')]; overlay: runs/overlays/X-37.mp4
- X-39 preds [(1.7, 'C1'), (45.3, 'C1')] events [('C1', 'to_staff_side'), ('C1', 'to_staff_side')]; overlay: runs/overlays/X-39.mp4

## Per video

| clip | label | pred | preds (t, code) | label t | want | time s | perception | rules | vlm |
|---|---|---|---|---|---|---|---|---|---|
| X-1 | V | - | [] | [22.0] | ['A3'] | 3.39 | 3.13 | 0.02 | 0.0 |
| X-2 | V | V | [(14.7, 'C1')] | [10.0] | ['A5'] | 0.25 | 0.11 | 0.01 | 0.0 |
| X-3 | - | - | [] | [] | [] | 0.13 | 0.11 | 0.02 | 0.0 |
| X-4 | V | V | [(5.0, 'C1'), (55.3, 'C1')] | [57.0, 68.0] | ['B1', 'A2'] | 0.7 | 0.12 | 0.05 | 0.0 |
| X-5 | - | V | [(1.0, 'C1'), (62.0, 'C1')] | [] | [] | 0.7 | 0.11 | 0.06 | 0.0 |
| X-8 | - | - | [] | [] | [] | 0.13 | 0.11 | 0.03 | 0.0 |
| X-10 | - | - | [] | [] | [] | 0.12 | 0.1 | 0.01 | 0.0 |
| X-11 | V | V | [(10.3, 'B5'), (19.7, 'B5')] | [18.0] | ['B5'] | 0.42 | 0.11 | 0.02 | 0.0 |
| X-12 | - | V | [(37.0, 'B5'), (40.3, 'B5'), (40.3, 'C1')] | [] | [] | 1.14 | 0.12 | 0.02 | 0.0 |
| X-13 | V | V | [(62.3, 'B5')] | [47.0] | ['B7'] | 0.66 | 0.12 | 0.03 | 0.0 |
| X-14 | - | - | [] | [] | [] | 0.22 | 0.13 | 0.09 | 0.0 |
| X-15 | V | V | [(16.0, 'B5')] | [15.0] | ['B1'] | 0.29 | 0.12 | 0.02 | 0.0 |
| X-16 | - | V | [(2.3, 'C1')] | [] | [] | 0.19 | 0.12 | 0.02 | 0.0 |
| X-17 | V | - | [] | [7.0] | ['D1'] | 0.12 | 0.11 | 0.01 | 0.0 |
| X-18 | V | V | [(13.3, 'C1'), (17.0, 'C1')] | [20.0] | ['C1'] | 0.4 | 0.11 | 0.02 | 0.0 |
| X-19 | - | V | [(1.7, 'C1'), (49.7, 'C1')] | [] | [] | 0.6 | 0.12 | 0.03 | 0.0 |
| X-20 | V | - | [] | [5.0] | ['B5'] | 0.12 | 0.11 | 0.01 | 0.0 |
| X-21 | - | - | [] | [] | [] | 0.12 | 0.11 | 0.01 | 0.0 |
| X-22 | V | V | [(18.7, 'B5')] | [14.0] | ['B5'] | 0.29 | 0.11 | 0.01 | 0.0 |
| X-23 | - | V | [(5.7, 'B5'), (14.7, 'B5')] | [] | [] | 0.33 | 0.11 | 0.01 | 0.0 |
| X-24 | V | V | [(12.3, 'B5')] | [8.0] | ['B5'] | 0.23 | 0.1 | 0.01 | 0.0 |
| X-25 | - | V | [(28.3, 'B5')] | [] | [] | 0.36 | 0.11 | 0.02 | 0.0 |
| X-26 | V | V | [(15.7, 'B5')] | [9.0] | ['B1'] | 0.26 | 0.1 | 0.01 | 0.0 |
| X-27 | - | V | [(21.7, 'B5')] | [] | [] | 0.31 | 0.11 | 0.01 | 0.0 |
| X-28 | V | V | [(5.7, 'B5'), (9.0, 'C1'), (12.3, 'B5')] | [7.0] | ['B5'] | 0.39 | 0.1 | 0.01 | 0.0 |
| X-29 | - | V | [(15.7, 'B5')] | [] | [] | 0.25 | 0.1 | 0.01 | 0.0 |
| X-30 | V | V | [(3.7, 'C1')] | [13.0] | ['C1'] | 0.17 | 0.11 | 0.01 | 0.0 |
| X-31 | - | V | [(3.3, 'C1')] | [] | [] | 0.18 | 0.12 | 0.01 | 0.0 |
| X-32 | V | V | [(2.3, 'C1')] | [5.0] | ['C2'] | 0.16 | 0.11 | 0.01 | 0.0 |
| X-33 | V | V | [(1.7, 'B5'), (8.7, 'C1')] | [6.0] | ['C2'] | 0.24 | 0.1 | 0.01 | 0.0 |
| X-34 | V | V | [(1.0, 'C1'), (8.7, 'C1')] | [13.0] | ['A1'] | 0.25 | 0.1 | 0.01 | 0.0 |
| X-35 | - | V | [(3.0, 'C1')] | [] | [] | 0.19 | 0.12 | 0.03 | 0.0 |
| X-37 | - | V | [(2.7, 'C1')] | [] | [] | 0.16 | 0.1 | 0.01 | 0.0 |
| X-38 | V | V | [(2.0, 'C1'), (32.7, 'C1')] | [33.0] | ['B1'] | 0.46 | 0.12 | 0.03 | 0.0 |
| X-39 | - | V | [(1.7, 'C1'), (45.3, 'C1')] | [] | [] | 0.55 | 0.12 | 0.03 | 0.0 |
| X-40 | V | V | [(4.0, 'C1')] | [17.0] | ['B3'] | 0.18 | 0.11 | 0.02 | 0.0 |
| X-43 | - | - | [] | [] | [] | 0.12 | 0.11 | 0.01 | 0.0 |
| X-44 | V | - | [] | [16.0] | ['B7'] | 0.12 | 0.11 | 0.01 | 0.0 |
| X-45 | - | - | [] | [] | [] | 0.13 | 0.11 | 0.02 | 0.0 |
| X-46 | V | - | [] | [11.0] | ['X0'] | 0.12 | 0.1 | 0.01 | 0.0 |
