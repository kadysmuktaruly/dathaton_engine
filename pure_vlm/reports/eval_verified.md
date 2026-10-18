# pure_vlm eval `verified`

| metric | value |
|---|---|
| videos | 40 |
| tp | 19 |
| fp | 2 |
| fn | 2 |
| tn | 17 |
| precision | 0.905 |
| recall | 0.905 |
| f1 | 0.905 |
| labels | 22 |
| type_hit | 19 |
| type_hit_within_2s | 14 |
| err_median | 1.5 |
| type_err_median | 2.0 |
| vlm_s_mean | 117.1 |
| vlm_s_max | 230.96 |

| clip | label | pred | want (t) | events (code, t, reason) | dropped by the check | type hit (err s) | VLM s / video s |
|---|---|---|---|---|---|---|---|
| X-1 | V | V | A3@22 | A6@9.5 cash_without_id |  | no | 111 / 26 |
| X-2 | V | V | A6@10 | A6@9.5 cash_without_id |  | yes (0.5) | 85 / 17 |
| X-3 | - | - |  |  |  |  | 0 / 20 |
| X-4 | V | V | B1@57, B1@68 | B1@56.5 cash_on_arrival |  | yes (0.5), yes (11.5) | 218 / 73 |
| X-5 | - | - |  |  |  |  | 229 / 72 |
| X-8 | - | - |  |  |  |  | 77 / 38 |
| X-10 | - | - |  |  | B3@4 personal_clothing |  | 64 / 16 |
| X-11 | V | V | B9@18 | B9@16 laptop_active_left |  | yes (2.0) | 99 / 26 |
| X-12 | - | - |  |  |  |  | 131 / 42 |
| X-13 | V | V | D6@47 | D6@49 bag_to_safe_unregistered |  | yes (2.0) | 227 / 65 |
| X-14 | - | - |  |  | B3@31 personal_handbag; C2@80 door_left_open |  | 229 / 142 |
| X-15 | V | V | B1@15 | B1@11 bag_left; D7@15 bag_without_papers | B3@1 personal_clothing; B8@11 stamps_left | yes (4.0) | 116 / 21 |
| X-16 | - | - |  |  |  |  | 149 / 23 |
| X-17 | V | V | D7@7 | D7@6.5 bag_without_papers |  | yes (0.5) | 89 / 15 |
| X-18 | V | V | A7@20 | A7@18 two_visitors |  | yes (2.0) | 139 / 23 |
| X-19 | - | V |  | A6@7.5 cash_without_id; A6@54.5 cash_without_id |  |  | 229 / 57 |
| X-20 | V | V | B5@5 | B5@5 safe_open_at_work |  | yes (0.0) | 0 / 12 |
| X-21 | - | - |  |  |  |  | 82 / 20 |
| X-22 | V | - | B5@14 |  |  | no | 75 / 21 |
| X-23 | - | - |  |  |  |  | 74 / 18 |
| X-24 | V | V | B8@8 | B8@11.5 items_left |  | yes (3.5) | 90 / 15 |
| X-25 | - | - |  |  | B5@16 safe_open_at_work |  | 124 / 32 |
| X-26 | V | V | B1@9 | B1@11 cash_left; B8@11 items_left | C2@16.5 door_left_open | yes (2.0) | 76 / 20 |
| X-27 | - | - |  |  |  |  | 119 / 26 |
| X-28 | V | V | B9@7 | B9@9 laptop_active_left |  | yes (2.0) | 63 / 16 |
| X-29 | - | - |  |  |  |  | 67 / 17 |
| X-30 | V | V | C2@13 | C2@5 door_left_open |  | yes (8.0) | 82 / 19 |
| X-31 | - | - |  |  |  |  | 94 / 24 |
| X-32 | V | V | C2@5 | C2@5 door_left_open |  | yes (0.0) | 82 / 21 |
| X-33 | V | V | C2@6 | C2@5.5 door_left_open |  | yes (0.5) | 71 / 14 |
| X-34 | V | V | A1@13 | A1@13.5 left_during_count |  | yes (0.5) | 135 / 18 |
| X-35 | - | V |  | A6@8 cash_without_id |  |  | 228 / 57 |
| X-37 | - | - |  |  |  |  | 91 / 16 |
| X-38 | V | V | B1@33 | A6@7 cash_without_id; B1@33 cash_on_arrival |  | yes (0.0) | 229 / 59 |
| X-39 | - | - |  |  |  |  | 231 / 79 |
| X-40 | V | V | B3@17 | B3@9 personal_handbag |  | yes (8.0) | 108 / 25 |
| X-43 | - | - |  |  |  |  | 88 / 23 |
| X-44 | V | - | A2@16 |  |  | no | 98 / 20 |
| X-45 | - | - |  |  |  |  | 116 / 33 |
| X-46 | V | V | B3@11 | B3@10 personal_drink |  | yes (1.0) | 67 / 19 |
