# Case-3 SV5/SV6/SV8 — SimtVF GREEN

**2026-09-23 ~17:20 HKT** — all primary tags PASS on pto-b10 Ascend950PR_9599 (deps_backup_ab). VMI twins (`sv5v_*`/`sv6v_*`/`sv8v_*`) may exist on box; do not overwrite.

## Results
| tag | PASS | µs | IPC |
|-----|------|---:|----:|
| sv5_r64_c128_g16_t32 | PASS | 9.34 | 0.104 |
| sv6_r64_c128_g64_t32 | PASS | 14.44 | 0.090 |
| sv6_r64_c128_g128_t32 | PASS | 14.69 | 0.088 |
| sv8_r64_c128_g16_t32_live | PASS | 17.04 | 0.141 |
| sv8_r64_c128_g16_t32_spill_dist | PASS | 18.23 | 0.138 |

See `reports/ST_CASE3_DESIGN.md`. Lib restored to deps_backup_ab after oneshot.
