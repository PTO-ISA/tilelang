#!/usr/bin/env bash
set +e
OUT=/tmp/st_simtvf_parallel
RFL=/tmp/pr272_sv567_rfladder_20261006
echo ===SIMT_PRIMARY_OPSIMS===
for t in \
  sv1_e256_t32 sv1_e2048_t32 \
  sv2_r16_c32_t32_input_keep sv2_r16_c64_t32_input_keep \
  sv2_r32_c64_t32_input_stream sv2_r32_c128_t32_input_stream sv2_r32_c64_t32_reload \
  sv3_m24_vl64_k16_t32_keep sv3_m32_vl64_k16_t32_keep sv3_m32_vl64_k16_t32_split_cm16 \
  sv4_e256_b8_t32_keep_idx sv4_e256_b8_t32_remat_idx_calc \
  sv5_r64_c128_g16_t32_keep_in_rf sv5_r64_c128_g16_t32_ub_stream \
  sv6_r64_c128_g32_t32_keep_in_warp sv6_r64_c128_g32_t32_reload \
  sv7_r64_c128_g64_t32_multiwarp_ub sv7_r64_c128_g64_t32_reload \
  sv7_r64_c128_g128_t32_multiwarp_ub sv7_r64_c128_g128_t32_reload \
  sv8_r64_c128_g16_t32_live sv8_r64_c128_g16_t32_spill_dist \
  sv9_e256_k8_t32_keep sv9_e256_k8_t32_remat_scores sv9_e256_k8_t32_remat_idx
do
  if [[ -d "$OUT/opsim_$t" ]]; then echo "FOUND $OUT/opsim_$t"
  elif [[ -d "$RFL/opsim_$t" ]]; then echo "FOUND_ALT $RFL/opsim_$t"
  else echo "MISSING $t"
  fi
done
echo ===CSV_SV2===
find "$OUT/opsim_sv2_r32_c64_t32_input_stream" -name '*instr_exe.csv' 2>/dev/null | head -3
find "$OUT/ptodsl_sv1_sv9d/opsim_sv2d_r32_c64_t32_input_stream" -name '*instr_exe.csv' 2>/dev/null | head -3
echo ===WALL_SV2===
for t in sv2_r16_c32_t32_input_keep sv2_r32_c64_t32_input_stream sv2_r64_c64_t32_input_stream; do
  echo -n "$t: "
  grep -E 'PASS|FAIL|core0.veccore0' "$OUT/opsim_${t}.log" 2>/dev/null | head -5 | tr '\n' ' ; '
  echo
done
echo ===SUITE_KERNEL_HEAD===
head -30 /tmp/st_simtvf_parallel_suite/kernels/sv2_eltwise_bcast_rf.py
echo ===SO===
ls "$OUT/so"/sv2* 2>/dev/null | head -20
