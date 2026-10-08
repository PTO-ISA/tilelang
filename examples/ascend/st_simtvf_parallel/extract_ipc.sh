#!/usr/bin/env bash
cd /tmp/st_simtvf_parallel_suite/reports || exit 1
OUT=/tmp/st_simtvf_parallel/PASS_TABLE_sv1_sv9_ipc.txt
{
printf '%-42s %-6s %8s %12s\n' TAG PASS us IPC_proxy
printf '%s\n' '------------------------------------------------------------------------------'
for t in sv1_e256_t32 sv1_e2048_t32 sv2_r32_c32_t32_frag_live sv2_r32_c32_t32_reload sv3_m24_vl64_k16_t32_keep sv3_m32_vl64_k16_t32_keep sv3_m32_vl64_k16_t32_split_cm16 sv4_e256_b8_t32_keep_idx sv4_e256_b8_t32_remat_idx sv4_e256_b16_t32_keep_idx sv4_e256_b16_t32_remat_idx sv5_r64_c128_g16_t32 sv6_r64_c128_g32_t32 sv7_r64_c128_g64_t32 sv7_r64_c128_g128_t32 sv8_r64_c128_g16_t32_live sv8_r64_c128_g16_t32_spill_dist sv9_e256_k1_t32_keep sv9_e256_k8_t32_keep sv9_e256_k8_t32_remat_scores sv9_e256_k8_t32_remat_idx; do
  us=$(grep -oE 'wall_us: [0-9.]+' "$t.md" 2>/dev/null | head -1 | awk '{print $2}')
  ipc=$(grep -oE 'IPC_proxy=[0-9.]+' "$t.md" 2>/dev/null | head -1 | sed 's/IPC_proxy=//')
  st=$(grep -oE 'status: \*\*[A-Z]+\*\*' "$t.md" 2>/dev/null | head -1 | tr -d '*' | awk '{print $2}')
  printf '%-42s %-6s %8s %12s\n' "$t" "${st:-?}" "${us:-?}" "${ipc:-n/a}"
done
} | tee "$OUT"
# also append IPC column update into reports tgz
cp -f "$OUT" /tmp/st_simtvf_parallel/PASS_TABLE_sv1_sv9.txt
cd /tmp/st_simtvf_parallel
tar -rzf /tmp/st_simtvf_sv1_sv9_reports.tgz PASS_TABLE_sv1_sv9.txt PASS_TABLE_sv1_sv9_ipc.txt 2>/dev/null || \
  tar -czf /tmp/st_simtvf_sv1_sv9_reports_ipc.tgz PASS_TABLE_sv1_sv9.txt PASS_TABLE_sv1_sv9_ipc.txt SUMMARY_sv1_sv9_raw.txt
ls -la "$OUT" /tmp/st_simtvf_sv1_sv9_reports.tgz
