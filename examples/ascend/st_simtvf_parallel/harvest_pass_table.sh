#!/usr/bin/env bash
set -euo pipefail
OUT=/tmp/st_simtvf_parallel
cd "$OUT"
python3 - <<'PY'
import re, os
out="/tmp/st_simtvf_parallel"
tags=[
"sv1_e256_t32","sv1_e2048_t32",
"sv2_r32_c32_t32_frag_live","sv2_r32_c32_t32_reload",
"sv3_m24_vl64_k16_t32_keep","sv3_m32_vl64_k16_t32_keep","sv3_m32_vl64_k16_t32_split_cm16",
"sv4_e256_b8_t32_keep_idx","sv4_e256_b8_t32_remat_idx","sv4_e256_b16_t32_keep_idx","sv4_e256_b16_t32_remat_idx",
"sv5_r64_c128_g16_t32","sv6_r64_c128_g32_t32",
"sv7_r64_c128_g64_t32","sv7_r64_c128_g128_t32",
"sv8_r64_c128_g16_t32_live","sv8_r64_c128_g16_t32_spill_dist",
"sv9_e256_k1_t32_keep","sv9_e256_k8_t32_keep","sv9_e256_k8_t32_remat_scores","sv9_e256_k8_t32_remat_idx",
]
lines=[]
lines.append(f"{'TAG':<42} {'PASS':<6} {'us':>8} {'IPC':>12} so")
lines.append('-'*78)
for t in tags:
    log=f"{out}/opsim_{t}.log"
    us='?'; ipc='n/a'; status='MISSING'
    if os.path.isfile(log):
        txt=open(log,errors='ignore').read()
        m=re.search(r'core0\.veccore0\s+([0-9.]+)', txt)
        if m: us=m.group(1)
        for pat in [r'IPC_proxy[=:\s]+([0-9.]+)', r'ipc_proxy[=:\s]+([0-9.]+)', r'VF_?IPC[=:\s]+([0-9.]+)', r'\bIPC[=:\s]+([0-9.]+)']:
            m=re.search(pat, txt, re.I)
            if m:
                ipc=m.group(1); break
        # also check harvest/csv under opsim dir
        if ipc=='n/a':
            for root,dirs,files in os.walk(f"{out}/opsim_{t}"):
                for fn in files:
                    if fn.endswith(('.csv','.txt','.json','.log')):
                        try:
                            t2=open(os.path.join(root,fn),errors='ignore').read()
                        except Exception:
                            continue
                        for pat in [r'IPC_proxy[=:\s,]+([0-9.]+)', r'ipc[=:\s,]+([0-9.]+)']:
                            m=re.search(pat, t2, re.I)
                            if m:
                                ipc=m.group(1); break
                if ipc!='n/a': break
        if re.search(r'^PASS', txt, re.M): status='PASS'
        elif re.search(r'^FAIL', txt, re.M): status='FAIL'
        else: status='NO_PASSLINE'
    so='Y' if os.path.isfile(f"{out}/so/{t}.so") else 'N'
    lines.append(f"{t:<42} {status:<6} {us:>8} {ipc:>12} {so}")
path=f"{out}/PASS_TABLE_sv1_sv9.txt"
open(path,'w').write('\n'.join(lines)+'\n')
print('\n'.join(lines))
PY

ls -la /home/happybot/projects/tilelang-pto-vmi-deps-stack/build/lib/libtilelang.so /tmp/libtilelang.so.deps_backup_ab

# lightweight reports tarball
tar -czf /tmp/st_simtvf_sv1_sv9_reports.tgz \
  PASS_TABLE_sv1_sv9.txt SUMMARY_sv1_sv9_raw.txt \
  oneshot_sv1_sv9.log parallel_resume_sv1_sv9.log harvest_sv1_sv9.log \
  logs/compile_sv1_e256_t32.log logs/compile_sv1_e2048_t32.log \
  logs/compile_sv2_r32_c32_t32_frag_live.log logs/compile_sv2_r32_c32_t32_reload.log \
  logs/compile_sv3_m24_vl64_k16_t32_keep.log logs/compile_sv3_m32_vl64_k16_t32_keep.log logs/compile_sv3_m32_vl64_k16_t32_split_cm16.log \
  logs/compile_sv4_e256_b8_t32_keep_idx.log logs/compile_sv4_e256_b8_t32_remat_idx.log \
  logs/compile_sv4_e256_b16_t32_keep_idx.log logs/compile_sv4_e256_b16_t32_remat_idx.log \
  logs/compile_sv5_r64_c128_g16_t32.log logs/compile_sv6_r64_c128_g32_t32.log \
  logs/compile_sv7_r64_c128_g64_t32.log logs/compile_sv7_r64_c128_g128_t32.log \
  logs/compile_sv8_r64_c128_g16_t32_live.log logs/compile_sv8_r64_c128_g16_t32_spill_dist.log \
  logs/compile_sv9_e256_k1_t32_keep.log logs/compile_sv9_e256_k8_t32_keep.log \
  logs/compile_sv9_e256_k8_t32_remat_scores.log logs/compile_sv9_e256_k8_t32_remat_idx.log \
  opsim_sv1_e256_t32.log opsim_sv1_e2048_t32.log \
  opsim_sv2_r32_c32_t32_frag_live.log opsim_sv2_r32_c32_t32_reload.log \
  opsim_sv3_m24_vl64_k16_t32_keep.log opsim_sv3_m32_vl64_k16_t32_keep.log opsim_sv3_m32_vl64_k16_t32_split_cm16.log \
  opsim_sv4_e256_b8_t32_keep_idx.log opsim_sv4_e256_b8_t32_remat_idx.log \
  opsim_sv4_e256_b16_t32_keep_idx.log opsim_sv4_e256_b16_t32_remat_idx.log \
  opsim_sv5_r64_c128_g16_t32.log opsim_sv6_r64_c128_g32_t32.log \
  opsim_sv7_r64_c128_g64_t32.log opsim_sv7_r64_c128_g128_t32.log \
  opsim_sv8_r64_c128_g16_t32_live.log opsim_sv8_r64_c128_g16_t32_spill_dist.log \
  opsim_sv9_e256_k1_t32_keep.log opsim_sv9_e256_k8_t32_keep.log \
  opsim_sv9_e256_k8_t32_remat_scores.log opsim_sv9_e256_k8_t32_remat_idx.log \
  so/sv1_e256_t32.so so/sv1_e2048_t32.so \
  so/sv2_r32_c32_t32_frag_live.so so/sv2_r32_c32_t32_reload.so \
  so/sv3_m24_vl64_k16_t32_keep.so so/sv3_m32_vl64_k16_t32_keep.so so/sv3_m32_vl64_k16_t32_split_cm16.so \
  so/sv4_e256_b8_t32_keep_idx.so so/sv4_e256_b8_t32_remat_idx.so \
  so/sv4_e256_b16_t32_keep_idx.so so/sv4_e256_b16_t32_remat_idx.so \
  so/sv5_r64_c128_g16_t32.so so/sv6_r64_c128_g32_t32.so \
  so/sv7_r64_c128_g64_t32.so so/sv7_r64_c128_g128_t32.so \
  so/sv8_r64_c128_g16_t32_live.so so/sv8_r64_c128_g16_t32_spill_dist.so \
  so/sv9_e256_k1_t32_keep.so so/sv9_e256_k8_t32_keep.so \
  so/sv9_e256_k8_t32_remat_scores.so so/sv9_e256_k8_t32_remat_idx.so \
  2>/dev/null
ls -la /tmp/st_simtvf_sv1_sv9_reports.tgz "$OUT/PASS_TABLE_sv1_sv9.txt"
# try harvest_report if present
if [[ -f /tmp/st_simtvf_parallel_suite/harvest_report.py ]]; then
  /home/happybot/projects/tilelang-deepseek/.venv-npu/bin/python /tmp/st_simtvf_parallel_suite/harvest_report.py > "$OUT/harvest_sv1_sv9_post.log" 2>&1 || true
  tail -40 "$OUT/harvest_sv1_sv9_post.log" || true
fi
