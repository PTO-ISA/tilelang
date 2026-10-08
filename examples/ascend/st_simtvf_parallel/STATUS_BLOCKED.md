# ST SimtVF Parallel suite — ACCESS BLOCKED on pto-b10

**Date:** 2026-09-22 ~19:58 UTC+8  
**Status:** Suite **implemented + packaged**; opsim **not executed** — executor Shell is bound to box (`grok-bot-vm-*`), not laptop `04894df4-…`.

## Access blocker
- `ssh -p 8022 happybot@pto-b10.ddns.net` from box → `Permission denied (publickey)` (needs laptop key).
- Shell `machineId` not honored for this Grok Bot executor (param stripped / no user_machine target).
- MCP catalog: Gmail/Outlook only — no `ListMachines` / `CopyFromBox` in this agent’s tool surface.
- Same mode as `simtvf_ab_pto/STATUS_BLOCKED.md`.

## Parent unblock (one shot)
1. `ListMachines` → confirm `04894df4-ce02-4d53-a618-932598da9ca2` (`LAPTOP-C8QEKIFC`) connected.
2. `CopyFromBox` `/workspace/st_simtvf_parallel_suite.tar.gz` → `C:\Users\Happy\st_simtvf_parallel\st_simtvf_parallel_suite.tar.gz`
3. Also copy `/workspace/st_simtvf_parallel/run_via_laptop.ps1` → same folder.
4. On laptop Shell (machineId bound):

```powershell
Set-Location $env:USERPROFILE\st_simtvf_parallel
powershell -File .\run_via_laptop.ps1
```

Or stdin b64 pipe:

```powershell
Get-Content -Raw .\st_simtvf_parallel_suite.tar.gz.b64 |
  ssh -o BatchMode=yes -o ConnectTimeout=90 pto-b10 "base64 -d > /tmp/st_simtvf_parallel_suite.tar.gz && rm -rf /tmp/st_simtvf_parallel_suite && mkdir -p /tmp/st_simtvf_parallel_suite && tar -xzf /tmp/st_simtvf_parallel_suite.tar.gz -C /tmp/st_simtvf_parallel_suite && sed -i 's/\r$//' /tmp/st_simtvf_parallel_suite/*.sh /tmp/st_simtvf_parallel_suite/*.py /tmp/st_simtvf_parallel_suite/kernels/*.py && bash /tmp/st_simtvf_parallel_suite/oneshot_st_simtvf_parallel.sh"
ssh -o BatchMode=yes pto-b10 "cat /tmp/st_simtvf_parallel/SUMMARY.md"
```

5. Pull reports back to box (CopyToBox / scp from laptop of `st_simtvf_reports.tgz`) into `/workspace/st_simtvf_parallel/reports/`.

## What oneshot runs (REGRESSION-2 recipe)
1. **SV2** 4 cells: E=256 × K={1,8} × T={32,128} — extract SimtVF tournament; expect T32K8≈1.48µs T128K8≈3.03µs
2. SV3 @ T=32 K={1,8}; SV4 @ K=8 T={32,128}
3. SV1 E={256,2048} T=32; SV7 N=128 T={32,128}; SV8 32×32 T={32,128}
4. SV5 frag/reload; SV6 remat/reload (best-effort)
5. `harvest_report.py` → `/tmp/st_simtvf_parallel/SUMMARY.md` + per-tag reports

Env: `.venv-npu` + `cann_91b3` + `TORCH_DEVICE_BACKEND_AUTOLOAD=0` + deps-native lib + int2 patch + `target=ascend` cython.
