# Run on LAPTOP — Case-3 VMI twins (SimdVF / target=pto / overlay)
# Prefer after Simt oneshot_sv5_sv6_sv8 is idle (script self-guards on pto-b10).
$ErrorActionPreference = 'Continue'
$dir = Join-Path $env:USERPROFILE 'st_simtvf_parallel'
if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir | Out-Null }
$tg = Join-Path $dir 'st_simtvf_parallel_suite.tar.gz'
$b64 = Join-Path $dir 'st_simtvf_parallel_suite.tar.gz.b64'
if (-not (Test-Path $tg)) {
  if (Test-Path $b64) {
    Write-Host "Decoding b64 -> tar.gz"
    [IO.File]::WriteAllBytes($tg, [Convert]::FromBase64String((Get-Content -Raw $b64)))
  } else {
    Write-Host "MISSING $tg — place suite tarball first"
    exit 2
  }
}
Write-Host "Uploading suite to pto-b10 and running Case-3 SV5V/SV6V/SV8V (VMI) oneshot..."
scp -o BatchMode=yes -o ConnectTimeout=90 $tg "pto-b10:/tmp/st_simtvf_parallel_suite.tar.gz"
ssh -o BatchMode=yes -o ConnectTimeout=90 pto-b10 @"
rm -rf /tmp/st_simtvf_parallel_suite && mkdir -p /tmp/st_simtvf_parallel_suite &&
tar -xzf /tmp/st_simtvf_parallel_suite.tar.gz -C /tmp/st_simtvf_parallel_suite &&
sed -i 's/\r$//' /tmp/st_simtvf_parallel_suite/*.sh /tmp/st_simtvf_parallel_suite/*.py /tmp/st_simtvf_parallel_suite/kernels/*.py &&
chmod +x /tmp/st_simtvf_parallel_suite/oneshot_sv5v_sv6v_sv8v.sh &&
bash /tmp/st_simtvf_parallel_suite/oneshot_sv5v_sv6v_sv8v.sh
"@ | Tee-Object -FilePath (Join-Path $dir 'oneshot_sv5v_sv6v_sv8v_out.txt')
ssh -o BatchMode=yes pto-b10 "cat /tmp/st_simtvf_parallel/SUMMARY_sv5v_sv6v_sv8v_raw.txt; echo '---'; ls -la /tmp/st_simtvf_parallel/so/sv{5,6,8}v* 2>/dev/null; echo '---OPSIM---'; for t in sv5v_r64_c128_g16_t32 sv6v_r64_c128_g64_t32 sv6v_r64_c128_g128_t32 sv8v_r64_c128_g16_t32_live sv8v_r64_c128_g16_t32_spill_dist; do echo ==== \$t ====; grep -E 'PASS|FAIL|duration|IPC|core0' /tmp/st_simtvf_parallel/opsim_\${t}.log 2>/dev/null | head -20; done" |
  Tee-Object -FilePath (Join-Path $dir 'SUMMARY_sv5v_sv6v_sv8v.md')
ssh -o BatchMode=yes pto-b10 "cd /tmp/st_simtvf_parallel && tar -czf /tmp/st_simtvf_sv5v_sv6v_sv8v_reports.tgz SUMMARY_sv5v_sv6v_sv8v_raw.txt oneshot_sv5v_sv6v_sv8v.log logs/compile_sv{5,6,8}v* opsim_sv{5,6,8}v* so/sv{5,6,8}v* sources/*sv{5,6,8}v* 2>/dev/null; ls -la /tmp/st_simtvf_sv5v_sv6v_sv8v_reports.tgz"
scp -o BatchMode=yes pto-b10:/tmp/st_simtvf_sv5v_sv6v_sv8v_reports.tgz (Join-Path $dir 'st_simtvf_sv5v_sv6v_sv8v_reports.tgz')
Write-Host "OUT=$dir"
