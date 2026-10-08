# Run on LAPTOP — deploy SV3 to pto-b10
$ErrorActionPreference = 'Continue'
$dir = Join-Path $env:USERPROFILE 'st_simtvf_parallel'
if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir | Out-Null }
$tg = Join-Path $dir 'st_simtvf_parallel_suite.tar.gz'
if (-not (Test-Path $tg)) {
  Write-Host "MISSING $tg — place suite tarball first"
  exit 2
}
Write-Host "Uploading suite to pto-b10 and running SV3 oneshot..."
scp -o BatchMode=yes -o ConnectTimeout=90 $tg "pto-b10:/tmp/st_simtvf_parallel_suite.tar.gz"
ssh -o BatchMode=yes -o ConnectTimeout=90 pto-b10 @"
rm -rf /tmp/st_simtvf_parallel_suite && mkdir -p /tmp/st_simtvf_parallel_suite &&
tar -xzf /tmp/st_simtvf_parallel_suite.tar.gz -C /tmp/st_simtvf_parallel_suite &&
sed -i 's/\r$//' /tmp/st_simtvf_parallel_suite/*.sh /tmp/st_simtvf_parallel_suite/*.py /tmp/st_simtvf_parallel_suite/kernels/*.py &&
chmod +x /tmp/st_simtvf_parallel_suite/oneshot_sv3.sh &&
bash /tmp/st_simtvf_parallel_suite/oneshot_sv3.sh
"@ | Tee-Object -FilePath (Join-Path $dir 'oneshot_sv3_out.txt')
ssh -o BatchMode=yes pto-b10 "cat /tmp/st_simtvf_parallel/SUMMARY_sv3_raw.txt; echo '---'; for t in sv3_m24_vl64_k16_t32_keep sv3_m32_vl64_k16_t32_keep sv3_m32_vl64_k16_t32_split_cm16; do echo ==== `$t ====; grep -E 'PASS|FAIL|duration|COMPILE|membar|MTE' /tmp/st_simtvf_parallel/opsim_`${t}.log /tmp/st_simtvf_parallel/logs/compile_`${t}.log 2>/dev/null | head -20; done" |
  Tee-Object -FilePath (Join-Path $dir 'SUMMARY_sv3.md')
ssh -o BatchMode=yes pto-b10 "cd /tmp/st_simtvf_parallel && tar -czf /tmp/st_simtvf_sv3_reports.tgz SUMMARY_sv3_raw.txt oneshot_sv3.log logs/compile_sv3* opsim_sv3* so/sv3* sources/*sv3* 2>/dev/null; ls -la /tmp/st_simtvf_sv3_reports.tgz"
scp -o BatchMode=yes pto-b10:/tmp/st_simtvf_sv3_reports.tgz (Join-Path $dir 'st_simtvf_sv3_reports.tgz')
Write-Host "OUT=$dir"
