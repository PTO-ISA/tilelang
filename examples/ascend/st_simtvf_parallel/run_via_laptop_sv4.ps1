# Run on LAPTOP — deploy SV4 gather_psum to pto-b10
$ErrorActionPreference = 'Continue'
$dir = Join-Path $env:USERPROFILE 'st_simtvf_parallel'
if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir | Out-Null }
$tg = Join-Path $dir 'st_simtvf_parallel_suite.tar.gz'
if (-not (Test-Path $tg)) {
  Write-Host "MISSING $tg — place suite tarball first"
  exit 2
}
Write-Host "Uploading suite to pto-b10 and running SV4 gather_psum oneshot..."
scp -o BatchMode=yes -o ConnectTimeout=90 $tg "pto-b10:/tmp/st_simtvf_parallel_suite.tar.gz"
ssh -o BatchMode=yes -o ConnectTimeout=90 pto-b10 @"
rm -rf /tmp/st_simtvf_parallel_suite && mkdir -p /tmp/st_simtvf_parallel_suite &&
tar -xzf /tmp/st_simtvf_parallel_suite.tar.gz -C /tmp/st_simtvf_parallel_suite &&
sed -i 's/\r$//' /tmp/st_simtvf_parallel_suite/*.sh /tmp/st_simtvf_parallel_suite/*.py /tmp/st_simtvf_parallel_suite/kernels/*.py &&
chmod +x /tmp/st_simtvf_parallel_suite/oneshot_sv4.sh &&
bash /tmp/st_simtvf_parallel_suite/oneshot_sv4.sh
"@ | Tee-Object -FilePath (Join-Path $dir 'oneshot_sv4_out.txt')
ssh -o BatchMode=yes pto-b10 "cat /tmp/st_simtvf_parallel/SUMMARY_sv4_raw.txt; echo '---'; for t in sv4_e256_b8_t32_keep_idx sv4_e256_b8_t32_remat_idx sv4_e256_b16_t32_keep_idx sv4_e256_b16_t32_remat_idx; do echo ==== `$t ====; grep -E 'PASS|FAIL|duration|COMPILE|membar|MTE' /tmp/st_simtvf_parallel/opsim_`${t}.log /tmp/st_simtvf_parallel/logs/compile_`${t}.log 2>/dev/null | head -20; done" |
  Tee-Object -FilePath (Join-Path $dir 'SUMMARY_sv4.md')
ssh -o BatchMode=yes pto-b10 "cd /tmp/st_simtvf_parallel && tar -czf /tmp/st_simtvf_sv4_reports.tgz SUMMARY_sv4_raw.txt oneshot_sv4.log logs/compile_sv4* opsim_sv4* so/sv4* sources/*sv4* 2>/dev/null; ls -la /tmp/st_simtvf_sv4_reports.tgz"
scp -o BatchMode=yes pto-b10:/tmp/st_simtvf_sv4_reports.tgz (Join-Path $dir 'st_simtvf_sv4_reports.tgz')
Write-Host "OUT=$dir"
