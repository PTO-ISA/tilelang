# Run on LAPTOP — deploy SV2 to pto-b10
$ErrorActionPreference = 'Continue'
$dir = Join-Path $env:USERPROFILE 'st_simtvf_parallel'
if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir | Out-Null }
$tg = Join-Path $dir 'st_simtvf_parallel_suite.tar.gz'
if (-not (Test-Path $tg)) {
  Write-Host "MISSING $tg — place suite tarball first"
  exit 2
}
Write-Host "Uploading suite to pto-b10 and running SV2 oneshot..."
scp -o BatchMode=yes -o ConnectTimeout=90 $tg "pto-b10:/tmp/st_simtvf_parallel_suite.tar.gz"
ssh -o BatchMode=yes -o ConnectTimeout=90 pto-b10 @"
rm -rf /tmp/st_simtvf_parallel_suite && mkdir -p /tmp/st_simtvf_parallel_suite &&
tar -xzf /tmp/st_simtvf_parallel_suite.tar.gz -C /tmp/st_simtvf_parallel_suite &&
sed -i 's/\r$//' /tmp/st_simtvf_parallel_suite/*.sh /tmp/st_simtvf_parallel_suite/*.py /tmp/st_simtvf_parallel_suite/kernels/*.py &&
chmod +x /tmp/st_simtvf_parallel_suite/oneshot_sv2.sh &&
bash /tmp/st_simtvf_parallel_suite/oneshot_sv2.sh
"@ | Tee-Object -FilePath (Join-Path $dir 'oneshot_sv2_out.txt')
ssh -o BatchMode=yes pto-b10 "cat /tmp/st_simtvf_parallel/SUMMARY_sv2_raw.txt; echo '---'; for t in sv2_r32_c32_t32_frag_live sv2_r32_c32_t32_reload sv2_r32_c32_t128_frag_live; do echo ==== `$t ====; grep -E 'PASS|FAIL|duration|COMPILE' /tmp/st_simtvf_parallel/opsim_`${t}.log /tmp/st_simtvf_parallel/logs/compile_`${t}.log 2>/dev/null | head -15; done" |
  Tee-Object -FilePath (Join-Path $dir 'SUMMARY_sv2.md')
ssh -o BatchMode=yes pto-b10 "cd /tmp/st_simtvf_parallel && tar -czf /tmp/st_simtvf_sv2_reports.tgz SUMMARY_sv2_raw.txt oneshot_sv2.log logs/compile_sv2* opsim_sv2* so/sv2* sources/*sv2* 2>/dev/null; ls -la /tmp/st_simtvf_sv2_reports.tgz"
scp -o BatchMode=yes pto-b10:/tmp/st_simtvf_sv2_reports.tgz (Join-Path $dir 'st_simtvf_sv2_reports.tgz')
Write-Host "OUT=$dir"
