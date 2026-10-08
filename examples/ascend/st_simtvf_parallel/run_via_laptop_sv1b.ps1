# Run on LAPTOP — deploy SV1B to pto-b10
$ErrorActionPreference = 'Continue'
$dir = Join-Path $env:USERPROFILE 'st_simtvf_parallel'
if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir | Out-Null }
$tg = Join-Path $dir 'st_simtvf_parallel_suite.tar.gz'
if (-not (Test-Path $tg)) {
  Write-Host "MISSING $tg — place suite tarball first"
  exit 2
}
Write-Host "Uploading suite to pto-b10 and running SV1B oneshot..."
scp -o BatchMode=yes -o ConnectTimeout=90 $tg "pto-b10:/tmp/st_simtvf_parallel_suite.tar.gz"
ssh -o BatchMode=yes -o ConnectTimeout=90 pto-b10 @"
rm -rf /tmp/st_simtvf_parallel_suite && mkdir -p /tmp/st_simtvf_parallel_suite &&
tar -xzf /tmp/st_simtvf_parallel_suite.tar.gz -C /tmp/st_simtvf_parallel_suite &&
sed -i 's/\r$//' /tmp/st_simtvf_parallel_suite/*.sh /tmp/st_simtvf_parallel_suite/*.py /tmp/st_simtvf_parallel_suite/kernels/*.py &&
chmod +x /tmp/st_simtvf_parallel_suite/oneshot_sv1b.sh &&
bash /tmp/st_simtvf_parallel_suite/oneshot_sv1b.sh
"@ | Tee-Object -FilePath (Join-Path $dir 'oneshot_sv1b_out.txt')
ssh -o BatchMode=yes pto-b10 "cat /tmp/st_simtvf_parallel/SUMMARY_sv1b_raw.txt; echo '---'; for t in sv1b_r32_c32_t32_frag_live sv1b_r32_c32_t32_reload sv1b_r32_c32_t128_frag_live; do echo ==== `$t ====; grep -E 'PASS|FAIL|duration|COMPILE' /tmp/st_simtvf_parallel/opsim_`${t}.log /tmp/st_simtvf_parallel/logs/compile_`${t}.log 2>/dev/null | head -15; done" |
  Tee-Object -FilePath (Join-Path $dir 'SUMMARY_sv1b.md')
ssh -o BatchMode=yes pto-b10 "cd /tmp/st_simtvf_parallel && tar -czf /tmp/st_simtvf_sv1b_reports.tgz SUMMARY_sv1b_raw.txt oneshot_sv1b.log logs/compile_sv1b* opsim_sv1b* so/sv1b* sources/*sv1b* 2>/dev/null; ls -la /tmp/st_simtvf_sv1b_reports.tgz"
scp -o BatchMode=yes pto-b10:/tmp/st_simtvf_sv1b_reports.tgz (Join-Path $dir 'st_simtvf_sv1b_reports.tgz')
Write-Host "OUT=$dir"
