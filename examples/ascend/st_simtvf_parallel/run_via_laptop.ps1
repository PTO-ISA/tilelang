# Run on LAPTOP-C8QEKIFC (machineId 04894df4-ce02-4d53-a618-932598da9ca2)
$ErrorActionPreference = 'Continue'
$dir = Join-Path $env:USERPROFILE 'st_simtvf_parallel'
if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir | Out-Null }
$tg = Join-Path $dir 'st_simtvf_parallel_suite.tar.gz'
if (-not (Test-Path $tg)) {
  Write-Host "MISSING $tg — place suite tarball first"
  exit 2
}
Write-Host "Uploading suite to pto-b10 and running oneshot..."
scp -o BatchMode=yes -o ConnectTimeout=90 $tg "pto-b10:/tmp/st_simtvf_parallel_suite.tar.gz"
ssh -o BatchMode=yes -o ConnectTimeout=90 pto-b10 "rm -rf /tmp/st_simtvf_parallel_suite && mkdir -p /tmp/st_simtvf_parallel_suite && tar -xzf /tmp/st_simtvf_parallel_suite.tar.gz -C /tmp/st_simtvf_parallel_suite && sed -i 's/\r$//' /tmp/st_simtvf_parallel_suite/*.sh /tmp/st_simtvf_parallel_suite/*.py /tmp/st_simtvf_parallel_suite/kernels/*.py && bash /tmp/st_simtvf_parallel_suite/oneshot_st_simtvf_parallel.sh" |
  Tee-Object -FilePath (Join-Path $dir 'oneshot_out.txt')
ssh -o BatchMode=yes pto-b10 "cat /tmp/st_simtvf_parallel/SUMMARY.md; echo '---RAW---'; cat /tmp/st_simtvf_parallel/SUMMARY_raw.txt" |
  Tee-Object -FilePath (Join-Path $dir 'SUMMARY.md')
# pull reports
ssh -o BatchMode=yes pto-b10 "cd /tmp/st_simtvf_parallel && tar -czf /tmp/st_simtvf_reports.tgz SUMMARY.md SUMMARY_raw.txt reports 2>/dev/null; ls -la /tmp/st_simtvf_reports.tgz"
scp -o BatchMode=yes pto-b10:/tmp/st_simtvf_reports.tgz (Join-Path $dir 'st_simtvf_reports.tgz')
Write-Host "OUT=$dir"
