# Run on LAPTOP — upload suite + full SV1–SV9 Simt oneshot on pto-b10
$ErrorActionPreference = 'Continue'
$dir = Join-Path $env:USERPROFILE 'st_simtvf_parallel'
if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir | Out-Null }
$tg = Join-Path $dir 'st_simtvf_parallel_sv1_sv9_src.tgz'
$b64 = Join-Path $dir 'st_simtvf_parallel_sv1_sv9_src.tgz.b64'
if (-not (Test-Path $tg)) {
  if (Test-Path $b64) {
    Write-Host "Decoding b64 -> tgz"
    [IO.File]::WriteAllBytes($tg, [Convert]::FromBase64String((Get-Content -Raw $b64)))
  } else {
    Write-Host "MISSING $tg — place suite tarball first"
    exit 2
  }
}
Write-Host "Uploading suite to pto-b10 and running SV1–SV9 oneshot..."
scp -o BatchMode=yes -o ConnectTimeout=90 $tg "pto-b10:/tmp/st_simtvf_parallel_sv1_sv9_src.tgz"
ssh -o BatchMode=yes -o ConnectTimeout=90 pto-b10 @"
rm -rf /tmp/st_simtvf_parallel_suite && mkdir -p /tmp/st_simtvf_parallel_suite &&
tar -xzf /tmp/st_simtvf_parallel_sv1_sv9_src.tgz -C /tmp/st_simtvf_parallel_suite &&
sed -i 's/\r`$//' /tmp/st_simtvf_parallel_suite/*.sh /tmp/st_simtvf_parallel_suite/*.py /tmp/st_simtvf_parallel_suite/kernels/*.py &&
chmod +x /tmp/st_simtvf_parallel_suite/oneshot_sv1_sv9.sh &&
bash /tmp/st_simtvf_parallel_suite/oneshot_sv1_sv9.sh
"@ | Tee-Object -FilePath (Join-Path $dir 'oneshot_sv1_sv9_out.txt')
ssh -o BatchMode=yes pto-b10 "cat /tmp/st_simtvf_parallel/SUMMARY_sv1_sv9_raw.txt; echo '---'; ls /tmp/st_simtvf_parallel/so/sv{1,2,3,4,5,6,7,8,9}* 2>/dev/null | head -80" |
  Tee-Object -FilePath (Join-Path $dir 'SUMMARY_sv1_sv9.md')
ssh -o BatchMode=yes pto-b10 "cd /tmp/st_simtvf_parallel && tar -czf /tmp/st_simtvf_sv1_sv9_reports.tgz SUMMARY_sv1_sv9_raw.txt oneshot_sv1_sv9.log harvest_sv1_sv9.log logs/compile_sv{1,2,3,4,5,6,7,8,9}* opsim_sv{1,2,3,4,5,6,7,8,9}* so/sv{1,2,3,4,5,6,7,8,9}* sources/*sv{1,2,3,4,5,6,7,8,9}* 2>/dev/null; ls -la /tmp/st_simtvf_sv1_sv9_reports.tgz"
scp -o BatchMode=yes pto-b10:/tmp/st_simtvf_sv1_sv9_reports.tgz (Join-Path $dir 'st_simtvf_sv1_sv9_reports.tgz')
Write-Host "OUT=$dir"
