# Run on LAPTOP — full VMI twins SV1V–SV9V (SimdVF / target=pto / overlay)
# Prefer after Simt oneshot_sv1_sv9 / Case-3 Simt is idle (script self-guards on pto-b10).
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
Write-Host "Uploading suite to pto-b10 and running SV1V–SV9V (VMI) oneshot..."
scp -o BatchMode=yes -o ConnectTimeout=90 $tg "pto-b10:/tmp/st_simtvf_parallel_suite.tar.gz"
ssh -o BatchMode=yes -o ConnectTimeout=90 pto-b10 @"
rm -rf /tmp/st_simtvf_parallel_suite && mkdir -p /tmp/st_simtvf_parallel_suite &&
tar -xzf /tmp/st_simtvf_parallel_suite.tar.gz -C /tmp/st_simtvf_parallel_suite &&
sed -i 's/\r$//' /tmp/st_simtvf_parallel_suite/*.sh /tmp/st_simtvf_parallel_suite/*.py /tmp/st_simtvf_parallel_suite/kernels/*.py &&
chmod +x /tmp/st_simtvf_parallel_suite/oneshot_sv1v_sv9v.sh &&
bash /tmp/st_simtvf_parallel_suite/oneshot_sv1v_sv9v.sh
"@ | Tee-Object -FilePath (Join-Path $dir 'oneshot_sv1v_sv9v_out.txt')
ssh -o BatchMode=yes pto-b10 "cat /tmp/st_simtvf_parallel/SUMMARY_sv1v_sv9v_raw.txt; echo '---'; ls -la /tmp/st_simtvf_parallel/so/sv{1,2,3,4,5,6,7,8,9}v* 2>/dev/null | head -40" |
  Tee-Object -FilePath (Join-Path $dir 'SUMMARY_sv1v_sv9v.md')
ssh -o BatchMode=yes pto-b10 "cd /tmp/st_simtvf_parallel && tar -czf /tmp/st_simtvf_sv1v_sv9v_reports.tgz SUMMARY_sv1v_sv9v_raw.txt oneshot_sv1v_sv9v.log logs/compile_sv*v* opsim_sv*v* so/sv*v* sources/*sv*v* 2>/dev/null; ls -la /tmp/st_simtvf_sv1v_sv9v_reports.tgz"
scp -o BatchMode=yes pto-b10:/tmp/st_simtvf_sv1v_sv9v_reports.tgz (Join-Path $dir 'st_simtvf_sv1v_sv9v_reports.tgz')
Write-Host "OUT=$dir"
