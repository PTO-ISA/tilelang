#!/bin/bash
set +e
tar xzf /tmp/sv2_input_keep.tgz -C /tmp/sv2_ik_unpack 2>/dev/null || true
rm -rf /tmp/sv2_ik_unpack && mkdir -p /tmp/sv2_ik_unpack
tar xzf /tmp/sv2_input_keep.tgz -C /tmp/sv2_ik_unpack
WT=/mnt/fluxdata/happybot/projects/tilelang-st-simtvf-parallel
ST=$WT/examples/ascend/st_simtvf_parallel
cp -f /tmp/sv2_ik_unpack/kernels/sv2_eltwise_bcast_rf.py "$ST/kernels/"
cp -f /tmp/sv2_ik_unpack/kernels/sv2v_eltwise_bcast_rf.py "$ST/kernels/"
cp -f /tmp/sv2_ik_unpack/kernels_ptodsl/sv2d_eltwise_bcast_rf.py "$ST/kernels_ptodsl/"
cp -f /tmp/sv2_ik_unpack/run_opsim_generic.py "$ST/"
cp -f /tmp/sv2_ik_unpack/oneshot_sv2.sh "$ST/"
cp -f /tmp/sv2_ik_unpack/oneshot_ptodsl_phase2.sh "$ST/"
cp -f /tmp/sv2_ik_unpack/reports/ST_RF_KEEP_STREAM_AUDIT.md "$ST/reports/"
cp -f /tmp/sv2_ik_unpack/reports/ST_PTODSL_PHASE2_COMPARE.md "$ST/reports/"
chmod +x "$ST/oneshot_sv2.sh" "$ST/oneshot_ptodsl_phase2.sh" "$ST/_run_sv2d_input.sh" 2>/dev/null
cp -f /tmp/sv2_ik_unpack/_run_sv2d_input.sh "$ST/" 2>/dev/null || true
chmod +x "$ST/_run_sv2d_input.sh" 2>/dev/null || true

echo "==== SV2d input KEEP/STREAM ===="
bash "$ST/_run_sv2d_input.sh"
echo "==== SV2 A oneshot ===="
export SUITE=$ST
bash "$ST/oneshot_sv2.sh"
echo "==== A SUMMARY ===="
cat /tmp/st_simtvf_parallel/SUMMARY_sv2_raw.txt
echo SV2_INPUT_KEEP_DONE
