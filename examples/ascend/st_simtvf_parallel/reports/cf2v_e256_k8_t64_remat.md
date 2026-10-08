# cf2v_e256_k8_t64_remat

- status: **PASS**
- wall_us: 1.04
- 
- layout: tir=cf2v_e256_k8_t64_remat_tir.txt; src=cf2v_e256_k8_t64_remat_source.txt; elems_per_thread≈4.00
- instr: top_opcodes: RV_VLDI:100, RV_VSTI:68, RV_SMOV:65, RV_VCMP_LT:32, RV_VSEL:32, RV_VLOOPv2:18, MOV_XD_IMM:17, MOVK:12, RV_VDUPS:9, RV_SLDI:8, RV_SADD:8, RV_SSHLI:8 | pipes: RVECSU:110, RVECLD:100, RVECEX:74, RVECST:68, SCALAR:52, RVECLP:18, FLOWCTRL:5, MTE2:3
