# cf3v_e256_k8_t64_remat_idx

- status: **PASS**
- wall_us: 1.02
- 
- layout: tir=cf3v_e256_k8_t64_remat_idx_tir.txt; src=cf3v_e256_k8_t64_remat_idx_source.txt; elems_per_thread≈4.00
- instr: top_opcodes: RV_SMOV:50, RV_VSTI:40, RV_VLDI:40, RV_VCI:32, RV_VSEL:32, RV_VCMP_EQ:32, RV_SADDI:32, MOV_XD_IMM:17, MOVK:12, RV_VLOOPv2:11, RV_VDUPS:9, RV_SLDI:8 | pipes: RVECSU:128, RVECEX:106, SCALAR:52, RVECST:40, RVECLD:40, RVECLP:11, FLOWCTRL:5, MTE2:3
