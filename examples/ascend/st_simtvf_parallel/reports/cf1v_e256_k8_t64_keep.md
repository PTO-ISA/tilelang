# cf1v_e256_k8_t64_keep

- status: **PASS**
- wall_us: 1.01
- 
- layout: tir=cf1v_e256_k8_t64_keep_tir.txt; src=cf1v_e256_k8_t64_keep_source.txt; elems_per_thread≈4.00
- instr: top_opcodes: RV_SMOV:42, RV_VLDI:40, RV_VSTI:40, RV_VCMP_LT:32, RV_VSEL:32, MOV_XD_IMM:17, MOVK:12, RV_VLOOPv2:11, RV_VDUPS:9, RV_SLDI:8, RV_SSHLI:8, RV_SADD:8 | pipes: RVECSU:88, RVECEX:74, SCALAR:52, RVECLD:40, RVECST:40, RVECLP:11, FLOWCTRL:5, MTE2:3
