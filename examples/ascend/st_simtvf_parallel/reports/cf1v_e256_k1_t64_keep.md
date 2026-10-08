# cf1v_e256_k1_t64_keep

- status: **PASS**
- wall_us: 1.17
- 
- layout: tir=cf1v_e256_k1_t64_keep_tir.txt; src=cf1v_e256_k1_t64_keep_source.txt; elems_per_thread≈4.00
- instr: top_opcodes: MOV_XD_IMM:17, RV_VSTI:12, RV_VLDI:12, MOVK:12, RV_SMOV:7, MOV_XD_SPR:6, RV_VSEL:4, RV_VCMP_LT:4, MOV_SRC_TO_DST_ALIGNv2:3, RV_SEND:3, SHL:3, INSERT_XD_XN:3 | pipes: SCALAR:52, RVECSU:13, RVECST:12, RVECLD:12, RVECEX:11, FLOWCTRL:5, MTE2:3, RVECLP:3
