# cf1_e256_k8_t32_keep

- status: **PASS**
- wall_us: 0.97
- 
- layout: launch_bounds=32; scores[] sizes=['8']; tir=cf1_e256_k8_t32_keep_tir.txt; src=cf1_e256_k8_t32_keep_source.txt; elems_per_thread≈8.00
- instr: VF_SIMT us=0.17 cyc=309 body_instr=266 IPC_proxy=0.861 | top_opcodes: SIMT_FSETP:64, SIMT_SEL:64, MOV_XD_IMM:25, SIMT_LDS:16, MOVK:16, MOV_XD_SPR:13, SIMT_STS:8, MOV_SPR_XN:6, ADD_IMM:5, SHL:4, ADD:4, SBITSET:4 | pipes: RVECEX:134, SCALAR:90, RVECLD:16, RVECST:8, FLOWCTRL:5, MTE2:4, PUSHQ:4, MTE3:3
