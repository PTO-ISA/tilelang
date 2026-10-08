# cf1_e256_k1_t32_keep

- status: **PASS**
- wall_us: 1.15
- 
- layout: launch_bounds=32; scores[] sizes=['8']; tir=cf1_e256_k1_t32_keep_tir.txt; src=cf1_e256_k1_t32_keep_source.txt; elems_per_thread≈8.00
- instr: VF_SIMT us=0.32 cyc=576 body_instr=147 IPC_proxy=0.255 | top_opcodes: MOV_XD_IMM:25, MOVK:16, MOV_XD_SPR:13, SIMT_LDS:9, SIMT_STS:8, SIMT_FSETP:8, SIMT_SEL:8, MOV_SPR_XN:6, ADD_IMM:5, SHL:4, ADD:4, SBITSET:4 | pipes: SCALAR:90, RVECEX:22, RVECLD:9, RVECST:8, FLOWCTRL:5, PUSHQ:4, MTE2:4, MTE3:3
