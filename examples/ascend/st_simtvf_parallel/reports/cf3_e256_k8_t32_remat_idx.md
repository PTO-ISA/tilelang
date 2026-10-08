# cf3_e256_k8_t32_remat_idx

- status: **PASS**
- wall_us: 1.33
- 
- layout: launch_bounds=32; scores[] sizes=['8']; tir=cf3_e256_k8_t32_remat_idx_tir.txt; src=cf3_e256_k8_t32_remat_idx_source.txt; elems_per_thread≈8.00
- instr: VF_SIMT us=0.47 cyc=851 body_instr=282 IPC_proxy=0.331 | top_opcodes: SIMT_SEL:64, SIMT_ISETP:64, MOV_XD_IMM:25, SIMT_LDS:16, MOVK:16, MOV_XD_SPR:13, SIMT_STS:8, SIMT_LEA:8, SIMT_MOVI:8, MOV_SPR_XN:6, ADD_IMM:5, SHL:4 | pipes: RVECEX:150, SCALAR:90, RVECLD:16, RVECST:8, FLOWCTRL:5, PUSHQ:4, MTE2:4, MTE3:3
