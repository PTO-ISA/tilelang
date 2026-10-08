# cf6_e256_t32_pnear5

- status: **PASS**
- wall_us: 1.37
- 
- layout: launch_bounds=32; x[] sizes=['8']; tir=cf6_e256_t32_pnear5_tir.txt; src=cf6_e256_t32_pnear5_source.txt; elems_per_thread≈8.00
- instr: VF_SIMT us=0.53 cyc=961 body_instr=202 IPC_proxy=0.210 | top_opcodes: SIMT_FFMA_I:24, SIMT_FMUL:24, MOV_XD_IMM:22, MOVK:14, MOV_XD_SPR:13, SIMT_MOVI:9, SIMT_LDS:8, SIMT_STS:8, SIMT_FDIV_I:8, SIMT_FSETP:8, SIMT_MOV:7, MOV_SPR_XN:6 | pipes: RVECEX:85, SCALAR:84, RVECLD:8, RVECST:8, FLOWCTRL:5, PUSHQ:4, MTE3:3, MTE2:3
