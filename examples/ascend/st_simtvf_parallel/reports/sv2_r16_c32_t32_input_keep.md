# sv2_r16_c32_t32_input_keep

- status: **PASS**
- wall_us: 1.16
- 
- layout: launch_bounds=32; x[] sizes=['16']; tir=sv2_r16_c32_t32_input_keep_tir.txt; src=sv2_r16_c32_t32_input_keep_source.txt
- instr: VF_SIMT us=0.29 cyc=530 body_instr=202 IPC_proxy=0.381 | top_opcodes: SIMT_F2F:32, MOV_XD_IMM:22, SIMT_LDS:16, SIMT_STS:16, SIMT_FDIV:16, SIMT_FMNMX_I:16, MOVK:14, MOV_XD_SPR:13, MOV_SPR_XN:6, ADD_IMM:5, ADD:4, SHL:4 | pipes: SCALAR:84, RVECEX:69, RVECLD:16, RVECST:16, FLOWCTRL:5, PUSHQ:4, MTE3:3, MTE2:3
