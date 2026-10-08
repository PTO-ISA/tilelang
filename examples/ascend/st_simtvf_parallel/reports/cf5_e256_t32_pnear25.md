# cf5_e256_t32_pnear25

- status: **PASS**
- wall_us: 1.18
- 
- layout: launch_bounds=32; tir=cf5_e256_t32_pnear25_tir.txt; src=cf5_e256_t32_pnear25_source.txt; elems_per_thread≈8.00
- instr: VF_SIMT us=0.36 cyc=656 body_instr=159 IPC_proxy=0.242 | top_opcodes: MOV_XD_IMM:23, SIMT_LDS:16, MOVK:14, MOV_XD_SPR:13, SIMT_STS:8, SIMT_FDIV:8, SIMT_FSETP_I:7, SIMT_SEL:7, MOV_SPR_XN:6, ADD_IMM:5, SHL:4, ADD:4 | pipes: SCALAR:86, RVECEX:31, RVECLD:16, RVECST:8, FLOWCTRL:5, PUSHQ:4, MTE2:4, MTE3:3
