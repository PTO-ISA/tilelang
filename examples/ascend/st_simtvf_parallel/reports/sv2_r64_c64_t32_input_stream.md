# sv2_r64_c64_t32_input_stream

- status: **PASS**
- wall_us: 8.34
- 
- layout: launch_bounds=32; tir=sv2_r64_c64_t32_input_stream_tir.txt; src=sv2_r64_c64_t32_input_stream_source.txt
- instr: VF_SIMT us=7.36 cyc=13253 body_instr=1625 IPC_proxy=0.123 | top_opcodes: SIMT_F2F:384, SIMT_LDS:255, SIMT_IADD_I:225, SIMT_STS:128, SIMT_FDIV:128, SIMT_FMNMX_I:128, SIMT_LDK:103, SIMT_STK:103, SIMT_FMNMX:64, MOV_XD_IMM:22, MOVK:14, MOV_XD_SPR:13 | pipes: RVECEX:935, RVECLD:358, RVECST:231, SCALAR:84, FLOWCTRL:5, PUSHQ:4, MTE3:3, MTE2:3
