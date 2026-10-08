# sv2_r32_c64_t32_input_stream

- status: **PASS**
- wall_us: 3.6
- 
- layout: launch_bounds=32; tir=sv2_r32_c64_t32_input_stream_tir.txt; src=sv2_r32_c64_t32_input_stream_source.txt
- instr: VF_SIMT us=2.75 cyc=4941 body_instr=745 IPC_proxy=0.151 | top_opcodes: SIMT_F2F:192, SIMT_LDS:127, SIMT_IADD_I:97, SIMT_STS:64, SIMT_FDIV:64, SIMT_FMNMX_I:64, SIMT_FMNMX:32, MOV_XD_IMM:22, MOVK:14, MOV_XD_SPR:13, MOV_SPR_XN:6, ADD_IMM:5 | pipes: RVECEX:453, RVECLD:127, SCALAR:84, RVECST:64, FLOWCTRL:5, PUSHQ:4, MTE3:3, MTE2:3
