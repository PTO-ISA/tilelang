# sv2_r32_c32_t32_input_stream

- status: **PASS**
- wall_us: 2.41
- 
- layout: launch_bounds=32; tir=sv2_r32_c32_t32_input_stream_tir.txt; src=sv2_r32_c32_t32_input_stream_source.txt
- instr: VF_SIMT us=1.57 cyc=2828 body_instr=393 IPC_proxy=0.139 | top_opcodes: SIMT_F2F:96, SIMT_LDS:63, SIMT_IADD_I:33, SIMT_STS:32, SIMT_FDIV:32, SIMT_FMNMX_I:32, MOV_XD_IMM:22, MOVK:14, MOV_XD_SPR:13, MOV_SPR_XN:6, ADD_IMM:5, SHL:4 | pipes: RVECEX:197, SCALAR:84, RVECLD:63, RVECST:32, FLOWCTRL:5, PUSHQ:4, MTE3:3, MTE2:3
