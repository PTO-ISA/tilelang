# sp1_t32_k2_h128_g32_t32_remat_pos

- status: **PASS**
- wall_us: 9.94
- 
- layout: launch_bounds=32; tir=sp1_t32_k2_h128_g32_t32_remat_pos_tir.txt; src=sp1_t32_k2_h128_g32_t32_remat_pos_source.txt
- instr: VF_SIMT us=8.34 cyc=15013 body_instr=2570 IPC_proxy=0.171 | top_opcodes: SIMT_IADD_I:431, SIMT_STS:327, SIMT_SHFI:323, SIMT_LDS:292, SIMT_BRANCH:192, SIMT_ISETP_I:165, SIMT_LEA:160, SIMT_LOP3:160, SIMT_IADD_X:126, SIMT_IADD:65, SIMT_MOV:64, SIMT_BAR_THREAD_BLOCK:64 | pipes: RVECEX:1568, RVECST:327, RVECLD:292, RVECLP:193, SCALAR:106, RVECSU:64, MTE2:6, FLOWCTRL:5
