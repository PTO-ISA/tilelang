# sp2_t32_k2_h128_g32_t32_sf1_w1_keep_pos

- status: **PASS**
- wall_us: 15.66
- 
- layout: launch_bounds=32; acc[] sizes=['4']; acc[4]; tir=sp2_t32_k2_h128_g32_t32_sf1_w1_keep_pos_tir.txt; src=sp2_t32_k2_h128_g32_t32_sf1_w1_keep_pos_source.txt
- instr: VF_SIMT us=14.08 cyc=25344 body_instr=4235 IPC_proxy=0.167 | top_opcodes: SIMT_IADD_I:974, SIMT_LDS:703, SIMT_STS:448, SIMT_LEA:399, SIMT_BRANCH:288, SIMT_FMUL:252, SIMT_ISETP_I:224, SIMT_SHFI:212, SIMT_IADD_X:158, SIMT_FFMA_I:128, SIMT_FFMA:124, SIMT_MOV:96 | pipes: RVECEX:2610, RVECLD:703, RVECST:448, RVECLP:289, SCALAR:102, RVECSU:64, MTE2:6, FLOWCTRL:5
