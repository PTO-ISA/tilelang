# sp2_t32_k2_h128_g32_t32_sf1_w1_remat_pos

- status: **PASS**
- wall_us: 14.5
- 
- layout: launch_bounds=32; acc[] sizes=['4']; acc[4]; tir=sp2_t32_k2_h128_g32_t32_sf1_w1_remat_pos_tir.txt; src=sp2_t32_k2_h128_g32_t32_sf1_w1_remat_pos_source.txt
- instr: VF_SIMT us=12.92 cyc=23259 body_instr=4011 IPC_proxy=0.172 | top_opcodes: SIMT_IADD_I:878, SIMT_LDS:639, SIMT_LEA:399, SIMT_STS:384, SIMT_BRANCH:288, SIMT_FMUL:252, SIMT_ISETP_I:224, SIMT_SHFI:212, SIMT_IADD_X:158, SIMT_FFMA_I:128, SIMT_FFMA:124, SIMT_MOV:96 | pipes: RVECEX:2514, RVECLD:639, RVECST:384, RVECLP:289, SCALAR:102, RVECSU:64, MTE2:6, FLOWCTRL:5
