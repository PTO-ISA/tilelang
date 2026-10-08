# sp5_n64_h128_g32_qg32_t32_interleave

- status: **PASS**
- wall_us: 31.32
- 
- layout: launch_bounds=32; tir=sp5_n64_h128_g32_qg32_t32_interleave_tir.txt; src=sp5_n64_h128_g32_qg32_t32_interleave_source.txt
- instr: VF_SIMT us=29.76 cyc=53576 body_instr=9510 IPC_proxy=0.178 | top_opcodes: SIMT_LEA:1632, SIMT_IADD_I:1571, SIMT_IADD_X:1152, SIMT_LDS:1088, SIMT_STS:1024, SIMT_FMUL:1024, SIMT_SHFI:547, SIMT_ISETP_I:545, SIMT_BRANCH:545, SIMT_IMUL:99, SIMT_IADD:66, SIMT_MOV:65 | pipes: RVECEX:6739, RVECLD:1088, RVECST:1024, RVECLP:549, SCALAR:92, FLOWCTRL:5, PUSHQ:4, MTE2:4
