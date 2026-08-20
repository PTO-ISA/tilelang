"""pytest test for example_simdvf_scalar_topk_scalar_write.py — scalar GM write topk."""

import pytest
import torch
import tilelang
import tilelang.testing

from example_simdvf_scalar_topk_scalar_write import make_kernel, ref_program

NUM_EXPERTS = 128
NUM_TOPK = 8
NUM_CORES = 32


def test_simdvf_scalar_topk_scalar_write():
    kernel = make_kernel()
    device = torch.device("npu")
    n = 4096
    torch.manual_seed(42)
    logits = torch.randn(n, NUM_EXPERTS, dtype=torch.float32, device=device)
    out = kernel(logits)
    torch.npu.synchronize()
    assert torch.equal(out.cpu(), ref_program(logits, NUM_TOPK).cpu())


@pytest.mark.pto
@pytest.mark.skip(reason="PTO VMI has no scalar GM DCache-bypass store")
def test_pto_scalar_topk_scalar_write():
    """The shared scalar top-k PTO test covers the algorithm without ASC GM bypass."""


if __name__ == "__main__":
    tilelang.testing.main()
