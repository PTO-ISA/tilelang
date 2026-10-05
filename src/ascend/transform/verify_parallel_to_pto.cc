/*!
 * \file verify_parallel_to_pto.cc
 * \brief Stage-1 identity stub for the PTO Parallel verification contract.
 *
 * Gate B only registers the pass so the formal PTO pipeline order
 * (VerifyParallelToPTO -> LegalizeParallelToPTO -> VectorizeParallelToPTO)
 * is routing-testable. The stub preserves the input IR and claims no semantic
 * support: it neither rewrites nor rejects any construct. The real contract
 * (logical/padded extent mapping checks and the rejection diagnostics) lands
 * with the PR269 migration stage (task 3a).
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/transform.h>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace tirx::transform;

tvm::transform::Pass VerifyParallelToPTO() {
  auto pass_func = [](PrimFunc func, const IRModule &mod,
                      const tvm::transform::PassContext &ctx) { return func; };
  return CreatePrimFuncPass(pass_func, 0, "tl.VerifyParallelToPTO", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.VerifyParallelToPTO",
                        VerifyParallelToPTO);
}

} // namespace tl
} // namespace tvm
