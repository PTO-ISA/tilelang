/*!
 * \file vectorize_parallel_to_pto.cc
 * \brief Stage-1 identity stub for the PTO Parallel vectorization stage.
 *
 * Gate B only registers the pass so the formal PTO pipeline order
 * (VerifyParallelToPTO -> LegalizeParallelToPTO -> VectorizeParallelToPTO)
 * is routing-testable. The stub preserves the input IR and claims no semantic
 * support: it neither rewrites nor rejects any construct. The real lowering
 * (E/P/Q/remaining chunk loop, masks, VMI emission and the internal Reduce
 * module) lands with the PR269 migration stage (task 3b) and later stages.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/transform.h>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace tirx::transform;

tvm::transform::Pass VectorizeParallelToPTO() {
  auto pass_func = [](PrimFunc func, const IRModule &mod,
                      const tvm::transform::PassContext &ctx) { return func; };
  return CreatePrimFuncPass(pass_func, 0, "tl.VectorizeParallelToPTO", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.VectorizeParallelToPTO",
                        VectorizeParallelToPTO);
}

} // namespace tl
} // namespace tvm
