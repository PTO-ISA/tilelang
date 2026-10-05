/*!
 * \file legalize_parallel_to_pto.cc
 * \brief Stage-1 identity stub for the PTO Parallel legalization stage.
 *
 * Gate B only registers the pass so the formal PTO pipeline order
 * (VerifyParallelToPTO -> LegalizeParallelToPTO -> VectorizeParallelToPTO)
 * is routing-testable. The stub preserves the input IR and claims no semantic
 * support: it neither rewrites nor rejects any construct. The safe scalar
 * If/Select/Min/Max rewrites land with the PR262 control-flow migration
 * (task 5).
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/transform.h>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace tirx::transform;

tvm::transform::Pass LegalizeParallelToPTO() {
  auto pass_func = [](PrimFunc func, const IRModule &mod,
                      const tvm::transform::PassContext &ctx) { return func; };
  return CreatePrimFuncPass(pass_func, 0, "tl.LegalizeParallelToPTO", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LegalizeParallelToPTO",
                        LegalizeParallelToPTO);
}

} // namespace tl
} // namespace tvm
