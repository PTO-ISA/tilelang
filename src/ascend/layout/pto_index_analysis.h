/*!
 * \file pto_index_analysis.h
 * \brief Low-level read-only index/address analysis shared by PTO lane-layout
 * planning and the PTO Verify/Vectorize consumers.
 *
 * These helpers are the single implementation of the bind-alias resolution
 * rules (cycle/depth failure, expression-size budget, loaded-value identity)
 * and the address-node whitelist; planning and consuming code must not
 * re-derive their own variants.
 */

#ifndef TVM_TL_ASCEND_LAYOUT_PTO_INDEX_ANALYSIS_H_
#define TVM_TL_ASCEND_LAYOUT_PTO_INDEX_ANALYSIS_H_

#include <tvm/arith/analyzer.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/stmt.h>

#include <stdexcept>
#include <string>
#include <utility>

namespace tvm {
namespace tl {
namespace pto {

using ffi::Array;
using ffi::Map;
using ffi::Optional;
using tirx::Var;
using tirx::VarNode;

/*! Error type thrown by the shared analysis on unsupported input; both
 * passes translate it into their own diagnostics. */
class PtoAnalysisError : public std::runtime_error {
public:
  explicit PtoAnalysisError(const std::string &msg) : std::runtime_error(msg) {}
};

/*! Substitute `from` -> `to` inside an expression tree. */
PrimExpr SubstituteVar(const PrimExpr &expr, const VarNode *from, PrimExpr to);

/*! Resolve Bind aliases inside an index expression recursively with a
 * visited set (cycle-safe) and a depth guard that *rejects* instead of
 * silently stopping. Loaded Bind definitions remain SSA leaves; they are
 * never expanded into another memory read. Returns nullopt when a chain
 * exceeds the resolution limit; the caller must reject the access. */
Optional<PrimExpr> ResolveIndexAliases(const PrimExpr &index,
                                       const Map<Var, PrimExpr> &bind_env);

/*! Check the address expression whitelist, including ordinary UB and local loads.
 * Opaque and side-effecting calls remain unsupported. */
bool IsSupportedAddressExpr(const PrimExpr &expr);

/*! Ordinary UB scope ("shared" / "shared.dyn") for indexed memory payloads. */
bool IsOrdinaryUbScope(const std::string &scope);

/*! Scalar 8-, 16- or 32-bit integer — the required element type of loaded index
 * values and emitted element offsets. */
bool IsPtoIndexDType(DataType dtype);

/*! Pure scalar address arithmetic independent of the parallel unit's values.
 * Preserve these coordinates as scalar expressions, including
 * compiler-generated buffer-version coordinates, instead of converting them to
 * vector values. */
bool IsPtoScalarCoordinate(const PrimExpr &expr, const Var &vector_var,
                           const ffi::Map<Var, PrimExpr> &bind_env,
                           const Var &lane_var = Var());

struct MemoryAccess;
struct PtoInverseMapping;

/*! Prove an integer coordinate is uniform within the canonical lane/chunk
 * domain. Loaded Bind definitions retain the same SSA symbols as address
 * classification, so algebraic coordinate simplification cannot speculate or
 * re-read a saved index value. */
bool IsPtoLaneUniform(const PrimExpr &expr, const tirx::Var &vector_var,
                      const PtoInverseMapping &mapping,
                      const Map<Var, PrimExpr> &bind_env);

/*! Check that the planned offset has a supported scalar/VMI expression
 * lowering under the selected lane mapping. Return inputs of its vector
 * int32-to-int16 casts for the B16 range check. */
Array<PrimExpr> CheckPtoIndexedOffsetExpr(const PrimExpr &offset,
                                          const Var &vector_var,
                                          const PtoInverseMapping &mapping,
                                          const Map<Var, PrimExpr> &bind_env);

/*! View metadata used by the physical address at this access. */
ffi::Array<PrimExpr> PtoAddressMetadata(const tirx::Buffer &buffer,
                                        const ffi::Array<PrimExpr> &indices);

/*! Analyze one access under a canonical inverse lane mapping. Loaded values
 * receive analysis-only symbols whose lane dependence follows their defining
 * access, so a varying loaded Bind can never masquerade as a uniform base.
 * These symbols never enter the returned expressions or generated TIR.
 * Indexed offset encoding is checked separately by PlanPtoIndexedAccess.
 */
MemoryAccess AnalyzePtoAddress(const tirx::Buffer &buffer,
                               const ffi::Array<PrimExpr> &indices,
                               bool is_write, const Var &vector_var,
                               const PtoInverseMapping &mapping,
                               const Map<Var, PrimExpr> &bind_env);

/*! Compute the padded fragment-input extent P = ceil(E / lanes) * lanes with
 * the shared overflow guard. The final execution space is int32-bounded
 * (the PTO lane loop uses int32 indices), so an extent above the int32
 * range is rejected before the padding addition could overflow int64.
 * Returns the pair (pad, padded); throws PtoAnalysisError with a
 * classified reason on rejection. The callers translate the error into
 * their own diagnostics. */
std::pair<int64_t, int64_t> CheckedPtoPadding(int64_t extent, int64_t lanes);

} // namespace pto
} // namespace tl
} // namespace tvm

#endif // TVM_TL_ASCEND_LAYOUT_PTO_INDEX_ANALYSIS_H_
