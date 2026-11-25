/*!
 * \file parallel_to_pto_utils.h
 * \brief Shared read-only analysis helpers for PTO Parallel vectorization.
 *
 * VerifyParallelToPTO proves owner legality and access-plan admission.
 * VectorizeParallelToPTO resolves the current owner objects and reconstructs
 * descriptors; address planning uses the same lexical facts as verification.
 * LegalizeParallelToPTO must preserve allocation scope, data, dtype, shape,
 * strides, storage offsets and coordinate bounds between these passes. A
 * legalization that changes those premises must verify its result again.
 */

#ifndef TVM_TL_ASCEND_TRANSFORM_PARALLEL_TO_PTO_UTILS_H_
#define TVM_TL_ASCEND_TRANSFORM_PARALLEL_TO_PTO_UTILS_H_

#include "../op/pto_finalize_reducer.h"
#include "support/check.h"
#include <tvm/arith/analyzer.h>
#include <tvm/arith/iter_affine_map.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/stmt.h>

#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "../../layout/layout.h"
#include "../../op/builtin.h"
#include "../target_utils.h"
#include "ascend/layout/pto_index_analysis.h"
#include "ascend/layout/pto_parallel_layout.h"

namespace tvm {
namespace tl {

using namespace tirx;
using ffi::Array;
using ffi::GetRef;
using ffi::Map;
using ffi::ObjectRef;
using ffi::Optional;

namespace pto {

/*! Scope the analyzer to a For body, including symbolic min and extent. */
class ScopedPtoLoopBounds {
public:
  ScopedPtoLoopBounds(arith::Analyzer *&analyzer, const For &loop);
  ~ScopedPtoLoopBounds();

private:
  arith::Analyzer *&active_;
  arith::Analyzer *outer_;
  std::unique_ptr<arith::Analyzer> inner_;
  With<arith::ConstraintContext> constraints_;
};

/*! Explicit legacy SIMD instructions already operate on vector blocks.
 * Their containing Parallel nest retains its pre-VMI treatment. */
bool UsesExplicitSIMD(const Stmt &stmt);

/*!
 * \brief Classification of one buffer access against the lane mapping.
 *
 * The classification describes how the
 * *address* varies over lanes/chunk, not the loaded data.
 */
enum class AccessPattern {
  kUnknown = 0,
  /*! Address = start + lane for lane in [0, L): continuous vector. */
  kContinuous,
  /*! Address independent of lane at the current program point. */
  kLaneUniform,
  /*! Per-lane element offsets lowered by gather/scatter. */
  kIndexed,
};

std::string AccessPatternToString(AccessPattern pattern);

/*!
 * \brief Result of the shared single-access analysis.
 *
 * Verify consumes pattern legality; Vectorize additionally consumes the
 * start indices/offset to build tl.access_ptr.
 */
struct MemoryAccess {
  Buffer buffer;
  bool is_write = false;

  /*! Source element offset; convert at the current source program point. */
  PrimExpr original_offset;
  /*! Unflattened source indices, preserving explicit casts. */
  Array<PrimExpr> source_indices;
  /*! indices with lane bound to 0: the start element of this vector access.
   */
  Array<PrimExpr> start_indices;
  AccessPattern pattern = AccessPattern::kUnknown;

  bool RequiresIndexedInstruction() const {
    return pattern == AccessPattern::kIndexed ||
           (is_write && pattern == AccessPattern::kLaneUniform);
  }
};

/*!
 * \brief The inverse lane mapping of a 1D PTO parallel loop.
 *
 * For a 1D loop of extent E with lanes L (E % L == 0): the fragment is
 * forward_thread = i % L, forward_index = [i // L]. Its inverse recovers
 *   i = p * L + lane
 * where p is the per-lane chunk position.
 */
struct PtoInverseMapping {
  /*! Lane variable of the enclosing SIMD_VF (analysis placeholder). */
  Var lane_var;
  /*! Chunk position variable (per-lane column index). */
  Var chunk_var;
  int64_t lanes = 0;
  /*! Logical extent E of the vectorized dimension (the original loop
   * extent, without padding). */
  int64_t logical_extent = 0;
  /*! Padded extent P = ceil(E / L) * L of the fragment input space on the
   * vectorized dimension. E == P for divisible extents. */
  int64_t padded_extent = 0;
  /*! Chunk count Q = P / L — the per-row chunk range of the vectorized
   * dimension (never the global position K*Q). */
  int64_t chunk_count = 0;
  /*! Tail lanes E % L (0 when E is divisible by L). */
  int64_t tail_lanes = 0;
  /*! Inverse expression of the vectorized coordinate:
   * logical index = chunk * L + lane. */
  PrimExpr index_expr;

  // ------------------------------------------------------------------
  // 2D units. A 2D unit is a two-layer adjacent
  // Parallel nest; the layout annotation sits on the outermost For and
  // the Fragment input coordinates stay (i, j) regardless of which
  // dimension is vectorized.
  // ------------------------------------------------------------------
  /*! True when the unit is a 2D nest. */
  bool is_2d = false;
  /*! Original loop vars of the nest (i = outer, j = inner). */
  Var outer_var;
  Var inner_var;
  /*! Extents of the two iterations (i and j). */
  int64_t outer_extent = 0;
  int64_t inner_extent = 0;
  /*! Vectorized dimension: true selects the inner (j) dim. */
  bool select_inner = true;

  /*! The original loop variable kept as a serial loop in a 2D unit. */
  Optional<Var> SerialVar() const {
    if (!is_2d) {
      return {};
    }
    return select_inner ? outer_var : inner_var;
  }

  /*! Extent of the kept serial loop, or zero for a 1D unit. */
  int64_t SerialExtent() const {
    return is_2d ? (select_inner ? outer_extent : inner_extent) : 0;
  }
};

/*! Invert the pto_parallel_loop_layout fragment of a Parallel unit.
 * Accepts a 1D Parallel For, or the outermost For of a two-layer adjacent
 * Parallel nest (2D). Throws PtoAnalysisError when the fragment is
 * missing, has an unsupported shape, or cannot be inverted. */
PtoInverseMapping InvertPtoLaneLayout(const For &loop);

/*! Classify one buffer access (load or store indices) against the inverse
 * mapping. \p indices are the buffer indices in the original loop body
 * expressed over the loop variable; \p loop_var is the original parallel
 * loop variable. \p bind_env maps Bind vars in scope to their defining
 * expressions so index aliases are resolved before classification
 * pass an empty map when no binds are in scope.
 *
 * The helper resolves bind aliases, substitutes
 * loop_var -> chunk * L + lane_var, computes ElemOffset, and proves the
 * access pattern over the full lane range (continuous:
 * data_offset == start_offset + lane as an identity; lane-uniform: the
 * offset does not depend on lane). Returns a fully populated MemoryAccess
 * or throws PtoAnalysisError with a reason. */
MemoryAccess AnalyzeBufferAccess(const Buffer &buffer,
                                 const Array<PrimExpr> &indices, bool is_write,
                                 const Var &loop_var,
                                 const PtoInverseMapping &mapping,
                                 const Map<Var, PrimExpr> &bind_env = {});

/*! The existing allocation described by an indexed access. Resolving an owner
 * does not allocate, copy or initialize storage. */
struct IndexedMemoryOwner {
  Buffer buffer;
  PrimExpr origin;
  PrimExpr extent;
};

/*! Shared UB descriptor primitives for the explicit VMI legalization path. */
void ValidateUbBuffer(const Buffer &buffer, const std::string &op_name,
                      const Span &span);
Array<PrimExpr> BufferZeros(const Buffer &buffer);
std::string BufferContext(const Buffer &buffer);
PrimExpr MakeDescriptor(const Buffer &buffer, const Array<PrimExpr> &starts,
                        const PrimExpr &extent, int rw_mask);
bool RepresentableInInt64(arith::Analyzer *analyzer, const PrimExpr &value);
PrimExpr WidenToInt64(const PrimExpr &value);

struct ExpressionStabilityScan {
  bool has_load = false;
  bool has_call = false;
  std::string first_unstable_type;
};

bool StableIndexedAddressNode(const ObjectRef &node);
ExpressionStabilityScan
ScanExpressionStability(const PrimExpr &expr,
                        bool (*is_stable)(const ObjectRef &));

/*! Prove the physical owner's tight layout and stable capacity. */
IndexedMemoryOwner CheckIndexedAllocationStorage(arith::Analyzer *analyzer,
                                                 const Buffer &owner,
                                                 const std::string &op_name,
                                                 const Span &span);

/*! Prove a dtype-changing view lies within the allocation. */
void CheckIndexedTypedView(arith::Analyzer *analyzer, const Buffer &view,
                           const Buffer &owner, const PrimExpr &typed_extent,
                           const std::string &op_name, const Span &span);

/*! Find the unique active allocation sharing the source buffer's data Var. */
Buffer FindAllocationOwner(const std::vector<Buffer> &active_allocations,
                           const Buffer &source, const std::string &op_name,
                           const Span &span = Span());

/*! Check that an allocation can describe a same-width Parallel access. */
IndexedMemoryOwner CheckIndexedMemoryOwner(const Buffer &source,
                                           const Buffer &owner,
                                           arith::Analyzer *analyzer,
                                           const std::string &context,
                                           const Span &span = Span());

/*! Reconstruct the current owner's origin and capacity after Verify. The
 * descriptor uses allocation-time shape Vars, regardless of analyzer facts at
 * the access. Vectorize recomputes the offset encoding from those lexical facts
 * while reusing Verify's owner storage proof across LegalizeParallelToPTO. */
IndexedMemoryOwner DescribeIndexedMemoryOwner(const Buffer &source,
                                              const Buffer &owner);

/*! Build a whole-owner may-access descriptor. Scatter does not assert dense or
 * unconditional write coverage through this descriptor. */
PrimExpr MakeIndexedMemoryDescriptor(const IndexedMemoryOwner &owner,
                                     bool is_write,
                                     const Optional<PrimExpr> &version = {},
                                     const Optional<PrimExpr> &extent = {});

/*! Scalar instruction base and vector element offsets. Computed coordinates
 * may reference the mapping's chunk or lane variables. Loaded indices retain
 * their source expressions. */
struct PtoIndexedAccessPlan {
  PrimExpr base;
  PrimExpr offsets;
  /*! Address offsets are relative to this physical version slot, when present.
   */
  Optional<PrimExpr> version_index;
  PrimExpr descriptor_extent;
  bool generate_uint16_coordinates = false;
};

/*! Select a legal scalar base and vector offset encoding for layout
 * inference, Verify and Vectorize without rewriting source indices or
 * evaluating saved loads again. */
PtoIndexedAccessPlan PlanPtoIndexedAccess(
    const MemoryAccess &access, const IndexedMemoryOwner &owner,
    const Var &loop_var, const PtoInverseMapping &mapping,
    arith::Analyzer *analyzer, const Map<Var, int> &buffer_versions = {},
    const Map<Var, PrimExpr> &bind_env = {});

/*! Check whether a dtype is supported by the unified element
 * addressing convention: the buffer element dtype itself must not carry
 * vector lanes and must not be a packed sub-byte type (FP4 etc.).
 *
 * E4M3FN and E5M2 are supported element types; the other FP8 variants
 * (float8_e4m3 with its own TIR dtype, E8M0, FNUZ, HiF8) are not, even
 * where the PTO codegen would map them to the same target type — only the
 * two dtypes listed in IsSupportedFloat8 take part in the FP8 conversion
 * matrix. */
bool IsSupportedElementDType(DataType dtype);
void CheckBroadcastDType(DataType dtype, const std::string &context);

/*! The two FP8 element types the conversion matrix supports: float8_e4m3fn and
 * float8_e5m2. Deliberately narrower than the codegen's FP8 name mapping:
 * admitting a type here means its conversion directions were validated, not
 * just that a target name exists. */
bool IsSupportedFloat8(DataType dtype);

/*! Logical value representation. One-lane VMI values retain register
 * provenance even though their TIR dtype has one lane. */
enum class ValueKind { kUnknown, kScalar, kOneLaneVmi, kVectorVmi };

struct ValueInfo {
  ValueKind kind = ValueKind::kUnknown;
  int64_t logical_lanes = 0;

  static ValueInfo Vmi(DataType dtype);
  static ValueInfo Vmi(int64_t lanes);

  bool IsVmi() const {
    return kind == ValueKind::kOneLaneVmi || kind == ValueKind::kVectorVmi;
  }
};

using ValueInfoMap =
    std::unordered_map<Var, ValueInfo, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;

/*! Classify a value emitted by Parallel conversion. Bound variables carry
 * their producer's representation, including one-lane register results. */
ValueInfo GetValueInfo(const PrimExpr &value, const ValueInfoMap &bindings);

enum class WidthAdaptation { kKeep, kBroadcastScalar, kBroadcastOneLane };

/*! Adapt only scalar or one-lane values to a consumer's logical width.
 * Arbitrary vector resizing requires an explicit future conversion rule. */
WidthAdaptation PlanWidthAdaptation(const ValueInfo &value, int64_t lanes);

struct PtoReducerStates {
  int64_t lanes{0};
  Map<Var, Buffer> accumulators;
  Array<PtoFinalizeReducerOp> finalizes;
  ValueInfoMap storage_values;
};
PtoReducerStates CollectReducerStates(const SBlock &vf,
                                      const ParallelStorageQuery &storage);
int64_t AnalyzePtoFinalize(const PtoFinalizeReducerOp &op,
                           const PtoReducerStates &states);
Optional<Buffer> AnalyzeReducerAccess(const PtoReducerStates &states,
                                      const Buffer &buffer,
                                      const Array<PrimExpr> &indices,
                                      const ParallelStorageQuery &storage,
                                      arith::Analyzer *analyzer,
                                      const Map<Var, PrimExpr> &bind_env = {});

/*! Value category of one value-Cast position.  The kind is supplied by the
 * caller (Verify from AnalyzeLaneUse, Vectorize from the converted input
 * shape); the planner never infers it from lanes. */
using CastValueKind = ValueKind;

/*! Per-Cast planning context.  `logical_lanes` is validation information
 * only; `value_kind` is the dispatch key. */
struct CastContext {
  ValueKind value_kind = ValueKind::kScalar;
  int64_t logical_lanes = 1;
};

/*! Scalar value-Cast plan: either keep the input (identity) or rebuild an
 * ordinary scalar Cast with the validated annotations and the original span.
 * No new scalar rounding/saturation mode is introduced. */
enum class ScalarCastAction {
  kKeepInput,
  kRebuildCast,
};

struct ScalarCastPlan {
  ScalarCastAction action = ScalarCastAction::kKeepInput;
  /*! Validated annotations to carry onto the rebuilt Cast (the same shape as
   * CastNode::annotations).  Attributes that are proven to match the
   * existing scalar lowering are normalized away, so this map is empty for
   * the currently admitted directions. */
  ffi::Map<ffi::String, ffi::Any> annotations;
};

/*! One ordered vector conversion action. */
enum class VectorCastOp {
  /*! Same-sign widening: one logical vcvt. */
  kIntegerExtend,
  /*! Same-width signedness change / source-sign re-view: one
   * vinterpret_cast. */
  kIntegerReinterpret,
  /*! Narrowing including cross-sign: one logical vcvt with NOSAT. */
  kIntegerTruncate,
  /*! Real non-integer conversion (f16/bf16/FP8 -> f32, f32 -> f16/bf16/FP8);
   * empty options still mean a conversion, never a no-op. */
  kNonIntegerConvert,
};

struct VectorCastStep {
  VectorCastOp op = VectorCastOp::kIntegerExtend;
  DataType from; // element type consumed by this step
  DataType to;   // element type produced by this step
  /*! PTODSL rounding annotation value, e.g. "R"; empty when N/A. */
  std::string rounding;
  /*! PTODSL saturate annotation value, e.g. "NOSAT"; empty when N/A. */
  std::string saturate;
};

struct VectorCastPlan {
  /*! Ordered actions; an empty list is a proven identity (no conversion).
   * A non-empty list still may be a single real conversion with empty
   * options (that is not an identity). */
  std::vector<VectorCastStep> steps;
};

using CastPlan = std::variant<ScalarCastPlan, VectorCastPlan>;

// Maps a bool-valued bitwise call to its VMI mask operation.
const char *GetPtoMaskLogicOp(const tirx::Call &call);

// The tirx.* unary math ops with a VMI mapping (fabs/exp/log/sqrt):
// returns the VMI op name, or nullptr for any other name.
const char *VectorUnaryVmiOpName(const std::string &tirx_op_name);

/*! Operation-level type support for the vector arithmetic emitted
 * here. VMI vmul only supports i16/i32/f16/bf16/f32
 * elements; vadd/vsub additionally allow i8; vdiv admits f16/f32 only.
 * Unsupported combinations are rejected instead of synthesizing
 * widening chains. */
const char *GetPtoUnaryOp(const tirx::Call &call);
bool IsSupportedVectorUnaryOp(const std::string &tirx_op, DataType dtype);
bool IsSupportedVectorBinaryOp(const std::string &vmi_op, DataType dtype);
bool ContainsRuntimeValue(const PrimExpr &expr);
bool IsFloatFamily(DataType dtype);

/*! TIR value calls with a direct bit-preserving or bitwise PTO lowering. */
enum class PtoBitOp {
  kNone,
  kAnd,
  kOr,
  kXor,
  kNot,
  kShiftLeft,
  kShiftRight,
  kReinterpret
};

PtoBitOp GetPtoBitOp(const tirx::Call &call);

/*! Check the operand contract of a supported value call. Returns false for
 * other calls, which may still be valid scalar address intrinsics. */
bool CheckPtoValueCall(const tirx::Call &call);

/*! Read-only lane-use classification of a value expression.
 * Shared by the metadata check and the operation type check so both
 * consume the same source semantics. Mirrors the structural conversion
 * the Vectorize pass performs (no new algebraic simplification). */
enum class LaneUse {
  kUniform, ///< stays scalar under the current conversion rules
  kVarying, ///< needs a vector value (incl. the loop index as a value)
  kUnknown, ///< cannot analyze; caller must reject with the reason
};

std::string LaneUseToString(LaneUse use);

/*! Analysis context for AnalyzeLaneUse: the unit's loop var, inverse
 * mapping, the bind-definition environment, a cache of
 * already-classified bind vars, the set of known external scalar vars
 * (params / buffer data pointers / outer-scope definitions), and the
 * recursion guards of the classification walk. */
struct LaneUseContext {
  const ValueInfoMap *storage_values = nullptr;
  const ValueInfoMap *value_infos = nullptr;
  const pto::ParallelStorageQuery *storage = nullptr;
  For unit;
  arith::Analyzer *analyzer = nullptr;
  Var loop_var;
  PtoInverseMapping mapping;
  Map<Var, PrimExpr> bind_env;
  Map<Var, Integer> classified; // bind var -> 0 uniform / 1 varying
  /*! Known external scalar vars: vars outside the unit that are
   * legitimately defined. A var that is neither in bind_env nor here is
   * *undefined* -> Unknown, not Uniform. */
  std::set<const VarNode *> external_defs;
  /*! Bind vars currently being analyzed (cycle guard). */
  std::set<const VarNode *> visiting;
  /*! Recursion budget for on-demand bind analysis. */
  int budget = 256;
};

/*! Classify \p expr bottom-up. Constants and *known*
 * external scalar vars are Uniform; the loop var as a value is Varying;
 * Ordinary private BufferLoads are Varying and external scalar slots are
 * Uniform; UB loads use AnalyzeBufferAccess (uniform -> Uniform,
 * continuous -> Varying); Add/Mul/Cast propagate operands; original Bind
 * vars use their recorded classification (on-demand analysis is guarded
 * by the visiting set and budget; cycles and exhaustion -> Unknown).
 * \p reason is filled on kUnknown. */
ValueInfo AnalyzeComputeValue(const PrimExpr &expr, LaneUseContext *ctx);

LaneUse AnalyzeLaneUse(const PrimExpr &expr, LaneUseContext *ctx,
                       std::string *reason);

/*! Metadata-mode lane classification: same Var/Bind/BufferLoad
 * rules as AnalyzeLaneUse, but the *node table* is the metadata table —
 * pure scalar Sub/Div/Mod/FloorDiv/FloorMod/Min/Max/comparison/boolean
 * operators and StringImm constants propagate their operands' lane use
 * instead of being rejected by the value whitelist. Unknown nodes still
 * produce kUnknown. */
LaneUse AnalyzeLaneUseMeta(const PrimExpr &expr, LaneUseContext *ctx,
                           std::string *reason);

/*! Decide the conversion plan for a value Cast. This is the single admission
 * point for every Cast on the vectorize path; Verify and Vectorize both pass
 * an explicit CastContext and consume the same plan, so they cannot diverge.
 *
 * Dispatch is first scalar vs VMI, then integer vs non-integer:
 *  - scalar: identity KeepInput; integer widening/narrowing and the newly
 *    admitted same-width signedness change rebuild an ordinary scalar Cast;
 *    f16/bf16 <-> f32 keep their existing scalar lowering; non-identity FP8,
 *    float<->int and f16<->bf16 stay rejected;
 *  - VMI integer: identity 0 steps; same-width cross-sign one
 *    vinterpret_cast; same-sign widening one vcvt; cross-sign widening
 *    source-sign widen then same-width vinterpret_cast; narrowing one direct
 *    vcvt NOSAT. All 36 supported pairs are represented; no consumer-layout
 *    feasibility is guessed here;
 *  - VMI non-integer: a real conversion action for f16/bf16/FP8 -> f32
 *    (empty options are not a no-op) and for f32 -> f16/bf16/FP8 with the
 *    existing rounding/saturation policy.
 *
 * `cast` carries the backend annotations written by the frontend cast
 * helpers (rounding mode, saturation flag, stochastic rbits operand). They
 * are read here — never interpreted by the expression visitors — and an
 * attribute the selected conversion cannot honour is a rejection, not a
 * silently dropped hint. When `reason` is non-null it receives a diagnostic
 * naming the category, source/target types, lanes and the offending option. */
Optional<CastPlan> PlanValueCast(const CastNode *cast,
                                 const CastContext &context,
                                 std::string *reason = nullptr);

/*! Check whether an expression only uses nodes from the supported
 * whitelist: BufferLoad, constants, scalar Vars, loop
 * indices, Add, Mul, limited Cast, Bind values. Returns nullopt when
 * supported; otherwise a short description of the offending node. */
std::optional<std::string> FindUnsupportedExprNode(const PrimExpr &expr);

/*! Logical direct-reduction plan, independent of physical register width. */
struct PtoDirectReducePlan {
  BufferRegion source;
  BufferRegion destination;
  int source_prefix = 0;
  int destination_prefix = 0;
  int logical_rank = 0;
  int reduction_axis = 0;
  int output_axis = 0;
  int destination_output_axis = 0;
  int64_t reduction_extent = 0;
  int64_t output_extent = 0;
  bool reduce_across_lanes = false;
  bool clear = true;
  std::string operation;
  DataType accumulator_dtype;
};

/*! Whether the call denotes a direct tile reduction. */
bool IsPtoDirectReduce(const Call &call);

/*! Check direct-reduce admission without building a lowering plan.
 * Throws PtoAnalysisError for unsupported calls; does not change IR. VF lanes
 * and lexical structure are checked by VerifyParallelToPTO itself. */
void CheckPtoDirectReduce(const Call &call, arith::Analyzer *analyzer,
                          const Map<Var, int> &buffer_versions = {});

/*! Build a direct-reduce lowering plan for already verified input.
 * Requires CheckPtoDirectReduce and the VF checks to have succeeded. */
PtoDirectReducePlan
PlanPtoDirectReduce(const Call &call, arith::Analyzer *analyzer,
                    const Map<Var, int> &buffer_versions = {});

/*! Ordinary UB fill plan. Version coordinates remain in destination. */
struct PtoFillPlan {
  BufferRegion destination;
  PrimExpr value;
  int continuous_axis = 0;
  int outer_axis = -1;
  int64_t run_extent = 0;
  int64_t outer_extent = 1;
};

bool IsPtoFill(const Call &call);

/*! Check fill/clear admission without building a lowering plan.
 * Throws PtoAnalysisError for unsupported calls; does not change IR. */
void CheckPtoFill(const Call &call, int64_t lanes, arith::Analyzer *analyzer,
                  const Map<Var, int> &buffer_versions = {});

/*! Build a fill/clear lowering plan for already verified input.
 * Requires CheckPtoFill and the VF checks to have succeeded. */
PtoFillPlan PlanPtoFill(const Call &call, arith::Analyzer *analyzer,
                        const Map<Var, int> &buffer_versions = {});

/*! Result of scanning a SIMD_VF region for parallel units. Shared by
 * Verify and Vectorize so both passes agree on skip/reject/convert
 * decisions. */
struct PtoRegionScan {
  /*! Sequential 1D parallel units found (outermost For of each chain). */
  std::vector<For> units;
  /*! True when a Parallel exists under a wrapper we cannot currently
   * convert (IfThenElse / While / nested 2D+ parallel). Verify
   * rejects with \p issue; Vectorize treats it as an internal error. */
  bool has_unsupported_wrapper = false;
  std::string issue;
  /*! True when the region contains any T.Parallel (directly or under wrappers).
   * A region without Parallel, direct reduce or fill/clear remains on the
   * existing hand-written VMI path. */
  bool has_any_parallel = false;
  /*! True when a direct reduce occurs anywhere in the region. */
  bool has_direct_reduce = false;
  /*! True when a region-level fill/clear occurs in the region. */
  bool has_fill = false;
};

/*! Scan a SIMD_VF region body (the SBlock's body, including any scope
 * AttrStmt wrapper) for parallel units. One implementation shared by
 * Verify and Vectorize. */
PtoRegionScan ScanPtoRegion(const Stmt &region);

/*! A statement that carries no work: an Evaluate of an IntImm constant
 * (e.g. a store simplified away into a constant), an empty SeqStmt, or a
 * SeqStmt whose every element is itself a no-op. The shared contract for
 * the "no operation" arm of a conditional branch under T.Parallel — the
 * shape Legalize rewrites as "no else" (the masked-update Select form) and
 * Verify admits as Select-rewritable. */
bool IsNoOpStmt(const Stmt &stmt);

/*! Whether \p value may be substituted for its `Bind` var in a predicate proof.
 * A `Bind` is an SSA snapshot, so only expressions that cannot observe state
 * (constants, vars, arithmetic, comparisons, logic, Select, Min/Max, Cast) are
 * substitutable; a buffer load or a call keeps the var opaque. */
bool IsSafelyReevaluableValue(const PrimExpr &value);

/*! Whether \p predicate is provably true under \p analyzer. The scalar `Bind`
 * values in \p bind_env are resolved first, and the enclosing loop ranges must
 * already be bound into the analyzer. Only a proven-constant-true result
 * counts: anything undecided is not provably true. Dropping a predicate is
 * sound only when it holds for every lane and every iteration. */
bool IsProvablyTruePredicate(const PrimExpr &predicate,
                             arith::Analyzer *analyzer,
                             const Map<Var, PrimExpr> &bind_env);

/*! Collects the target Vars of the handle-valued `Bind` statements of a
 * statement tree (`q = A.data`, `r = q`, a reinterpret cast, ...). A buffer
 * declared over one of these pointers is a view whose storage the lane analysis
 * cannot follow, so Verify rejects every access to such a view.
 *
 * The pointer origin is deliberately not traced: a handle `BufferLoad` yields a
 * pointer value, not the storage that holds it. */
void CollectHandleBindTargets(const Stmt &stmt, std::set<const VarNode *> *out);

/*! What a function knows about where its buffers live. */
struct StorageOrigins {
  /*! Data Vars of real allocations (`AllocBuffer` / block `alloc_buffers`);
   * distinct allocations never overlap. */
  std::set<const VarNode *> allocations;
  /*! Data Vars of parameter buffers. */
  std::set<const VarNode *> parameters;
  /*! Parameter data Vars the caller declared as non-restrict; the source may
   * pass any of them the same storage. */
  std::set<const VarNode *> non_restrict;
  /*! True only when the function explicitly asserts `tir.noalias`. Absent is
   * not a promise: this pass runs before the kernel ABI is materialised, so two
   * parameters may still name one storage. */
  bool no_alias = false;
};

/*! Collects the storage origins of \p body plus the parameter ABI facts:
 * buffer_map entries, the no-alias attribute, and `tl.non_restrict_params`
 * (function attribute or block annotation). */
StorageOrigins CollectStorageOrigins(const Stmt &body,
                                     const Map<Var, Buffer> &buffer_map,
                                     bool no_alias,
                                     const Array<Var> &non_restrict_params);

/*! Whether two accesses may touch the same storage.
 *
 * Same data Var conflicts; two distinct allocations never do; two distinct
 * parameters are independent only under an explicit no-alias contract that does
 * not mark either as non-restrict. Anything else stays conservatively
 * conflicting. Verify and Legalize share this judgment. */
bool MayConflictStorage(const BufferNode *lhs, const BufferNode *rhs,
                        const StorageOrigins &origins);

/*!
 * \brief One relational node kind: how to take its operands, its PTODSL compare
 * name, and how to rebuild it when both operands stay scalar.
 *
 * The (node kind, compare name, scalar semantics) triple lives here once so a
 * new relational kind cannot drift between the verifier's whitelist, the
 * vectorizer's dispatch and its lane-uniform fold.
 */
struct RelationalKind {
  /*! \brief Fill a/b (when non-null) with the node's operands, in the node's
   *  own (a, b) order, and report whether \p e is of this kind. */
  bool (*take)(const PrimExpr &e, PrimExpr *a, PrimExpr *b);
  /*! \brief PTODSL compare name: "eq" / "ne" / "lt" / "le" / "gt" / "ge". */
  const char *cmp;
  /*! \brief Rebuild the comparison for two lane-uniform operands. */
  PrimExpr (*fold)(const PrimExpr &a, const PrimExpr &b);
};

/*!
 * \brief Match \p e against the relational table: on a hit fill a/b (either may
 * be null) with the node's operands and return the entry, else return nullptr.
 */
const RelationalKind *MatchRelational(const PrimExpr &e, PrimExpr *a,
                                      PrimExpr *b);

/*!
 * \brief Look the table up by PTODSL compare name; nullptr when unknown.
 */
const RelationalKind *RelationalByName(const char *cmp);

/*!
 * \brief Collect a flat list of stores from \p stmt (a store, a no-op, or a
 * Seq of those). Returns false when the statement is any other shape.
 */
bool CollectStores(const Stmt &stmt, std::vector<const BufferStoreNode *> *out);
} // namespace pto
} // namespace tl
} // namespace tvm

#endif // TVM_TL_ASCEND_TRANSFORM_PARALLEL_TO_PTO_UTILS_H_
