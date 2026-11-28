/*!
 * \file pto_parallel_layout.cc
 * \brief PTO lane-layout construction for T.Parallel units.
 */

#include "pto_parallel_layout.h"

#include <tvm/arith/analyzer.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

#include <limits>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "../transform/parallel_to_pto_utils.h"
#include "pto_index_analysis.h"

namespace tvm {
namespace tl {
namespace pto {

using namespace tirx;
using ffi::Array;
using ffi::Map;

namespace {

// Collect the T.Parallel loop variables in the same traversal order used by
// the generic parallel operator: outermost first, descending through the whole
// body (including serial loops between parallel layers).
class PtoParallelLoopCollector : public tirx::StmtExprVisitor {
public:
  Array<IterVar> loop_vars;

  void VisitStmt_(const ForNode *op) final {
    if (op->kind == ForKind::kParallel) {
      loop_vars.push_back(IterVar(Range(op->min, op->extent), op->loop_var,
                                  IterVarType::kDataPar));
    }
    tirx::StmtExprVisitor::VisitStmt_(op);
  }
};

// Pick the vectorized dimension of a two-layer Parallel nest from the buffer
// accesses. Both dimensions use the same address classification as the
// consumers. Minimize total iterations (kept extent * ceil(selected extent /
// lanes)), then per-iteration gather + scatter count, then scatter count; ties
// retain the non-degenerate-axis rule and inner-axis preference. Count source
// accesses without predicting CSE or physical instruction costs. The proof
// domain includes the padded tail lanes.
class PtoAccessCollector : public tirx::StmtExprVisitor {
public:
  PtoAccessCollector(const pto::ParallelStorageQuery &storage, const For &root)
      : storage_(storage), root_(root) {}
  bool IsMemory(const Buffer &b, const Array<PrimExpr> &indices) {
    auto kind = storage_.Classify(b, root_);
    if (kind == pto::StorageClass::kPrivate ||
        kind == pto::StorageClass::kExternal) {
      arith::Analyzer analyzer;
      storage_.ResolveSlot(b, indices, &analyzer, bind_env_);
      return false;
    }
    return true;
  }
  struct Access {
    Buffer buffer;
    Array<PrimExpr> indices;
    bool is_write;
    /*! The definition environment at this access's program point
     * (binds in scope, mirrors the consumers' lexical rules). */
    Map<Var, PrimExpr> bind_env;
  };
  struct ValueOperation {
    PrimExpr expr;
    std::string op_name;
    Map<Var, PrimExpr> bind_env;
    Optional<Var> binding;
  };
  std::vector<Access> accesses;
  std::vector<ValueOperation> value_operations;

  void VisitStmt_(const BufferStoreNode *op) final {
    if (IsMemory(op->buffer, op->indices)) {
      accesses.push_back(
          {op->buffer, op->indices, /*is_write=*/true, bind_env_});
      VisitAddressMetadata(op->buffer, op->indices);
    }
    CollectValueOperations_(op->value);
    tirx::StmtExprVisitor::VisitStmt_(op);
  }
  void VisitExpr_(const BufferLoadNode *op) final {
    if (IsMemory(op->buffer, op->indices)) {
      accesses.push_back(
          {op->buffer, op->indices, /*is_write=*/false, bind_env_});
      VisitAddressMetadata(op->buffer, op->indices);
    }
    tirx::StmtExprVisitor::VisitExpr_(op);
  }
  void VisitStmt_(const BindNode *op) final {
    // Flat sequence semantics (same as the consumers): the value is
    // visited *before* the var is registered, so a self reference is not
    // in scope; every bind — including load-valued ones — is registered
    // so ResolveIndexAliases can apply its loaded-value identity rule
    // along alias chains.
    CollectValueOperations_(op->value);
    tirx::StmtExprVisitor::VisitStmt_(op);
    value_operations.push_back({op->value, {}, bind_env_, op->var});
    bind_env_.Set(op->var, op->value);
  }
  void VisitStmt_(const EvaluateNode *op) final {
    CollectValueOperations_(op->value);
    tirx::StmtExprVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const AttrStmtNode *op) final {
    // Nested statement scope (mirror of the consumers' rule):
    // binds defined inside an AttrStmt body must not leak to later
    // siblings, while outer binds stay visible inside.
    Map<Var, PrimExpr> saved = bind_env_;
    tirx::StmtExprVisitor::VisitStmt_(op);
    bind_env_ = std::move(saved);
  }

private:
  void CollectValueOperations_(const PrimExpr &value) {
    PreOrderVisit(value, [&](const ObjectRef &node) {
      // Index operations are checked by the address and offset analysis.
      if (node.as<BufferLoadNode>()) {
        return false;
      }
      std::string name;
      if (node.as<CastNode>()) {
        name = "Cast";
      } else if (node.as<AddNode>()) {
        name = "vadd";
      } else if (node.as<SubNode>()) {
        name = "vsub";
      } else if (node.as<MulNode>()) {
        name = "vmul";
      } else if (node.as<DivNode>()) {
        name = "vdiv";
      } else if (node.as<FloorDivNode>()) {
        name = "vfloordiv";
      } else if (node.as<FloorModNode>()) {
        name = "vfloormod";
      } else if (const auto *call = node.as<CallNode>()) {
        if (auto op = call->op.as<Op>();
            op && IsSupportedVectorUnaryOp(op.value()->name, call->dtype)) {
          name = op.value()->name;
        }
      }
      if (!name.empty()) {
        value_operations.push_back({Downcast<PrimExpr>(node), std::move(name),
                                    bind_env_, std::nullopt});
      }
      return true;
    });
  }

  void VisitAddressMetadata(const Buffer &buffer,
                            const Array<PrimExpr> &indices) {
    // Buffer visitors see the indices but not metadata at a use site.
    // A view's address can contain another load with its own offset plan.
    for (const PrimExpr &expr : PtoAddressMetadata(buffer, indices)) {
      VisitExpr(expr);
    }
  }

  Map<Var, PrimExpr> bind_env_;
  const pto::ParallelStorageQuery &storage_;
  For root_;
};

std::string CheckValueOperation(const PtoAccessCollector::ValueOperation &op,
                                LaneUseContext *ctx, ValueInfoMap *bindings) {
  ctx->bind_env = op.bind_env;
  const auto *cast_op = op.expr.as<CastNode>();
  const PrimExpr &input = cast_op ? cast_op->value : op.expr;
  const LaneUse use = AnalyzeLaneUse(input, ctx, nullptr);
  if (use == LaneUse::kUnknown) {
    // Verify diagnoses unsupported expressions and undefined references.
    return {};
  }
  const auto info = AnalyzeComputeValue(input, ctx);
  if (op.binding) {
    (*bindings)[op.binding.value()] = info;
    return {};
  }
  if (cast_op) {
    CastContext cast_context{info.kind, info.logical_lanes};
    std::string why;
    if (!PlanValueCast(cast_op, cast_context, &why)) {
      return "Cast has no supported conversion: " + why;
    }
    return {};
  }

  const DataType dtype = op.expr.dtype();
  if (!info.IsVmi() && IsFloatFamily(dtype) && ContainsRuntimeValue(op.expr)) {
    return "runtime lane-uniform float computation requires a scalar float "
           "ALU inside SIMD_VF";
  }
  if (info.IsVmi()) {
    const bool supported = op.expr.as<CallNode>()
                               ? IsSupportedVectorUnaryOp(op.op_name, dtype)
                               : IsSupportedVectorBinaryOp(op.op_name, dtype);
    if (!supported) {
      std::ostringstream oss;
      oss << op.op_name << " on " << dtype
          << " elements is not supported by the VMI operation matrix";
      return oss.str();
    }
  }
  return {};
}

void CheckContinuousStore(const MemoryAccess &access, const Var &loop_var,
                          const PtoInverseMapping &mapping,
                          const Map<Var, PrimExpr> &bind_env,
                          const std::vector<Buffer> &allocations,
                          const Map<Var, int> &buffer_versions,
                          arith::Analyzer *analyzer) {
  const DataType dtype = access.buffer->dtype;
  const int bits = dtype.bits();
  if (!dtype.is_scalar() || dtype.is_bool() ||
      (bits != 8 && bits != 16 && bits != 32)) {
    return;
  }
  // Resolve aliases from the whole address, including view metadata, before
  // substituting the selected coordinate at lane zero.
  auto offset = ResolveIndexAliases(access.original_offset, bind_env);
  if (!offset) {
    throw PtoAnalysisError("continuous store start is not provably "
                           "32-byte aligned");
  }
  PrimExpr lane_zero = SubstituteVar(mapping.index_expr, mapping.lane_var.get(),
                                     make_zero(mapping.lane_var.dtype()));
  PrimExpr start = SubstituteVar(*offset, loop_var.get(), lane_zero);
  const DataType wide = DataType::Int(64);
  PrimExpr bit_offset = cast(wide, start) * bits;

  // UB allocation bases are aligned. Version padding can change the slot
  // start, but it preserves the offset within that slot.
  constexpr int kAlignmentBits = 256;
  auto versions = buffer_versions.Get(access.buffer->data);
  if (versions && *versions > 1) {
    for (const Buffer &owner : allocations) {
      if (!owner->data.same_as(access.buffer->data) || owner->shape.empty() ||
          !analyzer->CanProveEqual(owner->shape[0], *versions)) {
        continue;
      }
      PrimExpr slot_bits =
          make_const(wide, owner->dtype.bits() * owner->dtype.lanes());
      for (size_t axis = 1; axis < owner->shape.size(); ++axis) {
        slot_bits = slot_bits * cast(wide, owner->shape[axis]);
      }
      if (!analyzer->CanProveEqual(floormod(slot_bits, kAlignmentBits), 0)) {
        PrimExpr origin = cast(wide, owner->elem_offset) *
                          (owner->dtype.bits() * owner->dtype.lanes());
        PrimExpr within_slot = indexmod(bit_offset - origin, slot_bits);
        // The access descriptor covers the full vector, including masked
        // tail lanes. The version layout rewrite cannot remap a span that
        // crosses the padding between slots.
        PrimExpr span_bits = make_const(wide, mapping.lanes * bits);
        if (!analyzer->CanProve(within_slot <= slot_bits - span_bits)) {
          throw PtoAnalysisError("continuous store descriptor does not "
                                 "provably fit within one padded version slot");
        }
        bit_offset = origin + within_slot;
      }
      break;
    }
  }
  if (!analyzer->CanProveEqual(floormod(bit_offset, kAlignmentBits), 0)) {
    throw PtoAnalysisError("continuous store start is not provably "
                           "32-byte aligned");
  }
}

// 2D layout candidate. Both dimensions are tried with their canonical mapping
// — selected coordinate `q * L + lane`, the other coordinate left as-is — over
// the bound lane/chunk/kept ranges. Verify and Vectorize consume the same
// address analysis and indexed offset plan as layout inference.
// Continuous accesses and uniform loads do not need indexed instructions.
// Uniform stores use scatter and count as indexed stores.
// Candidates must support offset generation before entering cost comparison.
// Verify remains responsible for the complete program contract.
// Layout-inference code reports construct-time failures as
// LoopLayoutInjectiveException (the type LayoutInference catches to discard
// an attempt), while the shared padding helper speaks PtoAnalysisError;
// translate here so all three call sites share one helper.
std::pair<int64_t, int64_t> CheckedPaddingForInference(int64_t extent,
                                                       int64_t lanes) {
  try {
    return CheckedPtoPadding(extent, lanes);
  } catch (const PtoAnalysisError &err) {
    throw LoopLayoutInjectiveException(err.what());
  }
}

bool ShouldVectorizeInnerAxis(const For &root, const Var &iv, const Var &jv,
                              int64_t M, int64_t N, int64_t lanes,
                              const std::vector<Buffer> &active_allocations,
                              const Map<Var, int> &buffer_versions,
                              arith::Analyzer *analyzer,
                              const pto::ParallelStorageQuery &storage,
                              const ReducerLayoutValues &values) {
  PtoAccessCollector collector(storage, root);
  collector(LegalizePrivateIndexValues(root));

  Array<Var> external_vars;
  if (!collector.value_operations.empty()) {
    external_vars = UndefinedVars(root, {});
  }

  PtoInverseMapping common_mapping;
  SetPtoIndexStorage(&common_mapping, storage.PrivateAllocations(root),
                     values.results);
  common_mapping.lanes = lanes;
  common_mapping.is_2d = true;
  common_mapping.outer_var = iv;
  common_mapping.inner_var = jv;
  common_mapping.outer_extent = M;
  common_mapping.inner_extent = N;

  struct Candidate {
    int64_t chunk_count = 0;
    int64_t indexed_loads = 0;
    int64_t indexed_stores = 0;
    std::string rejection_reason;
  };
  auto analyze_candidate = [&](bool select_inner) {
    Candidate candidate;
    const int64_t selected_extent = select_inner ? N : M;
    const Var &selected_var = select_inner ? jv : iv;
    const int64_t kept_extent = select_inner ? M : N;
    const int64_t padded =
        CheckedPaddingForInference(selected_extent, lanes).second;
    candidate.chunk_count = padded / lanes;
    if (kept_extent > std::numeric_limits<int32_t>::max() / padded) {
      candidate.rejection_reason =
          "padded flattened extent overflows the int32 index space";
      return candidate;
    }
    const DataType dtype = selected_var.dtype();
    Var lane("probe_lane", dtype);
    Var q("probe_q", dtype);
    PrimExpr inverse = q * IntImm(dtype, lanes) + lane;
    PtoInverseMapping mapping = common_mapping;
    mapping.lane_var = lane;
    mapping.chunk_var = q;
    mapping.logical_extent = selected_extent;
    mapping.padded_extent = padded;
    mapping.chunk_count = candidate.chunk_count;
    mapping.tail_lanes = selected_extent % lanes;
    mapping.index_expr = inverse;
    mapping.select_inner = select_inner;
    auto store_proof = analyzer->Clone();
    const Var &kept_var = select_inner ? iv : jv;
    store_proof->Bind(kept_var, Range(make_zero(kept_var.dtype()),
                                      IntImm(kept_var.dtype(), kept_extent)));
    store_proof->Bind(
        q, Range(make_zero(dtype), IntImm(dtype, mapping.chunk_count)));
    for (const auto &access : collector.accesses) {
      try {
        MemoryAccess analysis =
            AnalyzeBufferAccess(access.buffer, access.indices, access.is_write,
                                selected_var, mapping, access.bind_env);
        if (analysis.RequiresIndexedInstruction()) {
          // A view may be legal for scalar/continuous access but not for an
          // indexed instruction, e.g. when it reinterprets the owner's dtype.
          Buffer allocation = FindAllocationOwner(
              active_allocations, access.buffer, "PTO layout inference");
          IndexedMemoryOwner owner = CheckIndexedMemoryOwner(
              access.buffer, allocation, analyzer, "PTO layout inference");
          PlanPtoIndexedAccess(analysis, owner, selected_var, mapping, analyzer,
                               buffer_versions, access.bind_env);
          if (access.is_write) {
            ++candidate.indexed_stores;
          } else {
            ++candidate.indexed_loads;
          }
        } else if (access.is_write &&
                   IsOrdinaryUbScope(access.buffer.scope())) {
          CheckContinuousStore(analysis, selected_var, mapping, access.bind_env,
                               active_allocations, buffer_versions,
                               store_proof.get());
        }
      } catch (const PtoAnalysisError &err) {
        std::ostringstream oss;
        oss << "buffer `" << access.buffer->name << "` ("
            << (access.is_write ? "store" : "load") << "): " << err.what();
        candidate.rejection_reason = oss.str();
        return candidate;
      }
    }
    auto bindings = values.aliases;
    LaneUseContext lane_use;
    lane_use.value_infos = &bindings;
    lane_use.storage_values = &values.results;
    lane_use.loop_var = selected_var;
    lane_use.mapping = mapping;
    lane_use.storage = &storage;
    lane_use.unit = root;
    lane_use.analyzer = store_proof.get();
    for (const Var &var : external_vars) {
      lane_use.external_defs.insert(var.get());
    }
    const int lane_use_budget = lane_use.budget;
    for (const auto &operation : collector.value_operations) {
      lane_use.budget = lane_use_budget;
      candidate.rejection_reason =
          CheckValueOperation(operation, &lane_use, &bindings);
      if (!candidate.rejection_reason.empty()) {
        break;
      }
    }
    return candidate;
  };

  Candidate inner = analyze_candidate(true);
  Candidate outer = analyze_candidate(false);
  if (inner.rejection_reason.empty() && outer.rejection_reason.empty()) {
    const auto inner_cost = std::make_tuple(
        M * inner.chunk_count, inner.indexed_loads + inner.indexed_stores,
        inner.indexed_stores);
    const auto outer_cost = std::make_tuple(
        N * outer.chunk_count, outer.indexed_loads + outer.indexed_stores,
        outer.indexed_stores);
    if (inner_cost != outer_cost) {
      return inner_cost < outer_cost;
    }
    return N != 1 || M == 1;
  }
  if (inner.rejection_reason.empty()) {
    return true;
  }
  if (outer.rejection_reason.empty()) {
    return false;
  }
  std::ostringstream oss;
  oss << "[PTO layout inference] the 2D unit has no vectorizable dimension (M="
      << M << ", N=" << N << ", L=" << lanes << "): inner (j) -> "
      << inner.rejection_reason << "; outer (i) -> " << outer.rejection_reason;
  throw LoopLayoutInjectiveException(oss.str());
}

} // namespace

// PTO SIMD lane layout: logical points of the parallel loop are distributed
// over [0, L) lanes. The flattened logical index is
//   1D:      t = i
//   2D:      t = i * N + j (j vectorized) or t = j * M + i (i vectorized)
// with lane = t % L and per-lane chunk position p = t // L; forward_index
// holds p. vectorize_size is fixed at 1 — the per-thread SIMT vector-width
// selection does not apply.
//
// Scope note: this function only *builds* the lane fragment. The full input
// contract (access patterns, dtype, expression whitelist) belongs to
// VerifyParallelToPTO; the checks below are only the minimal conditions
// needed to construct a valid layout, reported as layout-inference errors.
Fragment InferPtoParallelLayout(const For &root, const Range &lane_bounds,
                                const std::vector<Buffer> &active_allocations,
                                const Map<Var, int> &buffer_versions,
                                arith::Analyzer *analyzer,
                                const ParallelStorageQuery &storage,
                                const ReducerLayoutValues &values) {
  PtoParallelLoopCollector loop_collector;
  loop_collector(root);
  const Array<IterVar> &loop_vars = loop_collector.loop_vars;

  const int64_t *lanes_ptr = as_const_int(lane_bounds->extent);
  ICHECK(lanes_ptr != nullptr && *lanes_ptr > 0)
      << "[PTO layout inference] lane count must be a positive compile-time "
         "constant, got "
      << lane_bounds->extent;
  int64_t lanes = *lanes_ptr;

  // Loop extents must be compile-time constants. Keep the
  // int64 value: narrowing here would silently truncate large static
  // extents.
  std::vector<int64_t> extents;
  for (const auto &iv : loop_vars) {
    const int64_t *ext = as_const_int(iv->dom->extent);
    ICHECK(ext != nullptr && *ext > 0)
        << "[PTO layout inference] requires positive compile-time constant "
           "loop "
           "extents, got "
        << iv->dom->extent << " for loop " << iv->var->name_hint;
    ICHECK(is_zero(iv->dom->min))
        << "[PTO layout inference] requires loop min = 0, got " << iv->dom->min
        << " for loop " << iv->var->name_hint;
    extents.push_back(*ext);
  }
  ICHECK(loop_vars.size() == 1u || loop_vars.size() == 2u)
      << "[PTO layout inference] supports a 1D T.Parallel or a two-layer "
         "Parallel nest; got "
      << loop_vars.size() << " layers";

  // 1D mapping: t = i, lane = i % L,
  // forward_index = [i // L]. A non-divisible extent E is padded to
  // P = ceil(E/L)*L in the *fragment input space only* — the IterVar extent
  // below is the padded one, while buffer addresses keep using the real
  // extents/strides (the tail's readable range is a caller precondition).
  DataType idx_dtype = loop_vars[0]->var.dtype();
  PrimExpr flattened;
  // The forward vars handed to the Fragment constructor: the selected
  // dimension's IterVar is padded up to P so the thread extent L times the
  // chunk extent P/L covers the padded input space bijectively; the kept
  // dimension keeps its original extent.
  Array<IterVar> frag_vars;
  if (loop_vars.size() == 1u) {
    const int64_t E = extents[0];
    const auto [pad, P] = CheckedPaddingForInference(E, lanes);
    frag_vars = loop_vars;
    if (pad != 0) {
      // Reuse the loop var; only the input-space extent is padded.
      frag_vars.Set(0, IterVar(Range::FromMinExtent(make_zero(idx_dtype),
                                                    IntImm(idx_dtype, P)),
                               loop_vars[0]->var, IterVarType::kDataPar));
    }
    flattened = InputPlaceholder(0);
  } else {
    // 2D mapping: the vectorized dimension is
    // chosen from the buffer accesses (indices + strides), then the
    // logical space is flattened as t = i * P_N + j (j vectorized) or
    // t = j * P_M + i (i vectorized), where P_* is the *padded* selected
    // extent — the flattened global position p lives in [0, K*Q*L) over the
    // padded input space. Both extents stay compile-time constants.
    const int64_t M = extents[0];
    const int64_t N = extents[1];
    const Var &i_var = loop_vars[0]->var;
    const Var &j_var = loop_vars[1]->var;
    const bool select_inner = ShouldVectorizeInnerAxis(
        root, i_var, j_var, M, N, lanes, active_allocations, buffer_versions,
        analyzer, storage, values);
    const int64_t sel_extent = select_inner ? N : M;
    // Shared checked padding for the selected dimension.
    const auto [pad, P_sel] = CheckedPaddingForInference(sel_extent, lanes);
    frag_vars = loop_vars;
    if (pad != 0) {
      frag_vars.Set(select_inner ? 1 : 0,
                    IterVar(Range::FromMinExtent(make_zero(idx_dtype),
                                                 IntImm(idx_dtype, P_sel)),
                            select_inner ? j_var : i_var,
                            IterVarType::kDataPar));
    }
    flattened = select_inner ? InputPlaceholder(0) * IntImm(idx_dtype, P_sel) +
                                   InputPlaceholder(1)
                             : InputPlaceholder(1) * IntImm(idx_dtype, P_sel) +
                                   InputPlaceholder(0);
  }
  PrimExpr lane = FloorMod(flattened, IntImm(idx_dtype, lanes));
  PrimExpr chunk = FloorDiv(flattened, IntImm(idx_dtype, lanes));
  Var rep_var;
  IterVar rep_iter =
      IterVar(Range::FromMinExtent(make_zero(idx_dtype), IntImm(idx_dtype, 1)),
              rep_var, IterVarType::kDataPar);
  auto fragment = Fragment(frag_vars, {chunk}, lane, rep_iter)
                      ->BindThreadRange(lane_bounds);

  // The fragment must not only be forward-injective: Vectorize consumes it
  // through Fragment::InverseWithLevel, so a layout that cannot be inverted
  // with the expected level has to fail here, closest to its construction,
  // not deep inside the vectorize pass.
  try {
    auto inverse_info = fragment->InverseWithLevel(
        /*require_padding_guard=*/false);
    ICHECK(inverse_info.second == arith::IterMapLevel::Bijective)
        << "[PTO layout inference] PTO lane layout must invert at "
           "Bijective level, got "
        << static_cast<int>(inverse_info.second) << " for "
        << fragment->DebugOutput();
  } catch (const ffi::Error &err) {
    std::ostringstream oss;
    oss << "[PTO layout inference] lane layout cannot be inverted by "
           "InverseWithLevel: "
        << fragment->DebugOutput() << "\n  error: " << err.what()
        << "\n  loop AST: " << root;
    throw LoopLayoutInjectiveException(oss.str());
  }

  return fragment;
}

} // namespace pto
} // namespace tl
} // namespace tvm
