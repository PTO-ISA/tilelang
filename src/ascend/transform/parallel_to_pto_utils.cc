/*!
 * \file parallel_to_pto_utils.cc
 * \brief Shared read-only analysis for VerifyParallelToPTO and
 * VectorizeParallelToPTO.
 */

#include "parallel_to_pto_utils.h"

#include <tvm/arith/pattern.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

#include "../../layout/utils.h"
#include "../../op/utils.h"
#include "span_utils.h"

#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string_view>
#include <utility>

namespace tvm {
namespace tl {
namespace pto {

using namespace tirx;
using ffi::Array;
using ffi::GetRef;
using ffi::Map;

int64_t AnalyzePtoFinalize(const PtoFinalizeReducerOp &op,
                           const PtoReducerStates &states) {
  auto acc = op->accumulator->buffer;
  ICHECK(states.accumulators.count(acc->data))
      << "PTO finalize accumulator missing from tl.pto.reducer_states";
  ICHECK(states.lanes == 64 || states.lanes == 128 || states.lanes == 256)
      << "PTO reducer requires VL 64/128/256";
  ICHECK(op->plan_steps.size() == 2 &&
         op->plan_steps[0]->value == states.lanes &&
         op->plan_steps[1]->value == 1)
      << "PTO reducer supports only plan_steps=[VL,1]";
  ICHECK(IsSupportedPtoReducerDType(op->op, acc->dtype))
      << "PTO reducer requires float16/float32 or int32/uint32 with min/max";
  auto check = [&](const BufferRegion &region) {
    auto b = region->buffer;
    ICHECK(b.scope() == "local")
        << "PTO reducer requires ordinary local storage";
    ICHECK(b->shape.size() == 1 && is_one(b->shape[0]) &&
           region->region.size() == 1 && is_zero(region->region[0]->min) &&
           is_one(region->region[0]->extent))
        << "PTO reducer requires a single local slot region";
    ICHECK_EQ(b->dtype, acc->dtype) << "PTO reducer dtype mismatch";
  };
  check(op->accumulator);
  check(op->result);
  ICHECK(!acc->data.same_as(op->result->buffer->data))
      << "PTO reducer result must be independent";
  if (op->seed.defined()) {
    check(op->seed.value());
    auto seed = op->seed.value()->buffer->data;
    ICHECK(!seed.same_as(acc->data) && !seed.same_as(op->result->buffer->data))
        << "PTO reducer seed must be independent";
  }
  return 1;
}

PtoReducerStates CollectReducerStates(const SBlock &vf,
                                      const ParallelStorageQuery &storage) {
  PtoReducerStates states;
  if (auto lanes = vf->annotations.Get("tl.simdvf_lanes"))
    states.lanes = Downcast<IntImm>(lanes.value())->value;
  if (auto metadata = vf->annotations.Get(attr::kPtoReducerStates)) {
    auto vars = metadata.value().as<Array<ffi::Any>>();
    ICHECK(vars.has_value()) << "tl.pto.reducer_states must contain data Vars";
    for (const auto &entry : vars.value()) {
      auto var = entry.try_cast<Var>();
      ICHECK(var.has_value()) << "tl.pto.reducer_states must contain data Vars";
      Var data = var.value();
      const auto &b = storage.Allocation(data);
      bool owned = false;
      for (const auto &a : vf->alloc_buffers)
        owned |= a->data.same_as(data);
      ICHECK(owned) << "PTO reducer metadata allocation must belong to its VF";
      ICHECK(!states.accumulators.count(data)) << "duplicate PTO reducer state";
      storage.CheckStorageV1(b);
      states.accumulators.Set(data, b);
      states.storage_values[data] = ValueInfo::Vmi(states.lanes);
    }
  }
  auto check_owned = [&](const Buffer &buffer) {
    storage.CheckStorageV1(buffer);
    const auto &allocation = storage.Allocation(buffer->data);
    bool owned = false;
    for (const auto &b : vf->alloc_buffers)
      owned |= b->data.same_as(buffer->data);
    ICHECK(owned && allocation->dtype == buffer->dtype &&
           ffi::StructuralEqual()(allocation->shape, buffer->shape))
        << "PTO finalize allocation must belong to its VF and match its "
           "storage";
  };
  PostOrderVisit(vf->body, [&](const ObjectRef &node) {
    if (const auto *block = node.as<SBlockNode>())
      ICHECK(block->name_hint != "SIMD_VF")
          << "nested PTO reducer VF is unsupported";
    if (const auto *eval = node.as<EvaluateNode>()) {
      const auto *call = eval->value.as<CallNode>();
      if (call && call->op.same_as(PtoFinalizeReducerOp::Get())) {
        auto op =
            Downcast<PtoFinalizeReducerOp>(ParseOperator(GetRef<Call>(call)));
        int64_t lanes = AnalyzePtoFinalize(op, states);
        states.storage_values[op->result->buffer->data] = ValueInfo::Vmi(lanes);
        check_owned(op->accumulator->buffer);
        check_owned(op->result->buffer);
        if (op->seed.defined())
          check_owned(op->seed.value()->buffer);
        states.finalizes.push_back(op);
      }
    }
  });
  return states;
}

Optional<Buffer> AnalyzeReducerAccess(const PtoReducerStates &states,
                                      const Buffer &buffer,
                                      const Array<PrimExpr> &indices,
                                      const ParallelStorageQuery &storage,
                                      arith::Analyzer *analyzer,
                                      const Map<Var, PrimExpr> &bind_env) {
  auto found = states.accumulators.Get(buffer->data);
  if (!found.has_value())
    return std::nullopt;
  storage.CheckStorageV1(buffer);
  auto slot = storage.ResolveSlot(buffer, indices, analyzer, bind_env);
  ICHECK_EQ(slot.slot, 0) << "PTO reducer slot must be zero";
  return found.value();
}

ScopedPtoLoopBounds::ScopedPtoLoopBounds(arith::Analyzer *&analyzer,
                                         const For &loop)
    : active_(analyzer), outer_(analyzer), inner_(analyzer->Clone()),
      constraints_(inner_.get(),
                   loop->loop_var >= loop->min &&
                       loop->loop_var < loop->min + loop->extent) {
  // ConstraintContext alone cannot give the constant-bound analyzer facts
  // from a symbolic loop minimum or extent. Bind only in that sub-analyzer
  // so a single-iteration range cannot substitute the loop minimum into
  // emitted code.
  inner_->const_int_bound.Bind(loop->loop_var,
                               Range::FromMinExtent(loop->min, loop->extent));
  active_ = inner_.get();
}

ScopedPtoLoopBounds::~ScopedPtoLoopBounds() { active_ = outer_; }

bool IsPtoDirectReduce(const Call &call) {
  return call->op.same_as(Op::Get("tl.tileop.reduce"));
}

bool IsPtoFill(const Call &call) {
  return call->op.same_as(Op::Get("tl.tileop.fill"));
}

namespace {

// These helpers decode buffer metadata without validating TileOp admission or
// constructing a lowering plan. Verify and Vectorize interpret the same
// compiler-generated version prefix and physical strides.
int PtoVersionPrefix(const BufferRegion &region, arith::Analyzer *analyzer,
                     const Map<Var, int> &buffer_versions) {
  const Buffer &buffer = region->buffer;
  auto versions = buffer_versions.Get(buffer->data);
  if (versions.has_value() && versions.value() > 1 &&
      buffer->shape.size() > 1 &&
      analyzer->CanProveEqual(buffer->shape[0], versions.value()) &&
      analyzer->CanProveEqual(region->region[0]->extent, 1)) {
    return 1;
  }
  return 0;
}

Array<PrimExpr> PtoBufferStrides(const Buffer &buffer,
                                 arith::Analyzer *analyzer) {
  if (!buffer->strides.empty()) {
    return buffer->strides;
  }
  Array<PrimExpr> result;
  PrimExpr stride = Integer(1);
  for (size_t i = buffer->shape.size(); i-- > 0;) {
    result.insert(result.begin(), stride);
    stride = analyzer->Simplify(stride * buffer->shape[i]);
  }
  return result;
}

int PtoFillContinuousAxis(const BufferRegion &destination,
                          const Array<PrimExpr> &strides, int prefix,
                          arith::Analyzer *analyzer) {
  // A trailing singleton can have stride 1 as well. Prefer an axis with
  // actual elements so a compact column such as [128, 1] remains one full run.
  for (int axis = static_cast<int>(strides.size()) - 1; axis >= prefix;
       --axis) {
    if (analyzer->CanProveEqual(strides[axis], 1) &&
        !analyzer->CanProveEqual(destination->region[axis]->extent, 1)) {
      return axis;
    }
  }
  int axis = static_cast<int>(strides.size()) - 1;
  return analyzer->CanProveEqual(strides[axis], 1) ? axis : prefix;
}

} // namespace

void CheckPtoFill(const Call &call, int64_t lanes, arith::Analyzer *analyzer,
                  const Map<Var, int> &buffer_versions) {
  auto fail = [](const std::string &message) {
    throw PtoAnalysisError("fill/clear: " + message);
  };
  if (call->args.size() != 2) {
    fail("expected destination region and scalar value");
  }
  BufferRegion destination = NormalizeToBufferRegion(call->args[0]);
  Buffer dst = destination->buffer;
  if ((dst.scope() != "shared" && dst.scope() != "shared.dyn") ||
      !dst->axis_separators.empty()) {
    fail("only ordinary UB (shared/shared.dyn) destinations are supported; "
         "Fragment fill/clear is not supported");
  }
  DataType dtype = dst->dtype;
  if (dtype.lanes() != 1 ||
      !((dtype.is_float() && (dtype.bits() == 16 || dtype.bits() == 32)) ||
        dtype.is_bfloat16() ||
        ((dtype.is_int() || dtype.is_uint()) &&
         (dtype.bits() == 8 || dtype.bits() == 16 || dtype.bits() == 32)))) {
    fail("unsupported destination dtype");
  }
  int prefix = PtoVersionPrefix(destination, analyzer, buffer_versions);
  int rank = static_cast<int>(dst->shape.size()) - prefix;
  if (rank != 1 && rank != 2) {
    fail("requires a logical 1D/2D destination");
  }
  auto positive_constant = [&](const PrimExpr &expr) {
    const auto *value = expr.as<IntImmNode>();
    if (!value || value->value <= 0) {
      fail("buffer shapes and strides must be positive compile-time integers");
    }
    return value->value;
  };
  for (size_t i = prefix; i < dst->shape.size(); ++i) {
    positive_constant(dst->shape[i]);
    const Range &range = destination->region[i];
    if (!analyzer->CanProveEqual(range->min, 0)) {
      fail("nonzero destination region starts are not supported; use a "
           "zero-origin region");
    }
    positive_constant(range->extent);
    if (!analyzer->CanProve(range->extent <= dst->shape[i])) {
      fail("destination region must be provably in bounds");
    }
  }
  Array<PrimExpr> strides = PtoBufferStrides(dst, analyzer);
  for (const PrimExpr &stride : strides) {
    positive_constant(stride);
  }
  if (!analyzer->CanProveEqual(dst->elem_offset, 0)) {
    fail("nonzero destination storage offsets are not supported; use a "
         "zero-origin destination view");
  }
  int64_t alignment = 256 / dtype.bits();
  if (prefix != 0 &&
      !analyzer->CanProveEqual(floormod(strides[0], alignment), 0)) {
    fail("destination version slots must be 32-byte aligned");
  }
  int axis = PtoFillContinuousAxis(destination, strides, prefix, analyzer);
  if (!analyzer->CanProveEqual(strides[axis], 1)) {
    fail("requires a unit-stride axis");
  }
  int64_t run_extent =
      Downcast<IntImm>(destination->region[axis]->extent)->value;
  if (run_extent % lanes != 0) {
    fail("tail regions are not supported; the contiguous extent must be a "
         "multiple of SimdVF lanes");
  }
  if (rank == 2) {
    int outer_axis = prefix + (axis == prefix ? 1 : 0);
    if (!analyzer->CanProveEqual(destination->region[outer_axis]->extent, 1)) {
      if (positive_constant(strides[outer_axis]) <
          positive_constant(dst->shape[axis])) {
        fail("destination strides must describe non-overlapping rows or "
             "columns");
      }
      if (!analyzer->CanProveEqual(floormod(strides[outer_axis], alignment),
                                   0)) {
        fail("destination row or column starts must be 32-byte aligned");
      }
    }
  }
  PrimExpr value = analyzer->Simplify(cast(dtype, call->args[1]));
  std::function<void(const PrimExpr &)> check_value = [&](const PrimExpr
                                                              &value) {
    if (!value.dtype().is_scalar()) {
      fail("fill value must be a uniform scalar");
    }
    if (value.as<IntImmNode>() || value.as<FloatImmNode>() ||
        value.as<VarNode>()) {
      return;
    }
    if (const auto *load = value.as<BufferLoadNode>()) {
      if (load->buffer.scope() != "shared" &&
          load->buffer.scope() != "shared.dyn") {
        fail("scalar fill-value reads must use ordinary UB");
      }
      if (load->predicate.defined()) {
        fail("predicated scalar fill-value reads are not supported");
      }
      for (size_t i = 0; i < load->indices.size(); ++i) {
        if (SideEffect(load->indices[i]) != CallEffectKind::kPure ||
            !analyzer->CanProve(load->indices[i] >= 0) ||
            !analyzer->CanProve(load->indices[i] < load->buffer->shape[i])) {
          fail("scalar fill-value indices must be pure and provably in bounds");
        }
      }
      return;
    }
    if (const auto *conversion = value.as<CastNode>()) {
      CastContext context;
      context.value_kind = CastValueKind::kScalar;
      context.logical_lanes = 1;
      std::string reason;
      if (!PlanValueCast(conversion, context, &reason).has_value()) {
        fail(reason);
      }
      check_value(conversion->value);
      return;
    }
    fail("fill value must be a constant, uniform scalar or one UB read with "
         "supported scalar casts");
  };
  check_value(value);
}

PtoFillPlan PlanPtoFill(const Call &call, arith::Analyzer *analyzer,
                        const Map<Var, int> &buffer_versions) {
  // Verify owns admission and diagnostics. Planning consumes its established
  // invariants without rerunning the checks or changing the input IR.
  PtoFillPlan plan;
  plan.destination = NormalizeToBufferRegion(call->args[0]);
  const Buffer &dst = plan.destination->buffer;
  int prefix = PtoVersionPrefix(plan.destination, analyzer, buffer_versions);
  int rank = static_cast<int>(dst->shape.size()) - prefix;
  Array<PrimExpr> strides = PtoBufferStrides(dst, analyzer);
  plan.continuous_axis =
      PtoFillContinuousAxis(plan.destination, strides, prefix, analyzer);
  plan.run_extent =
      Downcast<IntImm>(plan.destination->region[plan.continuous_axis]->extent)
          ->value;
  if (rank == 2) {
    plan.outer_axis = prefix + (plan.continuous_axis == prefix ? 1 : 0);
    plan.outer_extent =
        Downcast<IntImm>(plan.destination->region[plan.outer_axis]->extent)
            ->value;
  }
  plan.value = analyzer->Simplify(cast(dst->dtype, call->args[1]));
  return plan;
}

void CheckPtoDirectReduce(const Call &call, arith::Analyzer *analyzer,
                          const Map<Var, int> &buffer_versions) {
  auto fail = [&](const std::string &message) {
    throw PtoAnalysisError("direct reduce: " + message);
  };
  if (call->args.size() != 5) {
    fail("expected source, destination, operation, dim and clear");
  }
  const auto *kind = call->args[2].as<StringImmNode>();
  const auto *dim = call->args[3].as<IntImmNode>();
  const auto *clear = call->args[4].as<IntImmNode>();
  if (!kind || !dim || !clear || (clear->value != 0 && clear->value != 1)) {
    fail("requires constant dim, operation and boolean clear");
  }
  std::string operation = kind->value;
  if (operation != "sum" && operation != "max" && operation != "min" &&
      operation != "abssum" && operation != "absmax") {
    fail("only sum, max, min, abssum and absmax are supported");
  }
  for (const auto &[key, value] : call->annotations) {
    if (key == "batch" || key == "nan_propagate") {
      const auto *integer = value.as<IntImmNode>();
      if (!integer || integer->value != (key == "batch" ? 1 : 0)) {
        fail("requires batch=1 and nan_propagate=False");
      }
    }
  }
  BufferRegion source = NormalizeToBufferRegion(call->args[0]);
  BufferRegion destination = NormalizeToBufferRegion(call->args[1]);
  Buffer src = source->buffer;
  Buffer dst = destination->buffer;
  auto ub = [](const Buffer &buffer) {
    return buffer.scope() == "shared" || buffer.scope() == "shared.dyn";
  };
  if (!ub(src) || !ub(dst) || !src->axis_separators.empty() ||
      !dst->axis_separators.empty()) {
    fail("source and destination must be ordinary linear UB buffers");
  }
  if (src->data.same_as(dst->data)) {
    fail("overlapping source/destination storage is not supported");
  }
  DataType from = src->dtype;
  DataType to = dst->dtype;
  bool same =
      from == to && ((to.is_float() && (to.bits() == 16 || to.bits() == 32)) ||
                     ((to.is_int() || to.is_uint()) && to.bits() == 32));
  bool widen = to == DataType::Float(32) &&
               (from == DataType::Float(16) || from.is_bfloat16());
  if ((!same && !widen) || from.lanes() != 1 || to.lanes() != 1) {
    fail("unsupported source/accumulator dtype combination");
  }
  auto logical_shape = [&](const BufferRegion &region, int *prefix) {
    const Buffer &buffer = region->buffer;
    if (region->region.size() != buffer->shape.size()) {
      fail("region rank does not match its buffer");
    }
    *prefix = PtoVersionPrefix(region, analyzer, buffer_versions);
    Array<PrimExpr> shape;
    for (size_t i = *prefix; i < buffer->shape.size(); ++i) {
      const Range &range = region->region[i];
      if (!analyzer->CanProveEqual(range->min, 0) ||
          !analyzer->CanProveEqual(range->extent, buffer->shape[i])) {
        fail("requires complete logical source and destination regions");
      }
      shape.push_back(buffer->shape[i]);
    }
    return shape;
  };
  int source_prefix = 0;
  int destination_prefix = 0;
  Array<PrimExpr> src_shape = logical_shape(source, &source_prefix);
  Array<PrimExpr> dst_shape = logical_shape(destination, &destination_prefix);
  int rank = src_shape.size();
  if ((rank != 1 && rank != 2) || dim->value < 0 || dim->value >= rank) {
    fail("requires a logical 1D/2D source and a valid reduction dim");
  }
  auto positive = [&](const PrimExpr &expr) {
    const auto *integer = expr.as<IntImmNode>();
    if (!integer || integer->value <= 0) {
      fail("all logical extents must be positive compile-time integers");
    }
    return integer->value;
  };
  for (const PrimExpr &extent : src_shape) {
    positive(extent);
  }
  for (const PrimExpr &extent : dst_shape) {
    positive(extent);
  }
  int reduction_axis = dim->value;
  int output_axis = rank == 1 ? 0 : 1 - reduction_axis;
  int64_t output_extent = rank == 1 ? 1 : positive(src_shape[output_axis]);
  bool keep_dims = dst_shape.size() == src_shape.size();
  if (rank == 1) {
    if (dst_shape.size() != 1 || positive(dst_shape[0]) != 1) {
      fail("1D reduction requires destination shape [1]");
    }
  } else if (keep_dims) {
    if (positive(dst_shape[reduction_axis]) != 1 ||
        positive(dst_shape[output_axis]) != output_extent) {
      fail("keepdims destination shape does not match the reduction");
    }
  } else if (dst_shape.size() != 1 || positive(dst_shape[0]) != output_extent) {
    fail("destination shape does not match the reduction");
  }
  int destination_output_axis =
      destination_prefix + (keep_dims && rank == 2 ? output_axis : 0);
  Array<PrimExpr> src_strides = PtoBufferStrides(src, analyzer);
  Array<PrimExpr> dst_strides = PtoBufferStrides(dst, analyzer);
  for (const PrimExpr &stride : src_strides) {
    positive(stride);
  }
  for (const PrimExpr &stride : dst_strides) {
    positive(stride);
  }
  if (rank == 2) {
    int64_t rows = positive(src_shape[0]);
    int64_t columns = positive(src_shape[1]);
    int64_t row_stride = positive(src_strides[source_prefix]);
    int64_t column_stride = positive(src_strides[source_prefix + 1]);
    if (!((column_stride == 1 && row_stride >= columns) ||
          (row_stride == 1 && column_stride >= rows))) {
      fail("source strides must describe non-overlapping rows or columns");
    }
  }
  bool reduce_across_lanes =
      analyzer->CanProveEqual(src_strides[source_prefix + reduction_axis], 1);
  if (!reduce_across_lanes &&
      (rank != 2 ||
       !analyzer->CanProveEqual(src_strides[source_prefix + output_axis], 1) ||
       !analyzer->CanProveEqual(dst_strides[destination_output_axis], 1))) {
    fail("no supported unit-stride reduction/output axis");
  }
}

PtoDirectReducePlan PlanPtoDirectReduce(const Call &call,
                                        arith::Analyzer *analyzer,
                                        const Map<Var, int> &buffer_versions) {
  PtoDirectReducePlan plan;
  plan.source = NormalizeToBufferRegion(call->args[0]);
  plan.destination = NormalizeToBufferRegion(call->args[1]);
  plan.operation = Downcast<StringImm>(call->args[2])->value;
  plan.clear = Downcast<IntImm>(call->args[4])->value != 0;
  const Buffer &src = plan.source->buffer;
  const Buffer &dst = plan.destination->buffer;
  plan.accumulator_dtype = dst->dtype;
  plan.source_prefix = PtoVersionPrefix(plan.source, analyzer, buffer_versions);
  plan.destination_prefix =
      PtoVersionPrefix(plan.destination, analyzer, buffer_versions);
  plan.logical_rank = static_cast<int>(src->shape.size()) - plan.source_prefix;
  plan.reduction_axis = Downcast<IntImm>(call->args[3])->value;
  plan.output_axis = plan.logical_rank == 1 ? 0 : 1 - plan.reduction_axis;
  plan.reduction_extent =
      Downcast<IntImm>(src->shape[plan.source_prefix + plan.reduction_axis])
          ->value;
  plan.output_extent =
      plan.logical_rank == 1
          ? 1
          : Downcast<IntImm>(src->shape[plan.source_prefix + plan.output_axis])
                ->value;
  bool keep_dims =
      static_cast<int>(dst->shape.size()) - plan.destination_prefix ==
      plan.logical_rank;
  plan.destination_output_axis =
      plan.destination_prefix +
      (keep_dims && plan.logical_rank == 2 ? plan.output_axis : 0);
  Array<PrimExpr> src_strides = PtoBufferStrides(src, analyzer);
  plan.reduce_across_lanes = analyzer->CanProveEqual(
      src_strides[plan.source_prefix + plan.reduction_axis], 1);
  return plan;
}

std::string AccessPatternToString(AccessPattern pattern) {
  switch (pattern) {
  case AccessPattern::kContinuous:
    return "continuous";
  case AccessPattern::kLaneUniform:
    return "lane-uniform";
  case AccessPattern::kIndexed:
    return "indexed";
  default:
    return "unknown";
  }
}

// ---------------------------------------------------------------------------
// Shared frontend-op-name -> VMI-op-name tables: one table per concept so
// the expression whitelist, the lane-use classification and the pass-side
// dispatch (Verify admission / Vectorize rewriting) cannot drift apart.
// ---------------------------------------------------------------------------

const char *GetPtoMaskLogicOp(const Call &call) {
  const auto *op = call->op.as<OpNode>();
  if (!op || !call->dtype.is_bool()) {
    return nullptr;
  }
  // TileLang `a & b` / `a | b` / `~a` on bool predicates arrive as
  // tirx.bitwise_* calls (plus the tir.bitwise_* and tir.And/Or/Not
  // spellings); Vectorize rewrites them to the PTODSL bitwise vector ops,
  // whose mask register overload implements the boolean combine.
  static const std::pair<const char *, const char *> kMaskLogicTable[] = {
      {"tirx.bitwise_and", "vand"}, {"tir.bitwise_and", "vand"},
      {"tir.And", "vand"},          {"tirx.bitwise_or", "vor"},
      {"tir.bitwise_or", "vor"},    {"tir.Or", "vor"},
      {"tirx.bitwise_not", "vnot"}, {"tir.bitwise_not", "vnot"},
      {"tir.Not", "vnot"},
  };
  for (const auto &entry : kMaskLogicTable) {
    if (op->name == entry.first) {
      return entry.second;
    }
  }
  return nullptr;
}

const char *VectorUnaryVmiOpName(const std::string &tirx_op_name) {
  // Hardware-backed unary math: fabs/exp/log/sqrt map to the VMI
  // vabs/vexp/vln/vsqrt instructions.
  static const std::pair<const char *, const char *> kUnaryMathTable[] = {
      {"tirx.fabs", "vabs"},
      {"tirx.exp", "vexp"},
      {"tirx.log", "vln"},
      {"tirx.sqrt", "vsqrt"},
  };
  for (const auto &entry : kUnaryMathTable) {
    if (tirx_op_name == entry.first) {
      return entry.second;
    }
  }
  return nullptr;
}

PtoInverseMapping InvertPtoLaneLayout(const For &loop) {
  auto anno = loop->annotations.Get(attr::kPtoParallelLoopLayout);
  if (!anno.has_value()) {
    throw PtoAnalysisError(
        "pto_parallel_loop_layout is missing on the outermost T.Parallel");
  }
  auto fragment_any = anno.value().try_cast<Fragment>();
  if (!fragment_any.has_value()) {
    throw PtoAnalysisError(
        "pto_parallel_loop_layout must be a tl.Fragment annotation");
  }
  Fragment fragment = fragment_any.value();

  // A 2D unit is the outermost For of a two-layer
  // adjacent Parallel nest. The annotation — and the fragment — lives on
  // the outer For and describes the full (i, j) input space; the input
  // coordinate order stays (i, j) whichever dimension is vectorized.
  const ForNode *inner_node = nullptr;
  if (const auto *body_for = loop->body.as<ForNode>()) {
    if (body_for->kind == ForKind::kParallel) {
      inner_node = body_for;
    }
  }
  const bool is_2d = inner_node != nullptr;
  const int expected_dim = is_2d ? 2 : 1;
  if (fragment->InputDim() != expected_dim) {
    std::ostringstream oss;
    oss << "pto_parallel_loop_layout input_size must be " << expected_dim
        << "D for a "
        << (is_2d ? "two-layer nested T.Parallel" : "1D T.Parallel")
        << ", got input dim " << fragment->InputDim();
    throw PtoAnalysisError(oss.str());
  }
  if (fragment->GetForwardIndex().size() != 1) {
    // Extra per-lane position dimensions are a legal-but-unsupported
    // layout shape; reject with a layout diagnostic instead of an internal
    // assertion inside Forward.
    std::ostringstream oss;
    oss << "pto_parallel_loop_layout must have exactly one internal "
           "position dimension, got "
        << fragment->GetForwardIndex().size();
    throw PtoAnalysisError(oss.str());
  }
  if (!is_one(fragment->ReplicateExtent())) {
    throw PtoAnalysisError(
        "pto_parallel_loop_layout requires replicate_size == 1");
  }

  PrimExpr thread_extent = fragment->ThreadExtent();
  const int64_t *lanes_ptr = as_const_int(thread_extent);
  if (lanes_ptr == nullptr || *lanes_ptr <= 0) {
    throw PtoAnalysisError(
        "pto_parallel_loop_layout thread extent must be a positive constant");
  }
  int64_t lanes = *lanes_ptr;
  if (lanes != 64 && lanes != 128 && lanes != 256) {
    std::ostringstream oss;
    oss << "T.SimdVF lanes must be one of {64, 128, 256}, got " << lanes;
    throw PtoAnalysisError(oss.str());
  }

  // Shape contract: the fragment input shape must match the loop extents
  // (1D: the single extent; 2D: (outer, inner)).
  const int64_t *extent_ptr = as_const_int(loop->extent);
  if (extent_ptr == nullptr) {
    // The helper must not assert on inputs the callers did not
    // pre-validate; a symbolic extent is a layout diagnostic (same rule
    // as the inner extent below).
    std::ostringstream oss;
    oss << "pto_parallel_loop_layout loop extent must be a compile-time "
           "constant, got "
        << loop->extent;
    throw PtoAnalysisError(oss.str());
  }
  int64_t outer_extent = *extent_ptr;
  int64_t inner_extent = 1;
  if (is_2d) {
    const int64_t *inner_extent_ptr = as_const_int(inner_node->extent);
    if (inner_extent_ptr == nullptr) {
      // A symbolic inner extent is a layout diagnostic, not an assert.
      std::ostringstream oss;
      oss << "pto_parallel_loop_layout inner loop extent must be a "
             "compile-time constant, got "
          << inner_node->extent;
      throw PtoAnalysisError(oss.str());
    }
    inner_extent = *inner_extent_ptr;
  }
  // Shape contract: the fragment input space is padded on the
  // vectorized dimension. 1D: the single input extent must be the padded
  // extent P = ceil(E / L) * L. 2D: the kept dimension keeps its original
  // loop extent and the selected dimension is padded to P — but which
  // dimension is selected is only known jointly with the canonical mapping
  // after the inversion, so the 2D shape check runs post-inversion. Here
  // the 1D case is checked directly; an unpadded [E] (E % L != 0), an
  // over-padded extent, and (for divisible E) any extent != E are layout
  // diagnostics.
  Array<PrimExpr> input_shape = fragment->InputShape();
  if (!is_2d) {
    if (input_shape.size() != 1u) {
      std::ostringstream oss;
      oss << "pto_parallel_loop_layout input shape " << input_shape
          << " must be 1D for a 1D T.Parallel";
      throw PtoAnalysisError(oss.str());
    }
    const auto [pad_1d, padded_1d] = CheckedPtoPadding(outer_extent, lanes);
    (void)pad_1d;
    const int64_t *frag_extent = as_const_int(input_shape[0]);
    if (frag_extent == nullptr || *frag_extent != padded_1d) {
      std::ostringstream oss;
      oss << "pto_parallel_loop_layout input shape " << input_shape
          << " does not match the padded extent " << padded_1d
          << " (loop extent " << outer_extent << ", lanes " << lanes << "); "
          << "the vectorized dimension must be padded to ceil(E/L)*L";
      throw PtoAnalysisError(oss.str());
    }
  } else {
    if (input_shape.size() != 2u) {
      std::ostringstream oss;
      oss << "pto_parallel_loop_layout input shape " << input_shape
          << " must be 2D for a two-layer nested T.Parallel";
      throw PtoAnalysisError(oss.str());
    }
    // Structural presence check only; the per-dimension contract (kept =
    // original, selected = padded) is a joint check with the canonical
    // mapping below.
    for (const auto &dim : input_shape) {
      if (as_const_int(dim) == nullptr) {
        std::ostringstream oss;
        oss << "pto_parallel_loop_layout input shape " << input_shape
            << " must be compile-time constants, got " << input_shape;
        throw PtoAnalysisError(oss.str());
      }
    }
  }

  // Thread-range contract: [0, L).
  Range thread_range = fragment->ThreadRange();
  if (thread_range.defined()) {
    const int64_t *tr_min = as_const_int(thread_range->min);
    const int64_t *tr_ext = as_const_int(thread_range->extent);
    if (tr_min == nullptr || *tr_min != 0 || tr_ext == nullptr ||
        *tr_ext != lanes) {
      std::ostringstream oss;
      oss << "pto_parallel_loop_layout thread_range must be [0, " << lanes
          << "), got [" << thread_range->min << ", " << thread_range->extent
          << ")";
      throw PtoAnalysisError(oss.str());
    }
  }

  // The fragment is built over int32 placeholder vars (the only
  // dtype LayoutInference produces); a loop var of any other dtype (e.g.
  // int64) crashes the tvm substitution machinery with a raw assertion
  // during inversion. Reject it here with a layout diagnostic instead.
  if (loop->loop_var.dtype() != DataType::Int(32)) {
    std::ostringstream oss;
    oss << "pto_parallel_loop_layout requires an int32 loop variable (the "
           "dtype LayoutInference produces), got "
        << loop->loop_var.dtype();
    throw PtoAnalysisError(oss.str());
  }
  if (is_2d && inner_node->loop_var.dtype() != DataType::Int(32)) {
    std::ostringstream oss;
    oss << "pto_parallel_loop_layout requires an int32 inner loop variable "
           "(the dtype LayoutInference produces), got "
        << inner_node->loop_var.dtype();
    throw PtoAnalysisError(oss.str());
  }

  // Invert through the real inverse layout (single source of truth for the
  // mapping). InverseWithLevel appends the lane (thread) input after the
  // per-lane position input. It raises when the fragment is not a valid
  // bijection — e.g. a hand-built layout whose per-thread position extent
  // does not cover the input space: Layout::InverseWithLevel throws
  // NormalizeIterException when the iteration map has errors, and the
  // iteration-map machinery itself reports tvm errors as ffi::Error. Both are
  // known input failures here, so they become layout diagnostics. Only this
  // call is wrapped, so a genuine implementation error anywhere else still
  // surfaces as one instead of being reported as bad input.
  auto invert = [&]() {
    try {
      return fragment->InverseWithLevel(/*require_padding_guard=*/false);
    } catch (const NormalizeIterException &err) {
      throw PtoAnalysisError(std::string("pto_parallel_loop_layout cannot be "
                                         "inverted: ") +
                             err.what());
    } catch (const ffi::Error &err) {
      throw PtoAnalysisError(std::string("pto_parallel_loop_layout cannot be "
                                         "inverted: ") +
                             err.what());
    }
  };
  auto inverse_info = invert();
  if (inverse_info.second != arith::IterMapLevel::Bijective) {
    std::ostringstream oss;
    oss << "pto_parallel_loop_layout is not bijectively invertible (level "
        << static_cast<int>(inverse_info.second)
        << "): " << fragment->DebugOutput();
    throw PtoAnalysisError(oss.str());
  }
  Layout inverse = inverse_info.first;

  // Forward (chunk, lane) through the inverse layout to obtain the logical
  // iteration coordinates followed by the replicate coordinate. With
  // replicate_size == 1, the trailing replicate coordinate is zero.
  // The branches below use the logical coordinates for the loop dimensionality.
  Var chunk_var("pto_chunk", loop->loop_var.dtype());
  Var lane_var("pto_lane", loop->loop_var.dtype());
  Array<PrimExpr> logical = inverse->Forward({chunk_var, lane_var});
  // The output count follows the input dimensionality (1D: (i, replicate);
  // 2D: (i, j, replicate)); each branch below checks its own arity.

  // Only the canonical mappings are accepted. Any other
  // invertible form (e.g. a reversed lane assignment) is rejected instead
  // of being silently replaced with our own formula.
  //   1D:      i = chunk * L + lane
  //   2D (j):  j = (chunk * L + lane) % N, i = (chunk * L + lane) / N
  //   2D (i):  i = (chunk * L + lane) % M, j = (chunk * L + lane) / M
  PtoInverseMapping mapping;
  mapping.lane_var = lane_var;
  mapping.chunk_var = chunk_var;
  mapping.lanes = lanes;

  if (!is_2d) {
    ICHECK_EQ(logical.size(), 2u);
    // The fragment input space is padded to P, so the chunk
    // position range is [0, P/L) — the padded chunk count Q.
    const int64_t padded = CheckedPtoPadding(outer_extent, lanes).second;
    arith::Analyzer probe;
    probe.Bind(chunk_var,
               Range::FromMinExtent(make_zero(chunk_var->dtype),
                                    IntImm(chunk_var->dtype, padded / lanes)));
    probe.Bind(lane_var, Range::FromMinExtent(make_zero(lane_var->dtype),
                                              IntImm(lane_var->dtype, lanes)));
    PrimExpr canonical = chunk_var * IntImm(chunk_var->dtype, lanes) + lane_var;
    if (!probe.CanProveEqual(logical[0], canonical)) {
      std::ostringstream oss;
      oss << "pto_parallel_loop_layout does not use the canonical 1D "
             "mapping i = chunk * L + lane (got "
          << logical[0] << " from " << fragment->DebugOutput() << ")";
      throw PtoAnalysisError(oss.str());
    }
    mapping.logical_extent = outer_extent;
    mapping.padded_extent = padded;
    mapping.chunk_count = padded / lanes;
    mapping.tail_lanes = outer_extent % lanes;
    mapping.index_expr = logical[0];
    return mapping;
  }

  ICHECK_EQ(logical.size(), 3u);
  const int64_t M = outer_extent;
  const int64_t N = inner_extent;
  // The fragment input space is padded on the vectorized dimension, so
  // the flattened position t = kept * P_sel + sel lives in [0, K * P_sel)
  // and the global chunk count is K * ceil(E / L) (each row padded
  // independently — not ceil(K * E / L), which is not equivalent). This
  // probe domain is for the canonical mapping check only; the
  // address-proof domain that AnalyzeBufferAccess uses binds the per-row
  // q range instead.
  // Each interpretation gets its own probe with the exact chunk domain
  // [0, kept * P_sel / L): a domain bound larger than the fragment's real
  // chunk count could make an identity that holds on the real domain
  // unprovable, and a smaller one could admit a mapping that only looks
  // canonical on part of the space. The lane range [0, L) is shared.
  const auto bind_probe = [&](arith::Analyzer *probe, int64_t chunk_bound) {
    probe->Bind(chunk_var,
                Range::FromMinExtent(make_zero(chunk_var->dtype),
                                     IntImm(chunk_var->dtype, chunk_bound)));
    probe->Bind(lane_var, Range::FromMinExtent(make_zero(lane_var->dtype),
                                               IntImm(lane_var->dtype, lanes)));
  };
  PrimExpr t = chunk_var * IntImm(chunk_var->dtype, lanes) + lane_var;
  // The vectorized coordinate is t % P_sel, the kept one t / P_sel — both
  // the modulus and the divisor use the *padded* selected extent. The
  // interpretation is accepted only when it also agrees with the fragment
  // input shape: the kept dimension keeps its original loop extent and
  // the selected dimension is padded up to P_sel (unpadded / over-padded
  // / wrongly-modified kept dims each fail here). A degenerate dimension
  // (extent 1) makes both canonical forms provable; the shape agreement
  // breaks the tie exactly when only one dim was padded, and the inner
  // preference remains for fully divisible shapes.
  const auto matches = [&](bool select_inner) -> bool {
    const int64_t sel_extent = select_inner ? N : M;
    const auto [pad, padded] = CheckedPtoPadding(sel_extent, lanes);
    const int64_t kept_extent = select_inner ? M : N;
    const int64_t *frag_kept = as_const_int(input_shape[select_inner ? 0 : 1]);
    const int64_t *frag_sel = as_const_int(input_shape[select_inner ? 1 : 0]);
    if (frag_kept == nullptr || frag_sel == nullptr ||
        *frag_kept != kept_extent || *frag_sel != padded) {
      return false;
    }
    (void)pad;
    arith::Analyzer probe;
    bind_probe(&probe, kept_extent * padded / lanes);
    const PrimExpr &sel = select_inner ? logical[1] : logical[0];
    const PrimExpr &kept = select_inner ? logical[0] : logical[1];
    PrimExpr sel_imm = IntImm(t.dtype(), padded);
    return probe.CanProveEqual(sel, FloorMod(t, sel_imm)) &&
           probe.CanProveEqual(kept, FloorDiv(t, sel_imm));
  };
  const bool inner_ok = matches(/*select_inner=*/true);
  const bool outer_ok = matches(/*select_inner=*/false);
  if (!inner_ok && !outer_ok) {
    std::ostringstream oss;
    oss << "pto_parallel_loop_layout does not use a canonical 2D mapping "
           "(j = t % P_N, i = t / P_N  or  i = t % P_M, j = t / P_M with "
           "t = chunk * L + lane, P_* = padded selected extent; the kept "
           "dim keeps its loop extent in the fragment shape); got (i, j) = ("
        << logical[0] << ", " << logical[1] << "), shape " << input_shape
        << ", loop extents (" << M << ", " << N << ") from "
        << fragment->DebugOutput();
    throw PtoAnalysisError(oss.str());
  }
  // Both dims matching is degenerate (e.g. M == N == 1); prefer the inner
  // dimension, consistent with the layout-selection rule.
  const bool select_inner = inner_ok;
  const int64_t sel_extent = select_inner ? N : M;
  const auto [pad_sel, P_sel] = CheckedPtoPadding(sel_extent, lanes);
  (void)pad_sel;
  mapping.is_2d = true;
  mapping.outer_var = loop->loop_var;
  mapping.inner_var = inner_node->loop_var;
  mapping.outer_extent = M;
  mapping.inner_extent = N;
  mapping.select_inner = select_inner;
  mapping.logical_extent = sel_extent;
  mapping.padded_extent = P_sel;
  mapping.chunk_count = P_sel / lanes;
  mapping.tail_lanes = sel_extent % lanes;
  mapping.index_expr = t;
  return mapping;
}

namespace {

[[noreturn]] void Fail(const std::string &message, const Span &span) {
  throw PtoAnalysisError(message + SpanHintSuffix(span));
}

} // namespace

// A value can be widened to signed 64-bit without changing its value for every
// signed integer and for unsigned integers narrower than 64 bits.  An unsigned
// 64-bit value is safe only when its range is proven finite; otherwise the
// widening cast could reinterpret values above INT64_MAX.
bool RepresentableInInt64(arith::Analyzer *analyzer, const PrimExpr &value) {
  const DataType dtype = value.dtype();
  if (!dtype.is_uint() || dtype.bits() < 64) {
    return true;
  }
  return analyzer->const_int_bound(value)->max_value !=
         arith::ConstIntBound::kPosInf;
}

PrimExpr WidenToInt64(const PrimExpr &value) {
  if (value.dtype().is_int() && value.dtype().bits() == 64) {
    return value;
  }
  return Cast(DataType::Int(64), value);
}

void ValidateUbBuffer(const Buffer &buffer, const std::string &op_name,
                      const Span &span) {
  String scope = buffer.scope();
  if (!IsOrdinaryUbScope(scope)) {
    Fail(op_name + " requires a UB pointer, got buffer scope '" +
             std::string(scope.c_str(), scope.size()) + "'",
         span);
  }
  if (!buffer->axis_separators.empty()) {
    Fail(op_name +
             " requires a single scalar linear buffer address, but buffer '" +
             std::string(buffer->name.c_str(), buffer->name.size()) +
             "' declares axis separators",
         span);
  }
}

Array<PrimExpr> BufferZeros(const Buffer &buffer) {
  DataType dtype =
      buffer->shape.empty() ? DataType::Int(32) : buffer->shape[0].dtype();
  Array<PrimExpr> starts;
  for (size_t i = 0; i < buffer->shape.size(); ++i) {
    starts.push_back(IntImm(dtype, 0));
  }
  return starts;
}

std::string BufferContext(const Buffer &buffer) {
  std::ostringstream oss;
  oss << "buffer '" << buffer->name << "': shape=[";
  for (size_t i = 0; i < buffer->shape.size(); ++i) {
    if (i != 0) {
      oss << ", ";
    }
    oss << buffer->shape[i];
  }
  oss << "]";
  if (!buffer->strides.empty()) {
    oss << ", strides=[";
    for (size_t i = 0; i < buffer->strides.size(); ++i) {
      if (i != 0) {
        oss << ", ";
      }
      oss << buffer->strides[i];
    }
    oss << "]";
  }
  oss << ", elem_offset=" << buffer->elem_offset;
  return oss.str();
}

PrimExpr MakeDescriptor(const Buffer &buffer, const Array<PrimExpr> &starts,
                        const PrimExpr &extent, int rw_mask) {
  return Call(
      DataType::Handle(), tl::access_ptr(),
      {BufferLoad(buffer, starts), extent, IntImm(DataType::Int(32), rw_mask)});
}

Buffer FindAllocationOwner(const std::vector<Buffer> &active_allocations,
                           const Buffer &source, const std::string &op_name,
                           const Span &span) {
  std::optional<Buffer> owner;
  for (const Buffer &buffer : active_allocations) {
    if (!buffer->data.same_as(source->data)) {
      continue;
    }
    if (!owner.has_value()) {
      owner = buffer;
    } else if (!owner->same_as(buffer)) {
      Fail(op_name +
               " found multiple active allocations sharing the source data; "
               "the owner is ambiguous, and different allocation objects are "
               "never merged by data pointer",
           span);
    }
  }
  if (!owner.has_value()) {
    Fail(op_name +
             " requires a unique local allocation that is active at this "
             "position and owns the source data; no active allocation was "
             "found (a parameter, view, or full-function collection table is "
             "not an owner)",
         span);
  }
  return owner.value();
}

bool StableIndexedAddressNode(const ObjectRef &node) {
  return node.as<VarNode>() || node.as<IntImmNode>() || node.as<CastNode>() ||
         node.as<AddNode>() || node.as<SubNode>() || node.as<MulNode>() ||
         node.as<FloorDivNode>() || node.as<FloorModNode>() ||
         node.as<MinNode>() || node.as<MaxNode>();
}

// One traversal for the checks shared by address and capacity snapshots.
// Each caller supplies its own admission set and keeps its own diagnostics;
// sharing the walk must not merge those sets.  Post-order reaches a read or a
// call only after its arguments, so collect the whole expression and let the
// caller give reads and calls priority over an incidental leaf type from
// inside them.  This pass conservatively rejects every direct Call in these
// snapshots, including a pure call; it does not classify whether a particular
// pure call would be evaluable at the use point.
ExpressionStabilityScan
ScanExpressionStability(const PrimExpr &expr,
                        bool (*is_stable)(const ObjectRef &)) {
  ExpressionStabilityScan scan;
  PostOrderVisit(expr, [&](const ObjectRef &node) {
    scan.has_load = scan.has_load || node.as<BufferLoadNode>() != nullptr;
    scan.has_call = scan.has_call || node.as<CallNode>() != nullptr;
    if (scan.first_unstable_type.empty() && !is_stable(node)) {
      scan.first_unstable_type = node->GetTypeKey();
    }
  });
  return scan;
}

namespace {

// The address snapshot whitelist above is deliberately narrow because it
// feeds bound-pointer reconstruction.  An owner capacity only needs a stable
// scalar expression that can be evaluated at the use point, so it also admits
// truncating div/mod, comparisons, boolean combinations, and Select.  Keep
// this classification local: extending it must not relax address admission.
bool StableOwnerCapacityNode(const ObjectRef &node) {
  return StableIndexedAddressNode(node) || node.as<DivNode>() ||
         node.as<ModNode>() || node.as<SelectNode>() || node.as<EQNode>() ||
         node.as<NENode>() || node.as<LTNode>() || node.as<LENode>() ||
         node.as<GTNode>() || node.as<GENode>() || node.as<AndNode>() ||
         node.as<OrNode>() || node.as<NotNode>();
}

// Inspect the allocation shape nodes before Analyzer simplification.
// A shape such as `read * 0 + 64` folds
// to a constant, but the physical capacity was still fixed by a mutable read
// at allocation time; the use-point extent must not re-evaluate that read.
// A Var is a snapshot leaf: its definition is deliberately not followed, so a
// scalar saved before the allocation remains a stable capacity even when the
// memory behind it changes later.
std::optional<std::string> OwnerCapacityStabilityReason(const Buffer &owner) {
  for (size_t dim = 0; dim < owner->shape.size(); ++dim) {
    const PrimExpr &shape = owner->shape[dim];
    ExpressionStabilityScan scan =
        ScanExpressionStability(shape, StableOwnerCapacityNode);
    std::string reason;
    if (scan.has_load) {
      reason = "the shape re-reads a buffer at the use point; save the shape "
               "value in a scalar Var before allocating the owner";
    } else if (scan.has_call) {
      reason =
          "the shape executes a call at the use point; save the shape value "
          "in a scalar Var before allocating the owner";
    } else if (!scan.first_unstable_type.empty()) {
      std::ostringstream oss;
      oss << "the shape contains a non-stable node of type "
          << scan.first_unstable_type
          << "; save the shape value in a scalar Var before allocating the "
             "owner";
      reason = oss.str();
    }
    if (!reason.empty()) {
      std::ostringstream oss;
      oss << "owner capacity is not a stable snapshot: shape dimension " << dim
          << " (length " << shape << ", " << BufferContext(owner)
          << ") cannot be rebuilt at the use point: " << reason;
      return oss.str();
    }
  }
  return std::nullopt;
}

std::optional<std::string> OwnerTightReason(arith::Analyzer *analyzer,
                                            const Buffer &owner) {
  size_t ndim = owner->shape.size();
  if (ndim == 0) {
    return std::string("the owner has no dimensions");
  }
  if (owner->strides.empty()) {
    return std::nullopt;
  }
  if (owner->strides.size() != ndim) {
    return std::string("the owner stride rank ") +
           std::to_string(owner->strides.size()) + " does not match its rank " +
           std::to_string(ndim);
  }
  // Compute the expected tight strides from the inner shape dimensions in a
  // value-preserving int64 domain.  Multiplying in the shape dtype wraps: for
  // shape (2, 67108865, 64) the int32 product 67108865 * 64 becomes 64, which
  // would accept an overlapping stride[0]=64 and reject the mathematically
  // tight stride[0]=4294967360.  The layout contract itself is unchanged.
  PrimExpr expected = IntImm(DataType::Int(64), 1);
  for (size_t dim = ndim; dim-- > 0;) {
    const PrimExpr &stride = owner->strides[dim];
    if (!RepresentableInInt64(analyzer, stride)) {
      std::ostringstream oss;
      oss << "the owner layout is not provably tight row-major: stride[" << dim
          << "]=" << stride
          << " has an unsigned 64-bit range that cannot be compared without "
             "losing integer precision";
      return oss.str();
    }
    if (!analyzer->CanProveEqual(WidenToInt64(stride), expected)) {
      std::ostringstream oss;
      oss << "the owner layout is not provably tight row-major: stride[" << dim
          << "]=" << stride << " does not equal the expected " << expected;
      return oss.str();
    }
    if (dim > 0) {
      const PrimExpr &factor = owner->shape[dim];
      if (!RepresentableInInt64(analyzer, factor)) {
        std::ostringstream oss;
        oss << "the owner layout is not provably tight row-major: dimension "
            << dim << " has an unsigned 64-bit length that cannot be widened "
            << "without losing integer precision";
        return oss.str();
      }
      __int128 product_bound =
          static_cast<__int128>(
              analyzer->const_int_bound(expected)->max_value) *
          static_cast<__int128>(analyzer->const_int_bound(factor)->max_value);
      if (product_bound > std::numeric_limits<int64_t>::max()) {
        std::ostringstream oss;
        oss << "the owner layout is not provably tight row-major within the "
               "signed 64-bit range: the expected stride outside dimension "
            << dim - 1 << " is the product of dimensions " << dim << ".."
            << ndim - 1 << ", whose upper bound exceeds int64";
        return oss.str();
      }
      expected = analyzer->Simplify(expected * WidenToInt64(factor));
    }
  }
  return std::nullopt;
}

PrimExpr DescribeOwnerExtent(const Buffer &owner) {
  // Multiply in int64 without changing the dtype of arithmetic inside each
  // shape expression. Keep allocation-time Vars in the descriptor.
  PrimExpr extent = IntImm(DataType::Int(64), 1);
  for (const PrimExpr &shape : owner->shape) {
    extent = extent * WidenToInt64(shape);
  }
  arith::Analyzer expression_analyzer;
  return expression_analyzer.Simplify(extent);
}

PrimExpr CheckOwnerCapacity(arith::Analyzer *analyzer, const Buffer &owner,
                            const std::string &op_name, const Span &span) {
  // Upper bound of the mathematical shape product, computed without ever
  // evaluating the product in the shape dtype.  `unbounded` counts symbolic
  // factors whose ConstIntBound has no finite maximum.
  int64_t max_extent = 1;
  int unbounded = 0;
  bool unbounded_uint = false;

  auto capacity_failure = [&](const std::string &detail) {
    Fail(op_name +
             " cannot represent the whole-owner element capacity as a stable "
             "descriptor extent: " +
             detail,
         span);
  };

  for (size_t dim = 0; dim < owner->shape.size(); ++dim) {
    const PrimExpr &shape = owner->shape[dim];
    const int64_t *length = as_const_int(shape);
    int64_t factor_bound = 1;
    if (length && !shape.dtype().is_bool()) {
      if (*length <= 0) {
        Fail(op_name + " owner dimension " + std::to_string(dim) +
                 " has non-positive length " + std::to_string(*length),
             span);
      }
      factor_bound = *length;
    } else {
      if (!analyzer->CanProve(shape > 0)) {
        Fail(op_name + " cannot prove owner dimension " + std::to_string(dim) +
                 " is positive",
             span);
      }
      if (!analyzer->CanProveEqual(shape, make_const(shape.dtype(), 1))) {
        arith::ConstIntBound bound = analyzer->const_int_bound(shape);
        if (bound->max_value == arith::ConstIntBound::kPosInf) {
          ++unbounded;
          unbounded_uint = unbounded_uint || shape.dtype().is_uint();
        } else {
          if (bound->max_value <= 0) {
            capacity_failure("dimension " + std::to_string(dim) +
                             " has no positive upper bound");
          }
          factor_bound = bound->max_value;
        }
      }
    }
    if (factor_bound > 1) {
      __int128 next = static_cast<__int128>(max_extent) * factor_bound;
      if (next > std::numeric_limits<int64_t>::max()) {
        capacity_failure("the shape product overflows int64");
      }
      max_extent = static_cast<int64_t>(next);
    }
  }
  if (unbounded > 0) {
    // A single unbounded factor is representable only when every other factor
    // is one: the capacity then is that factor itself, not a wider product.
    if (unbounded > 1 || max_extent > 1 || unbounded_uint) {
      capacity_failure(
          "a symbolic owner dimension has no finite upper bound, so the shape "
          "product cannot be proven representable in int64");
    }
  }

  // Check the mathematical product before simplifying in finite-width TIR.
  PrimExpr extent = DescribeOwnerExtent(owner);
  if (!analyzer->CanProve(extent > 0)) {
    Fail(op_name +
             " cannot prove the whole-owner element capacity is positive; the "
             "owner shape product is not a stable capacity proof",
         span);
  }
  return extent;
}

} // namespace

IndexedMemoryOwner CheckIndexedAllocationStorage(arith::Analyzer *analyzer,
                                                 const Buffer &owner,
                                                 const std::string &op_name,
                                                 const Span &span) {
  // Capacity stability is an owner property and must be checked before any
  // Analyzer folding, coverage, or extent construction can rewrite the shape.
  if (std::optional<std::string> stability =
          OwnerCapacityStabilityReason(owner)) {
    Fail(op_name + " " + stability.value(), span);
  }
  ValidateUbBuffer(owner, op_name + " owner", span);
  if (std::optional<std::string> tight = OwnerTightReason(analyzer, owner)) {
    Fail(op_name + " owner is not usable: " + tight.value(), span);
  }
  PrimExpr origin = owner.OffsetOf(BufferZeros(owner))[0];
  if (!analyzer->CanProveEqual(origin, make_zero(origin.dtype()))) {
    std::ostringstream oss;
    oss << op_name << " owner origin must be provably zero, but elem_offset "
        << origin << " (" << BufferContext(owner) << ") is not provably zero";
    Fail(oss.str(), span);
  }
  return {owner, origin, CheckOwnerCapacity(analyzer, owner, op_name, span)};
}

void CheckIndexedTypedView(arith::Analyzer *analyzer, const Buffer &view,
                           const Buffer &owner, const PrimExpr &typed_extent,
                           const std::string &op_name, const Span &span) {
  DataType owner_dtype = owner->dtype;
  DataType access_dtype = view->dtype;
  if (!owner_dtype.is_scalar() || owner_dtype.bits() < 8 ||
      owner_dtype.bits() % 8 != 0 || !access_dtype.is_scalar() ||
      access_dtype.bits() < 8 || access_dtype.bits() % 8 != 0) {
    Fail(op_name + " supports dtype-changing indexed views only for scalar "
                   "byte-addressable storage types",
         span);
  }
  int64_t access_bytes = access_dtype.bits() / 8;
  if (owner->data_alignment % access_bytes != 0) {
    Fail(op_name +
             " cannot prove the allocation owner is aligned for the access "
             "element type",
         span);
  }
  if (std::optional<std::string> stability =
          OwnerCapacityStabilityReason(view)) {
    Fail(op_name + " dtype-changing view " + stability.value(), span);
  }
  Array<PrimExpr> view_origin = view.OffsetOf(BufferZeros(view));
  if (view_origin.size() != 1 ||
      !RepresentableInInt64(analyzer, view_origin[0])) {
    Fail(op_name + " requires a single, value-preserving scalar origin for a "
                   "dtype-changing view",
         span);
  }
  PrimExpr origin64 = WidenToInt64(view_origin[0]);
  auto tight = OwnerTightReason(analyzer, view);
  if (tight && owner_dtype.bits() == access_dtype.bits() &&
      !view->strides.empty()) {
    // Prove the physical footprint in owner elements. Debit each axis from
    // the remaining capacity only after proving the product fits.
    auto range_error = [&]() {
      Fail(op_name + " dtype-changing view range is not provably contained in "
                     "the allocation owner",
           span);
    };
    if (view->strides.size() != view->shape.size() ||
        !analyzer->CanProve(origin64 >= 0) ||
        !analyzer->CanProve(origin64 < typed_extent)) {
      range_error();
    }
    PrimExpr remaining = analyzer->Simplify(typed_extent - 1 - origin64);
    for (size_t dim = 0; dim < view->shape.size(); ++dim) {
      const PrimExpr &length = view->shape[dim];
      const PrimExpr &stride = view->strides[dim];
      if (!RepresentableInInt64(analyzer, length) ||
          !RepresentableInInt64(analyzer, stride) ||
          !analyzer->CanProve(length > 0) || !analyzer->CanProve(stride >= 0) ||
          !ScanExpressionStability(stride, StableOwnerCapacityNode)
               .first_unstable_type.empty()) {
        range_error();
      }
      PrimExpr steps = analyzer->Simplify(WidenToInt64(length) - 1);
      if (analyzer->CanProveEqual(steps, 0) ||
          analyzer->CanProveEqual(stride, 0)) {
        continue;
      }
      PrimExpr pitch = WidenToInt64(stride);
      if (!analyzer->CanProve(steps > 0) ||
          !analyzer->CanProve(pitch <= FloorDiv(remaining, steps))) {
        range_error();
      }
      remaining = analyzer->Simplify(remaining - steps * pitch);
    }
    return;
  }
  if (tight) {
    Fail(op_name + " dtype-changing view is not usable: " + tight.value(),
         span);
  }

  PrimExpr view_extent = CheckOwnerCapacity(analyzer, view, op_name, span);
  if (!analyzer->CanProve(origin64 >= 0) ||
      !analyzer->CanProve(origin64 <= typed_extent) ||
      !analyzer->CanProve(view_extent <= typed_extent - origin64)) {
    Fail(op_name +
             " dtype-changing view range is not provably contained in the "
             "allocation owner",
         span);
  }
}

IndexedMemoryOwner CheckIndexedMemoryOwner(const Buffer &source,
                                           const Buffer &owner,
                                           arith::Analyzer *analyzer,
                                           const std::string &context,
                                           const Span &span) {
  ValidateUbBuffer(source, context + " source", span);
  IndexedMemoryOwner storage =
      CheckIndexedAllocationStorage(analyzer, owner, context, span);
  if (source->dtype == owner->dtype) {
    return storage;
  }
  if (source->dtype.bits() != owner->dtype.bits()) {
    Fail(context + " requires the indexed view and allocation owner to have "
                   "the same element width",
         span);
  }
  CheckIndexedTypedView(analyzer, source, owner, storage.extent, context, span);
  return DescribeIndexedMemoryOwner(source, owner);
}

IndexedMemoryOwner DescribeIndexedMemoryOwner(const Buffer &source,
                                              const Buffer &owner) {
  Buffer descriptor = owner;
  if (source->dtype != owner->dtype) {
    // Same-width views preserve the allocation's strides and version axis.
    descriptor.CopyOnWrite()->dtype = source->dtype;
  }
  return {descriptor, owner.OffsetOf(BufferZeros(owner))[0],
          DescribeOwnerExtent(owner)};
}

PrimExpr MakeIndexedMemoryDescriptor(const IndexedMemoryOwner &owner,
                                     bool is_write,
                                     const Optional<PrimExpr> &version,
                                     const Optional<PrimExpr> &extent) {
  Array<PrimExpr> starts = BufferZeros(owner.buffer);
  if (version) {
    starts.Set(0, version.value());
  }
  return MakeDescriptor(owner.buffer, starts, extent.value_or(owner.extent),
                        is_write ? 2 : 1);
}

namespace {
// A private slot is a vector after promotion, even when its last assignment
// read a scalar. Keep the assigned value as an SSA operand for address
// analysis. Taking the snapshot at the store also preserves an earlier UB
// read if its source is subsequently overwritten.
class PrivateIndexValues : public StmtExprMutator {
public:
  PrivateIndexValues(const For &unit, std::set<const VarNode *> *created)
      : created_(created) {
    auto owners = unit->annotations.Get(attr::kParallelPrivateAllocs);
    if (!owners)
      return;
    std::set<const VarNode *> private_data;
    for (const Var &local : Downcast<Array<Var>>(*owners))
      private_data.insert(local.get());
    std::map<const VarNode *, std::vector<PrimExpr>> definitions;
    std::vector<PrimExpr> pending;
    PostOrderVisit(unit->body, [&](const ObjectRef &node) {
      if (const auto *load = node.as<BufferLoadNode>()) {
        pending.insert(pending.end(), load->indices.begin(),
                       load->indices.end());
      } else if (const auto *store = node.as<BufferStoreNode>()) {
        pending.insert(pending.end(), store->indices.begin(),
                       store->indices.end());
        if (private_data.count(store->buffer->data.get()))
          definitions[store->buffer->data.get()].push_back(store->value);
      } else if (const auto *bind = node.as<BindNode>()) {
        definitions[bind->var.get()].push_back(bind->value);
      }
    });
    // Follow only definitions that feed an address, including slot-to-slot
    // copies and intermediate Bind variables.
    while (!pending.empty()) {
      PrimExpr expr = pending.back();
      pending.pop_back();
      PostOrderVisit(expr, [&](const ObjectRef &node) {
        auto *key = node.as<VarNode>();
        if (auto *load = node.as<BufferLoadNode>();
            load && private_data.count(load->buffer->data.get()) &&
            IsPlainSlot(load->buffer, load->indices, load->predicate))
          key = load->buffer->data.get();
        if (auto it = definitions.find(key); it != definitions.end()) {
          targets_.insert(key);
          pending.insert(pending.end(), it->second.begin(), it->second.end());
          definitions.erase(it);
        }
      });
    }
  }

  Stmt Rewrite(const Stmt &body) {
    return targets_.empty() ? body : VisitStmt(body);
  }

private:
  // Forward only the ordinary single-slot representation. Other views and
  // predicated reads still go through Verify's original storage checks.
  static bool IsPlainSlot(const Buffer &buffer, const Array<PrimExpr> &indices,
                          const Optional<PrimExpr> &predicate) {
    return !predicate && buffer->shape.size() == 1 &&
           is_one(buffer->shape[0]) && buffer->strides.empty() &&
           is_zero(buffer->elem_offset) && indices.size() == 1 &&
           is_zero(indices[0]) && IsPtoIndexDType(buffer->dtype);
  }

  void ForgetWrites(const Stmt &stmt) {
    PostOrderVisit(stmt, [&](const ObjectRef &node) {
      if (const auto *store = node.as<BufferStoreNode>()) {
        auto it = values_.find(store->buffer->data.get());
        if (it != values_.end() &&
            !(IsPlainSlot(store->buffer, store->indices, store->predicate) &&
              ExprDeepEqual()(it->second, store->value)))
          values_.erase(it);
      }
    });
  }

  Stmt VisitStmt_(const ForNode *op) final {
    // A loop-carried slot may differ on its second iteration and after the
    // loop. An untouched outer slot remains available inside the loop.
    ForgetWrites(op->body);
    auto entering = values_;
    Stmt body = StmtExprMutator::VisitStmt_(op);
    values_ = std::move(entering);
    return body;
  }

  Stmt VisitStmt_(const IfThenElseNode *op) final {
    auto entering = values_;
    PrimExpr condition = VisitExpr(op->condition);
    Stmt yes = VisitStmt(op->then_case);
    values_ = entering;
    Optional<Stmt> no;
    if (op->else_case)
      no = VisitStmt(*op->else_case);
    values_ = std::move(entering);
    ForgetWrites(GetRef<Stmt>(op));
    return IfThenElse(condition, yes, no, op->span);
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    auto entering = values_;
    Stmt result = StmtExprMutator::VisitStmt_(op);
    values_ = std::move(entering);
    ForgetWrites(op->body);
    return result;
  }

  PrimExpr SubstituteValue(const PrimExpr &expr) {
    class SubstituteReads : public ExprMutator {
    public:
      explicit SubstituteReads(
          const std::map<const VarNode *, PrimExpr> &values)
          : values_(values) {}
      PrimExpr VisitExpr_(const BufferLoadNode *op) final {
        auto it = values_.find(op->buffer->data.get());
        if (it != values_.end() && it->second.dtype() == op->dtype &&
            IsPlainSlot(op->buffer, op->indices, op->predicate))
          return it->second;
        return ExprMutator::VisitExpr_(op);
      }

    private:
      const std::map<const VarNode *, PrimExpr> &values_;
    } remap(values_);
    return remap(expr);
  }

  Stmt VisitStmt_(const BindNode *op) final {
    if (targets_.count(op->var.get()))
      return Bind(op->var, SubstituteValue(op->value), op->span);
    return StmtExprMutator::VisitStmt_(op);
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    auto indices =
        op->indices.Map([&](const PrimExpr &e) { return SubstituteValue(e); });
    return indices.same_as(op->indices)
               ? GetRef<PrimExpr>(op)
               : BufferLoad(op->buffer, indices, op->predicate, op->span);
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    auto indices =
        op->indices.Map([&](const PrimExpr &e) { return SubstituteValue(e); });
    auto value = VisitExpr(op->value);
    if (!targets_.count(op->buffer->data.get()) ||
        !IsPlainSlot(op->buffer, op->indices, op->predicate)) {
      values_.erase(op->buffer->data.get());
      return BufferStore(op->buffer, value, indices, op->predicate, op->span);
    }
    value = SubstituteValue(value);
    if (value.as<VarNode>() || value.as<IntImmNode>()) {
      values_[op->buffer->data.get()] = value;
      return BufferStore(op->buffer, value, indices, op->predicate, op->span);
    }
    Var snapshot(op->buffer->name + "_index", value.dtype());
    if (created_)
      created_->insert(snapshot.get());
    values_[op->buffer->data.get()] = snapshot;
    return SeqStmt::Flatten(
        Bind(snapshot, value),
        BufferStore(op->buffer, snapshot, indices, op->predicate, op->span));
  }

  std::set<const VarNode *> *created_;
  std::set<const VarNode *> targets_;
  std::map<const VarNode *, PrimExpr> values_;
};

} // namespace

Stmt LegalizePrivateIndexValues(const For &unit,
                                std::set<const VarNode *> *created) {
  return PrivateIndexValues(unit, created).Rewrite(unit->body);
}

void SetPtoIndexStorage(PtoInverseMapping *mapping,
                        const Array<Buffer> &private_allocations,
                        const ValueInfoMap &storage_values) {
  for (const Buffer &buffer : private_allocations)
    mapping->vector_storage.insert(buffer->data.get());
  for (const auto &[data, info] : storage_values)
    if (info.IsVmi())
      mapping->vector_storage.insert(data.get());
}

MemoryAccess AnalyzeBufferAccess(const Buffer &buffer,
                                 const Array<PrimExpr> &indices, bool is_write,
                                 const Var &loop_var,
                                 const PtoInverseMapping &mapping,
                                 const Map<Var, PrimExpr> &bind_env) {
  return AnalyzePtoAddress(buffer, indices, is_write, loop_var, mapping,
                           bind_env);
}

namespace {

Optional<PrimExpr> IntegerWideningInput(const PrimExpr &expr) {
  const auto *conversion = expr.as<CastNode>();
  if (conversion && conversion->annotations.empty() &&
      IsPtoIndexDType(expr.dtype()) &&
      IsPtoIndexDType(conversion->value.dtype()) &&
      conversion->value.dtype().bits() < expr.dtype().bits()) {
    return conversion->value;
  }
  return {};
}

// For a complete offset, only remove the promotion introduced by ElemOffset.
// The source's explicit width determines how the final offset is interpreted.
PrimExpr StripIndexWidening(const MemoryAccess &access,
                            const PrimExpr &offset) {
  PrimExpr native = offset;
  while (true) {
    for (const PrimExpr &index : access.source_indices) {
      if (ExprDeepEqual()(native, index)) {
        return native;
      }
    }
    auto input = IntegerWideningInput(native);
    if (!input) {
      return offset;
    }
    native = *input;
  }
}

void BindPtoExtent(arith::Analyzer *proof, const Var &var, int64_t extent) {
  proof->Bind(var, Range::FromMinExtent(make_zero(var.dtype()),
                                        IntImm(var.dtype(), extent)));
}

// Inactive tail lanes may produce arbitrary offsets. Try the rectangular
// domain first, then check the full chunks and the tail separately if needed.
bool CheckActivePtoLanes(arith::Analyzer *proof,
                         const PtoInverseMapping &mapping,
                         const std::function<bool()> &check) {
  if (check()) {
    return true;
  }
  if (mapping.tail_lanes == 0) {
    return false;
  }
  const Var &chunk = mapping.chunk_var;
  PrimExpr last_chunk = IntImm(chunk.dtype(), mapping.chunk_count - 1);
  if (mapping.chunk_count > 1) {
    With<arith::ConstraintContext> full(proof, chunk < last_chunk);
    if (!check()) {
      return false;
    }
  }
  const Var &lane = mapping.lane_var;
  With<arith::ConstraintContext> tail(proof, chunk == last_chunk);
  With<arith::ConstraintContext> active(
      proof, lane < IntImm(lane.dtype(), mapping.tail_lanes));
  return check();
}

// The bound analyzer intersects a Cast's input interval with its destination
// range and treats integer arithmetic as unbounded. In proofs, model values
// that may wrap with their full result type. A load's coordinates do not
// constrain the loaded value.
class PtoAddressRangeExpr : public ExprMutator {
public:
  explicit PtoAddressRangeExpr(arith::Analyzer *proof,
                               const Map<Var, PrimExpr> &bind_env = {})
      : proof_(proof), bind_env_(bind_env) {}

private:
  PrimExpr VisitExpr(const PrimExpr &expr) final {
    if (expr.as<BufferLoadNode>()) {
      return expr;
    }
    if (auto var = expr.as<Var>()) {
      return VisitExpr_(var->get());
    }
    PrimExpr result = ExprMutator::VisitExpr(expr);
    DataType dtype = result.dtype();
    if (!IsPtoIndexDType(dtype)) {
      return result;
    }
    const auto *conversion = result.as<CastNode>();
    auto bound =
        proof_->const_int_bound(conversion ? conversion->value : result);
    if (bound->min_value >= *as_const_int(min_value(dtype)) &&
        bound->max_value <= *as_const_int(max_value(dtype))) {
      return result;
    }
    return Var("offset_range", dtype);
  }

  PrimExpr VisitExpr_(const VarNode *op) final {
    Var var = GetRef<Var>(op);
    if (auto mapped = values_.Get(var)) {
      return *mapped;
    }
    if (!bind_env_.count(var) || visiting_.size() >= 64 ||
        !visiting_.insert(var).second) {
      return var;
    }
    PrimExpr value = VisitExpr(bind_env_[var]);
    visiting_.erase(var);
    values_.Set(var, value);
    return value;
  }

  arith::Analyzer *proof_;
  Map<Var, PrimExpr> bind_env_;
  Map<Var, PrimExpr> values_;
  std::unordered_set<Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> visiting_;
};

// Keep loads and saved scalar/vector values opaque while splitting the
// address. Restore them at their original coordinates before lowering.
class SplitAddressExpr : public ExprMutator {
public:
  SplitAddressExpr(const Var &loop_var, const PtoInverseMapping &mapping,
                   const Map<Var, PrimExpr> &bind_env,
                   arith::Analyzer *cast_proof)
      : loop_var_(loop_var), mapping_(mapping), bind_env_(bind_env),
        cast_proof_(cast_proof) {}

  PrimExpr Restore(const PrimExpr &expr) const {
    return Substitute(expr, values_);
  }

  PrimExpr AtLane(const PrimExpr &expr, const PrimExpr &lane,
                  bool keep_uniform_values) const {
    Map<Var, PrimExpr> substitutions{{mapping_.lane_var, lane}};
    for (const auto &[symbol, value] : values_) {
      if (!keep_uniform_values ||
          !IsPtoLaneUniform(value, loop_var_, mapping_, bind_env_)) {
        substitutions.Set(symbol, make_zero(value.dtype()));
      }
    }
    return Substitute(expr, substitutions);
  }

  bool HasCast() const { return has_cast_; }

private:
  PrimExpr VisitExpr(const PrimExpr &expr) final {
    if (auto var = expr.as<Var>()) {
      if (var->same_as(loop_var_)) {
        return mapping_.index_expr;
      }
      return bind_env_.count(*var) ? Save_(expr) : expr;
    }
    if (expr.as<IntImmNode>()) {
      return expr;
    }
    if (expr.as<AddNode>() || expr.as<SubNode>() || expr.as<MulNode>() ||
        expr.as<CastNode>()) {
      return ExprMutator::VisitExpr(expr);
    }
    return Save_(expr);
  }

  PrimExpr VisitExpr_(const CastNode *op) final {
    has_cast_ = true;
    // Use signed coordinates for decomposition only when the source Cast
    // preserves the value on every active lane.
    if (cast_proof_ && op->annotations.empty() && IsPtoIndexDType(op->dtype) &&
        IsPtoIndexDType(op->value.dtype()) && FitsType_(op->value, op->dtype)) {
      PrimExpr value = VisitExpr(op->value);
      if (value.dtype() == DataType::Int(32)) {
        return value;
      }
    }
    return Save_(GetRef<PrimExpr>(op));
  }

  PrimExpr VisitExpr_(const AddNode *op) final { return SplitBinary_<Add>(op); }
  PrimExpr VisitExpr_(const SubNode *op) final { return SplitBinary_<Sub>(op); }
  PrimExpr VisitExpr_(const MulNode *op) final { return SplitBinary_<Mul>(op); }

  template <typename Expr, typename Node>
  PrimExpr SplitBinary_(const Node *op) {
    PrimExpr a = VisitExpr(op->a);
    PrimExpr b = VisitExpr(op->b);
    DataType dtype = op->dtype;
    const DataType signed_index = DataType::Int(32);
    if (cast_proof_ && dtype == DataType::UInt(32) &&
        (a.dtype() == signed_index || b.dtype() == signed_index) &&
        FitsType_(op->a, signed_index) && FitsType_(op->b, signed_index) &&
        FitsType_(GetRef<PrimExpr>(op), signed_index)) {
      dtype = signed_index;
    }
    return Expr(cast(dtype, a), cast(dtype, b), op->span);
  }

  bool FitsType_(const PrimExpr &expr, DataType dtype) const {
    return CheckActivePtoLanes(cast_proof_, mapping_, [&] {
      PrimExpr value = PtoAddressRangeExpr(cast_proof_)(expr);
      auto bound = cast_proof_->const_int_bound(value);
      return bound->min_value >= *as_const_int(min_value(dtype)) &&
             bound->max_value <= *as_const_int(max_value(dtype));
    });
  }

  PrimExpr Save_(const PrimExpr &expr) {
    Var symbol("offset_value", expr.dtype());
    values_.Set(symbol, expr);
    return symbol;
  }

  const Var &loop_var_;
  const PtoInverseMapping &mapping_;
  const Map<Var, PrimExpr> &bind_env_;
  arith::Analyzer *cast_proof_;
  bool has_cast_ = false;
  Map<Var, PrimExpr> values_;
};

// Only coordinates and their integer arithmetic can be generated at a narrower
// width without converting an already loaded vector. Check every intermediate
// on active iterations. The indexed memory operation masks off inactive lanes,
// so integer wrap in their offsets cannot affect an access.
bool CanGenerateUint16Offsets(const PrimExpr &offset,
                              const PtoInverseMapping &mapping,
                              arith::Analyzer *proof,
                              const Map<Var, PrimExpr> &bind_env) {
  std::function<bool(const PrimExpr &)> check = [&](const PrimExpr &expr) {
    if (expr.dtype() != DataType::Int(32) &&
        expr.dtype() != DataType::UInt(32)) {
      return false;
    }
    if (const auto *add = expr.as<AddNode>()) {
      if (!check(add->a) || !check(add->b)) {
        return false;
      }
    } else if (const auto *sub = expr.as<SubNode>()) {
      if (!check(sub->a) || !check(sub->b)) {
        return false;
      }
    } else if (const auto *mul = expr.as<MulNode>()) {
      if (!check(mul->a) || !check(mul->b)) {
        return false;
      }
    } else if (auto var = expr.as<Var>()) {
      if (bind_env.count(*var)) {
        return false;
      }
    } else if (const auto *select = expr.as<SelectNode>()) {
      // Only the generated last-lane Select uses the private chunk variable
      // here. It is computed as a scalar before converting its result.
      const auto *cond = select->condition.as<LTNode>();
      if (!cond || !cond->a.same_as(mapping.chunk_var) ||
          !cond->b.as<IntImmNode>()) {
        return false;
      }
    } else if (!expr.as<IntImmNode>()) {
      return false;
    }
    auto bound = proof->const_int_bound(proof->rewrite_simplify(expr));
    return bound->min_value >= 0 && bound->max_value <= 65535;
  };
  return CheckActivePtoLanes(proof, mapping, [&] { return check(offset); });
}

using PtoOffsetPlanResult = std::variant<PtoIndexedAccessPlan, std::string>;

PtoOffsetPlanResult
TryPlanPtoOffsets(const MemoryAccess &access, const PrimExpr &offset,
                  const Var &loop_var, const PtoInverseMapping &mapping,
                  arith::Analyzer *analyzer, const Map<Var, PrimExpr> &bind_env,
                  const PrimExpr &descriptor_extent,
                  const Optional<PrimExpr> &version = {}) {
  const int offset_bits = access.buffer->dtype.bits() == 32 ? 32 : 16;
  std::unique_ptr<arith::Analyzer> proof;
  auto offset_proof = [&] {
    if (!proof) {
      proof = analyzer->Clone();
      BindPtoExtent(proof.get(), loop_var, mapping.logical_extent);
      BindPtoExtent(proof.get(), mapping.chunk_var, mapping.chunk_count);
      BindPtoExtent(proof.get(), mapping.lane_var, mapping.lanes);
      if (auto serial = mapping.SerialVar()) {
        BindPtoExtent(proof.get(), *serial, mapping.SerialExtent());
      }
    }
    return proof.get();
  };
  auto carrier = [&](const PrimExpr &relative, bool rebased) {
    PrimExpr native = rebased
                          ? IntegerWideningInput(relative).value_or(relative)
                          : StripIndexWidening(access, relative);
    // Once the base is separate, the carrier must preserve the numeric value
    // of the remaining address, including when the widening was explicit.
    if ((rebased || version.defined()) && !native.same_as(relative) &&
        native.dtype().is_int()) {
      auto *proof = offset_proof();
      if (!CheckActivePtoLanes(proof, mapping, [&] {
            PrimExpr value = PtoAddressRangeExpr(proof)(native);
            return proof->const_int_bound(value)->min_value >= 0;
          })) {
        return relative;
      }
    }
    return native;
  };
  const Var &lane = mapping.lane_var;
  std::string base_error;
  auto valid_rebase = [&](const PrimExpr &base,
                          const PrimExpr &relative_offsets) {
    auto *range_proof = offset_proof();
    // D and the relative offsets add without the source integer's wrapping
    // semantics. Prove their sum in a wider type.
    PrimExpr sum = WidenToInt64(base) + WidenToInt64(relative_offsets);
    const int64_t max_offset = *as_const_int(max_value(offset.dtype()));
    if (!CheckActivePtoLanes(range_proof, mapping, [&] {
          PrimExpr value = PtoAddressRangeExpr(range_proof, bind_env)(sum);
          auto bound = range_proof->const_int_bound(
              range_proof->rewrite_simplify(value));
          return bound->min_value >= 0 && bound->max_value <= max_offset;
        })) {
      return false;
    }
    PrimExpr mapped_base =
        SubstituteVar(base, loop_var.get(), mapping.index_expr);
    mapped_base =
        SubstituteVar(mapped_base, lane.get(), make_zero(lane.dtype()));
    if (CheckActivePtoLanes(range_proof, mapping, [&] {
          PrimExpr d = range_proof->rewrite_simplify(
              PtoAddressRangeExpr(range_proof)(mapped_base));
          return range_proof->CanProve(d >= 0) &&
                 range_proof->CanProve(d < descriptor_extent);
        })) {
      return true;
    }
    std::ostringstream oss;
    oss << "B16 indexed access requires the scalar instruction base D to be "
           "provably inside the owner (0 <= D < "
        << descriptor_extent << "), but the rebased access produces base "
        << range_proof->rewrite_simplify(mapped_base);
    base_error = oss.str();
    return false;
  };

  // Retain the first source-width conversion as a fallback while looking for
  // a native carrier or directly generated uint16 coordinates.
  std::optional<PtoIndexedAccessPlan> narrowed_plan;
  std::string signed_cast_error;
  std::string encoding_error;
  auto try_candidate =
      [&](const PrimExpr &base,
          const PrimExpr &offsets) -> std::optional<PtoIndexedAccessPlan> {
    if (!IsPtoIndexDType(offsets.dtype())) {
      encoding_error = "indexed element offsets require scalar 8-bit, "
                       "16-bit or 32-bit integer expressions";
      return std::nullopt;
    }
    PtoIndexedAccessPlan plan{base, offsets, version, descriptor_extent};
    const bool needs_narrowing = offsets.dtype().bits() > offset_bits;
    plan.generate_uint16_coordinates =
        needs_narrowing &&
        CanGenerateUint16Offsets(offsets, mapping, offset_proof(), bind_env);
    const bool needs_cast =
        needs_narrowing && !plan.generate_uint16_coordinates;
    if (needs_cast && narrowed_plan) {
      return std::nullopt;
    }
    Array<PrimExpr> signed_narrowing_inputs;
    if (!plan.generate_uint16_coordinates) {
      try {
        signed_narrowing_inputs =
            CheckPtoIndexedOffsetExpr(offsets, loop_var, mapping, bind_env);
      } catch (const PtoAnalysisError &err) {
        encoding_error = err.what();
        return std::nullopt;
      }
    }
    if (offset_bits == 16 && !signed_narrowing_inputs.empty()) {
      auto *range_proof = offset_proof();
      // A dense signed int32-to-int16 conversion can saturate even with
      // NOSAT. Check its input before any later operation bounds the offset.
      if (!CheckActivePtoLanes(range_proof, mapping, [&] {
            PtoAddressRangeExpr range(range_proof, bind_env);
            for (const PrimExpr &input : signed_narrowing_inputs) {
              auto bound = range_proof->const_int_bound(range(input));
              if (bound->min_value < std::numeric_limits<int16_t>::min() ||
                  bound->max_value > std::numeric_limits<int16_t>::max()) {
                return false;
              }
            }
            return true;
          })) {
        signed_cast_error =
            "B16 indexed access cannot prove the input of a vector "
            "int32-to-int16 cast fits in int16; bound it before narrowing";
        return std::nullopt;
      }
    }
    if (needs_cast) {
      auto *range_proof = offset_proof();
      if (!CheckActivePtoLanes(range_proof, mapping, [&] {
            PrimExpr value =
                PtoAddressRangeExpr(range_proof, bind_env)(offsets);
            auto bound = range_proof->const_int_bound(value);
            return bound->min_value >= 0 && bound->max_value <= 65535;
          })) {
        return std::nullopt;
      }
    }
    if (!is_zero(base) && !valid_rebase(base, offsets)) {
      return std::nullopt;
    }
    if (needs_cast) {
      plan.offsets = Cast(DataType::UInt(16), offsets);
      narrowed_plan = std::move(plan);
      return std::nullopt;
    }
    return plan;
  };

  PrimExpr offsets = carrier(offset, /*rebased=*/false);
  if (auto plan = try_candidate(make_zero(DataType::Int(32)), offsets)) {
    return *plan;
  }

  auto try_split =
      [&](SplitAddressExpr &split) -> std::optional<PtoIndexedAccessPlan> {
    arith::Analyzer expressions;
    const Var &chunk = mapping.chunk_var;
    BindPtoExtent(&expressions, lane, mapping.lanes);
    BindPtoExtent(&expressions, chunk, mapping.chunk_count);
    PrimExpr mapped = expressions.Simplify(split(offset));

    PrimExpr base_lane = make_zero(lane.dtype());
    auto affine = arith::DetectLinearEquation(mapped, {lane});
    if (!affine.empty() && expressions.CanProve(affine[0] < 0)) {
      base_lane = IntImm(lane.dtype(), mapping.lanes - 1);
      if (mapping.tail_lanes != 0) {
        base_lane = expressions.Simplify(
            Select(chunk < IntImm(chunk.dtype(), mapping.chunk_count - 1),
                   base_lane, IntImm(lane.dtype(), mapping.tail_lanes - 1)));
      }
    }

    PrimExpr previous_base = make_zero(mapped.dtype());
    for (bool keep_uniform_values : {false, true}) {
      PrimExpr split_base = expressions.Simplify(
          split.AtLane(mapped, base_lane, keep_uniform_values));
      if (is_zero(split_base) || ExprDeepEqual()(split_base, previous_base)) {
        continue;
      }
      previous_base = split_base;
      PrimExpr relative =
          split.Restore(expressions.Simplify(mapped - split_base));
      PrimExpr relative_offsets = carrier(relative, /*rebased=*/true);
      PrimExpr base = split.Restore(split_base);
      if (auto plan = try_candidate(base, relative_offsets)) {
        return plan;
      }
    }
    return std::nullopt;
  };
  if (offset_bits == 16 && offset.dtype().bits() == 32) {
    SplitAddressExpr split(loop_var, mapping, bind_env, nullptr);
    if (auto plan = try_split(split)) {
      return *plan;
    }
    if (split.HasCast()) {
      SplitAddressExpr proved_split(loop_var, mapping, bind_env,
                                    offset_proof());
      if (auto plan = try_split(proved_split)) {
        return *plan;
      }
    }
  }
  if (narrowed_plan) {
    return *narrowed_plan;
  }
  if (!signed_cast_error.empty()) {
    return signed_cast_error;
  }
  if (!base_error.empty()) {
    return base_error;
  }
  if (!encoding_error.empty()) {
    return encoding_error;
  }
  std::ostringstream oss;
  if (offset_bits == 16 && !analyzer->CanProve(descriptor_extent <= 65536)) {
    oss << "B16 indexed access requires a ui16 relative window: ";
  }
  oss << "indexed " << offsets.dtype().bits()
      << "-bit offsets require implicit narrowing to " << offset_bits
      << " bits; use a native 16-bit index or an explicit source Cast only "
         "after ensuring the complete element offset is representable";
  return oss.str();
}

bool HasLeadingVersionAxis(const Buffer &buffer, int versions,
                           arith::Analyzer *analyzer) {
  return buffer->shape.size() > 1 &&
         analyzer->CanProveEqual(buffer->shape[0], versions);
}

bool FitsPtoVersionSlot(const Buffer &view, const Array<PrimExpr> &strides,
                        const PrimExpr &origin, const PrimExpr &slot_extent,
                        arith::Analyzer *analyzer) {
  // Bound the whole trailing view because an active offset can come from a
  // loaded index. Check each product against the remaining space before
  // forming it, so a large extent or stride cannot wrap the proof.
  PrimExpr one = make_const(slot_extent.dtype(), 1);
  PrimExpr remaining = slot_extent - origin - one;
  for (size_t axis = 1; axis < view->shape.size(); ++axis) {
    PrimExpr extent = cast(slot_extent.dtype(), view->shape[axis]);
    PrimExpr stride = cast(slot_extent.dtype(), strides[axis]);
    if (analyzer->CanProveEqual(stride, 0)) {
      continue;
    }
    if (!analyzer->CanProve(extent > 0) || !analyzer->CanProve(stride >= 0) ||
        !analyzer->CanProve(extent - one <=
                            floordiv(remaining, Max(stride, one)))) {
      return false;
    }
    remaining = analyzer->Simplify(remaining - (extent - one) * stride);
  }
  return true;
}

PtoOffsetPlanResult TryPlanPtoVersionSlot(
    const MemoryAccess &access, const Var &loop_var,
    const PtoInverseMapping &mapping, arith::Analyzer *analyzer,
    const Map<Var, PrimExpr> &bind_env, int versions,
    const PrimExpr &slot_extent, arith::Analyzer &expr_analyzer) {
  Array<PrimExpr> strides = PtoBufferStrides(access.buffer, analyzer);
  if (!HasLeadingVersionAxis(access.buffer, versions, analyzer) ||
      !analyzer->CanProveEqual(cast(slot_extent.dtype(), strides[0]),
                               slot_extent)) {
    return "indexed access to padded versioned storage requires a leading "
           "version axis";
  }
  const PrimExpr &version = access.source_indices[0];
  if (!IsPtoLaneUniform(version, loop_var, mapping, bind_env)) {
    return "indexed access to padded versioned storage requires a "
           "lane-uniform version selector";
  }

  // A static view origin can move the first logical row to a later slot.
  // Apply that shift to the descriptor, where version alignment is handled.
  const PrimExpr &view_origin = access.buffer->elem_offset;
  const int64_t *origin = as_const_int(view_origin);
  if (origin == nullptr || *origin < 0) {
    return "indexed access to padded versioned storage requires a constant "
           "non-negative view origin";
  }
  PrimExpr wide_origin = make_const(slot_extent.dtype(), *origin);
  PrimExpr slot_shift =
      expr_analyzer.Simplify(floordiv(wide_origin, slot_extent));
  PrimExpr within_slot =
      expr_analyzer.Simplify(floormod(wide_origin, slot_extent));
  if (!FitsPtoVersionSlot(access.buffer, strides, within_slot, slot_extent,
                          analyzer)) {
    return "indexed access to padded versioned storage requires the view to "
           "fit in a single version slot";
  }

  PrimExpr selected_version = version;
  Buffer slot_view = access.buffer;
  if (!is_zero(slot_shift)) {
    selected_version =
        expr_analyzer.Simplify(cast(slot_extent.dtype(), version) + slot_shift);
    auto proof = analyzer->Clone();
    BindPtoExtent(proof.get(), loop_var, mapping.logical_extent);
    if (auto serial = mapping.SerialVar()) {
      BindPtoExtent(proof.get(), *serial, mapping.SerialExtent());
    }
    if (!proof->CanProve(selected_version >= 0) ||
        !proof->CanProve(selected_version < versions)) {
      return "indexed view selects a version outside the padded allocation";
    }
    slot_view.CopyOnWrite()->elem_offset =
        cast(view_origin.dtype(), within_slot);
  }
  Array<PrimExpr> slot_indices = access.source_indices;
  slot_indices.Set(0, make_zero(version.dtype()));
  PrimExpr slot_offset = slot_view->ElemOffset(slot_indices)[0];
  return TryPlanPtoOffsets(access, slot_offset, loop_var, mapping, analyzer,
                           bind_env, slot_extent, selected_version);
}

} // namespace

PtoIndexedAccessPlan PlanPtoIndexedAccess(const MemoryAccess &access,
                                          const IndexedMemoryOwner &owner,
                                          const Var &loop_var,
                                          const PtoInverseMapping &mapping,
                                          arith::Analyzer *analyzer,
                                          const Map<Var, int> &buffer_versions,
                                          const Map<Var, PrimExpr> &bind_env) {
  if (!IsSupportedAddressExpr(access.original_offset)) {
    throw PtoAnalysisError(
        "complete indexed offset contains an unsupported call or index load");
  }
  auto versions = buffer_versions.Get(owner.buffer->data);
  if (versions && *versions > 1 &&
      HasLeadingVersionAxis(owner.buffer, *versions, analyzer)) {
    arith::Analyzer expr_analyzer;
    PrimExpr slot_extent =
        expr_analyzer.Simplify(floordiv(owner.extent, *versions));
    int elements_per_alignment = 256 / owner.buffer->dtype.bits();
    const bool padded = !analyzer->CanProveEqual(
        floormod(slot_extent, elements_per_alignment), 0);
    // Padding requires a slot descriptor. For aligned B8/B16, try a slot
    // first to preserve narrow offsets and fall back to the whole owner.
    if (padded || access.buffer->dtype.bits() != 32) {
      auto result =
          TryPlanPtoVersionSlot(access, loop_var, mapping, analyzer, bind_env,
                                *versions, slot_extent, expr_analyzer);
      if (auto *plan = std::get_if<PtoIndexedAccessPlan>(&result)) {
        return std::move(*plan);
      }
      if (padded) {
        throw PtoAnalysisError(std::get<std::string>(result));
      }
    }
  }
  auto result = TryPlanPtoOffsets(access, access.original_offset, loop_var,
                                  mapping, analyzer, bind_env, owner.extent);
  if (auto *plan = std::get_if<PtoIndexedAccessPlan>(&result)) {
    return std::move(*plan);
  }
  throw PtoAnalysisError(std::get<std::string>(result));
}

void CheckBroadcastDType(DataType dtype, const std::string &context) {
  bool supported =
      dtype.lanes() == 1 &&
      (((dtype.is_int() || dtype.is_uint()) &&
        (dtype.bits() == 8 || dtype.bits() == 16 || dtype.bits() == 32)) ||
       dtype == DataType::Float(16) || dtype == DataType::Float(32) ||
       dtype == DataType::BFloat(16));
  ICHECK(supported)
      << "PTO Parallel unsupported scalar-to-vector broadcast dtype " << dtype
      << " for " << context;
}

bool IsSupportedElementDType(DataType dtype) {
  // No packed sub-byte element types (FP4 etc.), no
  // buffer-element dtypes that already carry vector lanes, and no FP64
  // (the current PTO codegen has no FP64 scalar-type mapping).
  if (dtype.bits() < 8 || dtype.lanes() != 1) {
    // FP4 (e2m1) and every other sub-byte element type land here: an FP4
    // element is 4 bits wide whether packed or not, so a separate FP4 check
    // would be unreachable.
    return false;
  }
  // The PTO codegen only maps 8/16/32-bit scalars; unusual widths (int24) and
  // FP64 (no scalar-type mapping) are rejected for buffers and arithmetic
  // alike by the width rule below, not just inside Cast planning.
  if (!dtype.is_bool() && dtype.bits() != 8 && dtype.bits() != 16 &&
      dtype.bits() != 32) {
    return false;
  }
  return dtype.is_float() || dtype.is_bfloat16() || dtype.is_int() ||
         dtype.is_uint() || dtype.is_bool() || IsSupportedFloat8(dtype);
}

bool IsSupportedFloat8(DataType dtype) {
  // Only these two take part in the FP8 conversion matrix. float8_e4m3 is a
  // distinct TIR dtype from float8_e4m3fn and is not admitted by name
  // similarity; E8M0/HiF8/FNUZ are outside the supported conversion range,
  // while FP4 is a packed sub-byte type with no unified element addressing.
  return dtype.is_float8_e4m3fn() || dtype.is_float8_e5m2();
}

PtoBitOp GetPtoBitOp(const Call &call) {
  if (!call->dtype.is_bool() && call->op.same_as(builtin::bitwise_and())) {
    return PtoBitOp::kAnd;
  }
  if (!call->dtype.is_bool() && call->op.same_as(builtin::bitwise_or())) {
    return PtoBitOp::kOr;
  }
  if (!call->dtype.is_bool() && call->op.same_as(builtin::bitwise_xor())) {
    return PtoBitOp::kXor;
  }
  if (!call->dtype.is_bool() && call->op.same_as(builtin::bitwise_not())) {
    return PtoBitOp::kNot;
  }
  if (call->op.same_as(builtin::shift_left())) {
    return PtoBitOp::kShiftLeft;
  }
  if (call->op.same_as(builtin::shift_right())) {
    return PtoBitOp::kShiftRight;
  }
  if (call->op.same_as(builtin::reinterpret())) {
    return PtoBitOp::kReinterpret;
  }
  return PtoBitOp::kNone;
}

static void CheckPtoBitOp(const Call &call, PtoBitOp op) {
  ICHECK(op != PtoBitOp::kNone);
  const bool unary = op == PtoBitOp::kNot || op == PtoBitOp::kReinterpret;
  const std::string name = Downcast<Op>(call->op)->name;
  if (call->args.size() != (unary ? 1U : 2U) || !call->annotations.empty()) {
    throw PtoAnalysisError(name + " expects " +
                           (unary ? "one operand" : "two operands") +
                           " and no annotations");
  }

  const DataType from = call->args[0].dtype();
  const DataType to = call->dtype;
  if (op == PtoBitOp::kReinterpret) {
    const DataType u32 = DataType::UInt(32);
    const DataType f32 = DataType::Float(32);
    const bool supported = (IsPtoIndexDType(from) && IsPtoIndexDType(to) &&
                            from.bits() == to.bits()) ||
                           (from == f32 && (to == f32 || to == u32)) ||
                           (from == u32 && to == f32);
    if (!supported) {
      std::ostringstream oss;
      oss << name << " does not support " << from << " -> " << to
          << "; expected same-width signed/unsigned 8-, 16- or 32-bit "
             "integers, uint32 <-> float32, or a supported identity";
      throw PtoAnalysisError(oss.str());
    }
    return;
  }

  const DataType rhs = unary ? from : call->args[1].dtype();
  const bool shift = op == PtoBitOp::kShiftLeft || op == PtoBitOp::kShiftRight;
  const int64_t *count =
      shift && rhs.is_scalar() && (rhs.is_int() || rhs.is_uint())
          ? as_const_int(call->args[1])
          : nullptr;
  if (!IsPtoIndexDType(to) || from != to || (rhs != to && !count)) {
    std::ostringstream oss;
    oss << name
        << " requires scalar 8-, 16- or 32-bit integer data operands "
           "and result of the same dtype, got "
        << from;
    if (!unary) {
      oss << ", " << rhs;
    }
    oss << " -> " << to;
    throw PtoAnalysisError(oss.str());
  }
  if (count && (*count < 0 || *count >= to.bits())) {
    std::ostringstream oss;
    oss << name << " on " << to << " requires a shift count in [0, "
        << to.bits() << "), got " << call->args[1];
    throw PtoAnalysisError(oss.str());
  }
}

namespace {

void CheckPtoMaskCall(const Call &call, std::string_view mask_op) {
  const std::string name = Downcast<Op>(call->op)->name;
  const size_t arity = mask_op == "vnot" ? 1U : 2U;
  if (call->args.size() != arity) {
    throw PtoAnalysisError(name + " has the wrong operand count");
  }
  for (const PrimExpr &arg : call->args) {
    if (arg.dtype() != call->dtype || !arg.dtype().is_scalar()) {
      throw PtoAnalysisError(
          name + " requires matching bool scalar operands and result");
    }
  }
}

// Whitelist check for *value* expressions. A
// BufferLoad's indices belong to address analysis, not value lowering
// the version-dim FloorMod/FloorDiv inside a load index must not
// be rejected here. The value whitelist does not descend into indices.
class WhitelistVisitor : public ExprVisitor {
public:
  std::optional<std::string> unsupported;

  void VisitExpr_(const BufferLoadNode *) final {}
  void VisitExpr_(const VarNode *) final {}
  void VisitExpr_(const IntImmNode *) final {}
  void VisitExpr_(const FloatImmNode *) final {}
  void VisitExpr_(const StringImmNode *) final {
    // String constants are metadata markers only. The base
    // ExprVisitor declares StringImm and accepts it as a leaf, so the
    // VisitExprDefault_ fallback never fired and `Evaluate(StringImm)`
    // slipped through Verify into Vectorize's internal error. Reject it
    // explicitly here, with the same wording as the lane-use analysis.
    if (!unsupported.has_value()) {
      unsupported = "string constants are metadata-only; they cannot "
                    "appear in computation values";
    }
  }
  void VisitExpr_(const AddNode *op) final { ExprVisitor::VisitExpr_(op); }
  void VisitExpr_(const SubNode *op) final { ExprVisitor::VisitExpr_(op); }
  void VisitExpr_(const MulNode *op) final { ExprVisitor::VisitExpr_(op); }
  void VisitExpr_(const DivNode *op) final { ExprVisitor::VisitExpr_(op); }
  void VisitExpr_(const CastNode *op) final {
    // Only the converted value is a computation value. The base visitor also
    // walks PrimExpr entries of the cast annotations, which would put a
    // StringImm rounding hint in front of the value whitelist and reject an
    // otherwise legal cast ("string constants are metadata-only"). Attribute
    // legality is decided by PlanValueCast, from the raw annotations.
    VisitExpr(op->value);
  }
  // The base ExprVisitor overrides VisitExpr_ for every standard node and
  // silently descends, so non-whitelist nodes never reach
  // VisitExprDefault_. Reject them explicitly.
  void MarkUnsupported(const char *kind) {
    if (!unsupported.has_value()) {
      unsupported = std::string("unsupported expression node `") + kind +
                    "` (admits buffer loads, constants, scalars, loop "
                    "indices, Add, Sub, Mul, Div, FloorDiv, FloorMod, "
                    "fabs/exp/log/sqrt, Cast, "
                    "Select, comparisons, Min/Max, And/Or/Not and the "
                    "bitwise mask-combine calls)";
    }
  }
  void VisitExpr_(const ModNode *) final { MarkUnsupported("Mod"); }
  // Min/Max are admitted here — Verify runs before
  // LegalizeParallelToPTO, which rewrites them into the Select form before
  // VectorizeParallelToPTO consumes the unit.
  void VisitExpr_(const MinNode *op) final { ExprVisitor::VisitExpr_(op); }
  void VisitExpr_(const MaxNode *op) final { ExprVisitor::VisitExpr_(op); }
  // Select + relational ops (the shape Legalize produces and Vectorize
  // lowers to vcmp/vcmps + vsel).
  void VisitExpr_(const EQNode *op) final { ExprVisitor::VisitExpr_(op); }
  void VisitExpr_(const NENode *op) final { ExprVisitor::VisitExpr_(op); }
  void VisitExpr_(const LTNode *op) final { ExprVisitor::VisitExpr_(op); }
  void VisitExpr_(const LENode *op) final { ExprVisitor::VisitExpr_(op); }
  void VisitExpr_(const GTNode *op) final { ExprVisitor::VisitExpr_(op); }
  void VisitExpr_(const GENode *op) final { ExprVisitor::VisitExpr_(op); }
  void VisitExpr_(const AndNode *op) final { ExprVisitor::VisitExpr_(op); }
  void VisitExpr_(const OrNode *op) final { ExprVisitor::VisitExpr_(op); }
  void VisitExpr_(const NotNode *op) final { ExprVisitor::VisitExpr_(op); }
  void VisitExpr_(const SelectNode *op) final { ExprVisitor::VisitExpr_(op); }
  void VisitExpr_(const RampNode *) final { MarkUnsupported("Ramp"); }
  void VisitExpr_(const BroadcastNode *) final { MarkUnsupported("Broadcast"); }
  void VisitExpr_(const ShuffleNode *) final { MarkUnsupported("Shuffle"); }
  void VisitExpr_(const ReduceNode *) final { MarkUnsupported("Reduce"); }
  void VisitExpr_(const CallNode *op) final {
    if (const auto *call_op = op->op.as<OpNode>()) {
      const std::string &name = call_op->name;
      // The infinity sentinel is lane-uniform and Legalize folds it.
      if (name == "tl.infinity" || name == "tir.infinity") {
        return;
      }
    }
    Call call = GetRef<Call>(op);
    try {
      if (const char *mask_op = GetPtoMaskLogicOp(call)) {
        CheckPtoMaskCall(call, mask_op);
        ExprVisitor::VisitExpr_(op);
        return;
      }
      if (CheckPtoValueCall(call)) {
        ExprVisitor::VisitExpr_(op);
        return;
      }
    } catch (const PtoAnalysisError &err) {
      if (!unsupported) {
        unsupported = err.what();
      }
      return;
    }
    {
      std::string name = "call";
      if (const auto *op_node = op->op.as<OpNode>()) {
        name = op_node->name;
      }
      if (!unsupported.has_value()) {
        unsupported =
            "unsupported Call `" + name +
            "` (expected a supported math intrinsic or bit operation)";
      }
    }
  }

  void VisitExprDefault_(const ffi::Object *op) final {
    std::string key = op->GetTypeKey();
    MarkUnsupported(key.c_str());
  }
};

} // namespace

// ---------------------------------------------------------------------------
// Shared SIMD_VF region scan: one implementation for both passes.
// ---------------------------------------------------------------------------
bool UsesExplicitSIMD(const Stmt &stmt) {
  bool found = false;
  PostOrderVisit(stmt, [&](const ObjectRef &node) {
    if (const auto *call = node.as<CallNode>()) {
      if (const auto *op = call->op.as<OpNode>()) {
        found |= std::string(op->name).rfind("tl.simd.", 0) == 0;
      }
    }
  });
  return found;
}

namespace {

void CollectAllParallelDeep(const Stmt &stmt,
                            std::vector<const ForNode *> &out) {
  if (const auto *for_node = stmt.as<ForNode>()) {
    if (for_node->kind == ForKind::kParallel) {
      if (UsesExplicitSIMD(for_node->body)) {
        // The explicit nest itself needs no conversion, but an automatic
        // Parallel below it must still be reported as unsupported.
        CollectAllParallelDeep(for_node->body, out);
        return;
      }
      out.push_back(for_node);
    }
    CollectAllParallelDeep(for_node->body, out);
    return;
  }
  if (const auto *seq = stmt.as<SeqStmtNode>()) {
    for (const auto &s : seq->seq) {
      CollectAllParallelDeep(s, out);
    }
    return;
  }
  if (const auto *attr = stmt.as<AttrStmtNode>()) {
    CollectAllParallelDeep(attr->body, out);
    return;
  }
  if (const auto *block = stmt.as<SBlockNode>()) {
    CollectAllParallelDeep(block->body, out);
    return;
  }
  if (const auto *realize = stmt.as<SBlockRealizeNode>()) {
    CollectAllParallelDeep(realize->block->body, out);
    return;
  }
  if (const auto *ite = stmt.as<IfThenElseNode>()) {
    CollectAllParallelDeep(ite->then_case, out);
    if (ite->else_case.defined()) {
      CollectAllParallelDeep(ite->else_case.value(), out);
    }
    return;
  }
  if (const auto *while_node = stmt.as<WhileNode>()) {
    CollectAllParallelDeep(while_node->body, out);
    return;
  }
}

void DiscoverUnitsShared(const Stmt &stmt, PtoRegionScan *scan) {
  if (const auto *for_node = stmt.as<ForNode>()) {
    if (for_node->kind == ForKind::kParallel) {
      if (UsesExplicitSIMD(for_node->body)) {
        std::vector<const ForNode *> under;
        CollectAllParallelDeep(for_node->body, under);
        if (!under.empty()) {
          scan->has_unsupported_wrapper = true;
          scan->issue = "an automatic T.Parallel nested inside an explicit "
                        "SIMD T.Parallel is not yet supported";
        }
        return;
      }
      // A two-layer adjacent Parallel nest is one
      // 2D unit; the layout annotation lives on the outermost For. A
      // third Parallel layer (or a Parallel deeper inside) stays
      // unsupported.
      const auto *adjacent_inner = for_node->body.as<ForNode>();
      const bool inner_is_parallel = adjacent_inner != nullptr &&
                                     adjacent_inner->kind == ForKind::kParallel;
      std::vector<const ForNode *> deeper;
      CollectAllParallelDeep(
          inner_is_parallel ? adjacent_inner->body : for_node->body, deeper);
      if (!deeper.empty()) {
        scan->has_unsupported_wrapper = true;
        // Two distinct illegal shapes: a genuine third parallel layer
        // inside an adjacent two-layer nest, or an outer Parallel whose
        // body mixes other statements with an inner Parallel (the nest is
        // not adjacent and cannot form a pure 2D unit).
        scan->issue =
            inner_is_parallel
                ? "three or more nested parallel layers are not yet "
                  "supported"
                : "a T.Parallel whose body mixes other statements with an "
                  "inner T.Parallel (a non-adjacent nest) is not yet "
                  "supported";
      }
      scan->units.push_back(GetRef<For>(for_node));
      return;
    }
    if (for_node->kind == ForKind::kVectorized ||
        for_node->kind == ForKind::kThreadBinding) {
      // Only sequential wrappers may surround a unit: a re-vectorizing loop
      // contradicts the vectorization this conversion already performs, and
      // a thread binding contradicts the no-thread-binding contract of the
      // converting region. A region without T.Parallel under them stays a
      // legitimate hand-written VMI region and is skipped entirely.
      std::vector<const ForNode *> under;
      CollectAllParallelDeep(stmt, under);
      if (!under.empty()) {
        scan->has_unsupported_wrapper = true;
        if (for_node->kind == ForKind::kThreadBinding) {
          std::ostringstream oss;
          oss << "a thread-bound loop (ForKind::kThreadBinding";
          if (const auto *iv = for_node->thread_binding.as<IterVarNode>()) {
            oss << ", thread_binding `" << iv->var->name_hint << "`";
          }
          oss << ") around T.Parallel is not supported: "
                 "thread bindings are not allowed inside a converting "
                 "SIMD_VF block";
          scan->issue = oss.str();
        } else {
          scan->issue =
              "a vectorized loop (ForKind::kVectorized) around T.Parallel is "
              "not yet supported";
        }
      }
      return;
    }
    // A serial loop (with or without an unroll hint) is a sequential
    // wrapper: its iterations run in order and each iteration keeps the
    // unit's semantics, so it stays accepted.
    DiscoverUnitsShared(for_node->body, scan);
    return;
  }
  if (const auto *seq = stmt.as<SeqStmtNode>()) {
    for (const auto &s : seq->seq) {
      DiscoverUnitsShared(s, scan);
    }
    return;
  }
  if (const auto *attr = stmt.as<AttrStmtNode>()) {
    DiscoverUnitsShared(attr->body, scan);
    return;
  }
  if (const auto *block = stmt.as<SBlockNode>()) {
    DiscoverUnitsShared(block->body, scan);
    return;
  }
  if (const auto *realize = stmt.as<SBlockRealizeNode>()) {
    DiscoverUnitsShared(realize->block->body, scan);
    return;
  }
  if (stmt.as<IfThenElseNode>() != nullptr || stmt.as<WhileNode>() != nullptr) {
    // Control-flow wrappers: a Parallel under them is rejected; without
    // one the region may still be a legitimate hand-written VMI region.
    std::vector<const ForNode *> under;
    CollectAllParallelDeep(stmt, under);
    if (!under.empty()) {
      scan->has_unsupported_wrapper = true;
      scan->issue = stmt.as<IfThenElseNode>() != nullptr
                        ? "T.Parallel under a conditional branch is not yet "
                          "supported"
                        : "T.Parallel under a while loop is not yet "
                          "supported";
    }
    return;
  }
}

} // namespace

PtoRegionScan ScanPtoRegion(const Stmt &region) {
  PtoRegionScan scan;
  DiscoverUnitsShared(region, &scan);
  class TileOpScanner : public StmtExprVisitor {
  public:
    explicit TileOpScanner(PtoRegionScan *scan) : scan_(scan) {}
    void VisitStmt_(const AttrStmtNode *op) final {
      if (auto node = op->node.try_cast<PrimExpr>()) {
        VisitExpr(node.value());
      }
      StmtExprVisitor::VisitStmt_(op);
    }
    void VisitStmt_(const BufferStoreNode *op) final {
      if (op->predicate.defined()) {
        VisitExpr(op->predicate.value());
      }
      StmtExprVisitor::VisitStmt_(op);
    }
    void VisitExpr_(const BufferLoadNode *op) final {
      if (op->predicate.defined()) {
        VisitExpr(op->predicate.value());
      }
      StmtExprVisitor::VisitExpr_(op);
    }
    void VisitStmt_(const SBlockNode *op) final {
      StmtExprVisitor::VisitStmt_(op);
      // The scan starts at the enclosing SIMD_VF's body. Another VF on
      // this lexical path would be checked with the outer lanes and then
      // lowered with its own lanes, so it cannot be a transparent wrapper.
      if (op->name_hint == "SIMD_VF") {
        scan_->has_unsupported_wrapper = true;
        scan_->issue = "nested SIMD_VF blocks are not supported; use "
                       "sequential T.SimdVF regions";
      }
    }
    void VisitStmt_(const ForNode *op) final {
      scan_->has_any_parallel |=
          op->kind == ForKind::kParallel && !UsesExplicitSIMD(op->body);
      int saved = forbidden_depth_;
      if (op->kind != ForKind::kSerial && op->kind != ForKind::kUnrolled) {
        ++forbidden_depth_;
      }
      StmtExprVisitor::VisitStmt_(op);
      forbidden_depth_ = saved;
    }
    void VisitStmt_(const IfThenElseNode *op) final {
      ++forbidden_depth_;
      StmtExprVisitor::VisitStmt_(op);
      --forbidden_depth_;
    }
    void VisitStmt_(const WhileNode *op) final {
      ++forbidden_depth_;
      StmtExprVisitor::VisitStmt_(op);
      --forbidden_depth_;
    }
    void VisitExpr_(const CallNode *op) final {
      bool reduce = IsPtoDirectReduce(GetRef<Call>(op));
      bool fill = IsPtoFill(GetRef<Call>(op));
      if (reduce || fill) {
        scan_->has_direct_reduce |= reduce;
        scan_->has_fill |= fill;
        if (forbidden_depth_ != 0) {
          scan_->has_unsupported_wrapper = true;
          scan_->issue = "direct reduce and fill/clear require a region-level "
                         "statement outside "
                         "Parallel, If/While and thread-bound loops";
        }
      }
      StmtExprVisitor::VisitExpr_(op);
    }

  private:
    PtoRegionScan *scan_;
    int forbidden_depth_ = 0;
  };
  TileOpScanner tileop_scanner(&scan);
  tileop_scanner(region);

  if (scan.has_any_parallel && scan.units.empty()) {
    // A Parallel exists but only under an unsupported wrapper.
    scan.has_unsupported_wrapper = true;
    if (scan.issue.empty()) {
      scan.issue = "T.Parallel exists only under an unsupported wrapper";
    }
  }
  return scan;
}

bool IsNoOpStmt(const Stmt &stmt) {
  // A no-op statement carries no work: an Evaluate of an IntImm constant
  // (e.g. a simplified identity else store; treating it as "no else"
  // yields the correct masked-update form), an empty SeqStmt, or a SeqStmt
  // whose every element is itself a no-op. One shared implementation so
  // the branch shapes Verify admits are exactly the ones Legalize
  // rewrites.
  if (const auto *eval = stmt.as<EvaluateNode>()) {
    return eval->value.as<IntImmNode>() != nullptr;
  }
  if (const auto *seq = stmt.as<SeqStmtNode>()) {
    for (const Stmt &s : seq->seq) {
      if (!IsNoOpStmt(s)) {
        return false;
      }
    }
    return true;
  }
  return false;
}

ValueInfo ValueInfo::Vmi(DataType dtype) { return Vmi(dtype.lanes()); }

ValueInfo ValueInfo::Vmi(int64_t lanes) {
  return {lanes == 1 ? ValueKind::kOneLaneVmi : ValueKind::kVectorVmi, lanes};
}

ValueInfo GetValueInfo(const PrimExpr &value, const ValueInfoMap &bindings) {
  if (auto var = value.as<Var>()) {
    auto it = bindings.find(var.value());
    if (it != bindings.end()) {
      return it->second;
    }
  }
  const DataType dtype = value.dtype();
  // Parallel binds every VMI data producer before returning it to a consumer.
  if (dtype.is_scalar()) {
    return {ValueKind::kScalar, 1};
  }
  return {};
}

WidthAdaptation PlanWidthAdaptation(const ValueInfo &value, int64_t lanes) {
  if (value.kind == ValueKind::kScalar) {
    return WidthAdaptation::kBroadcastScalar;
  }
  if (value.IsVmi() && value.logical_lanes == lanes) {
    return WidthAdaptation::kKeep;
  }
  if (value.kind == ValueKind::kOneLaneVmi) {
    return WidthAdaptation::kBroadcastOneLane;
  }
  throw PtoAnalysisError("cannot adapt PTO value to consumer lane width " +
                         std::to_string(lanes));
}

std::string LaneUseToString(LaneUse use) {
  switch (use) {
  case LaneUse::kUniform:
    return "uniform";
  case LaneUse::kVarying:
    return "varying";
  default:
    return "unknown";
  }
}

namespace {

// Shared per-node classification used by both the value mode and the
// metadata mode: value mode whitelists Add/Mul/Cast; metadata
// mode additionally propagates the pure scalar operators and accepts
// StringImm.
LaneUse AnalyzeLaneUseImpl(const PrimExpr &expr, LaneUseContext *ctx,
                           std::string *reason, bool metadata_mode) {
  if (expr.as<IntImmNode>() || expr.as<FloatImmNode>()) {
    return LaneUse::kUniform;
  }
  if (expr.as<StringImmNode>() != nullptr) {
    // String constants are legal metadata (markers); they never
    // appear as computation values (the value whitelist rejects them).
    if (metadata_mode) {
      return LaneUse::kUniform;
    }
    if (reason != nullptr) {
      // Every Unknown carries a concrete reason.
      *reason = "string constants are metadata-only; they cannot appear "
                "in computation values";
    }
    return LaneUse::kUnknown;
  }
  if (const auto *var = expr.as<VarNode>()) {
    if (ctx->value_infos) {
      auto it = ctx->value_infos->find(GetRef<Var>(var));
      if (it != ctx->value_infos->end() && it->second.IsVmi())
        return LaneUse::kVarying;
    }
    if (var == ctx->loop_var.get()) {
      return LaneUse::kVarying;
    }
    if (auto serial = ctx->mapping.SerialVar();
        serial && var == serial.value().get()) {
      // The coordinate of the non-vectorized
      // dimension is a uniform scalar at the program point.
      return LaneUse::kUniform;
    }
    if (ctx->bind_env.count(GetRef<Var>(var))) {
      // Use the recorded classification of the bind (SSA identity).
      if (ctx->classified.count(GetRef<Var>(var))) {
        Integer v = ctx->classified[GetRef<Var>(var)];
        return v->value == 0 ? LaneUse::kUniform : LaneUse::kVarying;
      }
      // On-demand analysis with cycle guard + budget.
      if (ctx->visiting.count(var) != 0) {
        if (reason != nullptr) {
          *reason = "bind self/forward reference detected in lane-use "
                    "analysis";
        }
        return LaneUse::kUnknown;
      }
      if (ctx->budget <= 0) {
        if (reason != nullptr) {
          *reason = "lane-use analysis recursion budget exhausted";
        }
        return LaneUse::kUnknown;
      }
      ctx->visiting.insert(var);
      ctx->budget -= 1;
      const PrimExpr &def = ctx->bind_env[GetRef<Var>(var)];
      LaneUse use = AnalyzeLaneUseImpl(def, ctx, reason, metadata_mode);
      ctx->visiting.erase(var);
      if (use == LaneUse::kUnknown) {
        return use;
      }
      // Record only when the definition is fully analyzed; a cycle hit
      // above leaves it unrecorded so later references re-report.
      ctx->classified.Set(GetRef<Var>(var),
                          Integer(use == LaneUse::kUniform ? 0 : 1));
      return use;
    }
    // Only *known* external scalar vars are Uniform; anything
    // else is undefined -> Unknown.
    if (ctx->external_defs.count(var) != 0) {
      // Being in scope is not enough. An external var consumed as
      // a computation value (or as numeric metadata) must be scalar; a
      // vector-typed var has no scalar input contract for this pass and
      // must not be waved through as Uniform. Buffer data pointers are
      // handle-typed (lanes == 1) and keep flowing through the address
      // and Buffer paths.
      if (var->dtype.lanes() != 1) {
        if (reason != nullptr) {
          std::ostringstream oss;
          oss << "variable `" << var->name_hint << "` has vector dtype "
              << var->dtype
              << "; only scalar inputs are supported as computation values";
          *reason = oss.str();
        }
        return LaneUse::kUnknown;
      }
      return LaneUse::kUniform;
    }
    if (reason != nullptr) {
      std::ostringstream oss;
      oss << "variable `" << var->name_hint
          << "` is not defined in this unit or any outer scope";
      // Record the reason so the caller reports a concrete cause.
      *reason = oss.str();
    }
    return LaneUse::kUnknown;
  }
  if (const auto *load = expr.as<BufferLoadNode>()) {
    if (ctx->storage_values) {
      auto it = ctx->storage_values->find(load->buffer->data);
      if (it != ctx->storage_values->end() && it->second.IsVmi())
        return LaneUse::kVarying;
    }
    if (ctx->storage) {
      auto kind = ctx->storage->Classify(load->buffer, ctx->unit);
      if (kind == pto::StorageClass::kPrivate ||
          kind == pto::StorageClass::kExternal) {
        ctx->storage->ResolveSlot(load->buffer, load->indices, ctx->analyzer,
                                  ctx->bind_env);
        return kind == pto::StorageClass::kPrivate ? LaneUse::kVarying
                                                   : LaneUse::kUniform;
      }
    }
    try {
      auto access =
          AnalyzeBufferAccess(load->buffer, load->indices, /*is_write=*/false,
                              ctx->loop_var, ctx->mapping, ctx->bind_env);
      if (access.pattern == AccessPattern::kLaneUniform) {
        return LaneUse::kUniform;
      }
      return LaneUse::kVarying;
    } catch (const PtoAnalysisError &err) {
      if (reason != nullptr) {
        std::ostringstream oss;
        oss << "nested access to buffer `" << load->buffer->name
            << "`: " << err.what();
        *reason = oss.str();
      }
      return LaneUse::kUnknown;
    }
  }
  if (const auto *call = expr.as<CallNode>()) {
    Call ref = GetRef<Call>(call);
    // Metadata keeps the original scalar expression, so it may use integer
    // widths outside the VMI data contract.
    if (metadata_mode && call->op.same_as(builtin::bitwise_not()) &&
        call->args.size() == 1 && call->dtype.is_scalar() &&
        (call->dtype.is_int() || call->dtype.is_uint())) {
      return AnalyzeLaneUseImpl(call->args[0], ctx, reason, metadata_mode);
    }
    PtoBitOp bit_op = GetPtoBitOp(ref);
    const char *mask_op = GetPtoMaskLogicOp(ref);
    if (bit_op != PtoBitOp::kNone || mask_op) {
      try {
        if (mask_op) {
          CheckPtoMaskCall(ref, mask_op);
        } else {
          CheckPtoBitOp(ref, bit_op);
        }
      } catch (const PtoAnalysisError &err) {
        if (reason) {
          *reason = err.what();
        }
        return LaneUse::kUnknown;
      }
      LaneUse use = LaneUse::kUniform;
      for (const PrimExpr &arg : call->args) {
        LaneUse arg_use = AnalyzeLaneUseImpl(arg, ctx, reason, metadata_mode);
        if (arg_use == LaneUse::kUnknown) {
          return arg_use;
        }
        if (arg_use == LaneUse::kVarying) {
          use = arg_use;
        }
      }
      return use;
    }
    if (GetPtoUnaryOp(ref)) {
      return AnalyzeLaneUseImpl(call->args[0], ctx, reason, metadata_mode);
    }
    if (const auto *op = call->op.as<OpNode>();
        op && (op->name == "tl.infinity" || op->name == "tir.infinity")) {
      return LaneUse::kUniform;
    }
  }
  // Binary/relational/boolean operators: both value and metadata modes
  // recurse into operands; the *type* checks live in the callers.
  LaneUse lhs, rhs;
  const PrimExpr *a = nullptr;
  const PrimExpr *b = nullptr;
  bool known_op = false;
  if (const auto *add = expr.as<AddNode>()) {
    a = &add->a;
    b = &add->b;
    known_op = true;
  } else if (const auto *sub = expr.as<SubNode>()) {
    a = &sub->a;
    b = &sub->b;
    known_op = true;
  } else if (const auto *mul = expr.as<MulNode>()) {
    a = &mul->a;
    b = &mul->b;
    known_op = true;
  } else if (const auto *div = expr.as<DivNode>()) {
    a = &div->a;
    b = &div->b;
    known_op = true;
  } else if (const auto *mod = expr.as<ModNode>()) {
    a = &mod->a;
    b = &mod->b;
    known_op = metadata_mode;
  } else if (const auto *fdiv = expr.as<FloorDivNode>()) {
    a = &fdiv->a;
    b = &fdiv->b;
    known_op = true;
  } else if (const auto *fmod = expr.as<FloorModNode>()) {
    a = &fmod->a;
    b = &fmod->b;
    known_op = true;
  } else if (const auto *minn = expr.as<MinNode>()) {
    // Min/Max classify like the comparisons in value mode too;
    // LegalizeParallelToPTO rewrites them into Select before Vectorize.
    a = &minn->a;
    b = &minn->b;
    known_op = true;
  } else if (const auto *maxx = expr.as<MaxNode>()) {
    a = &maxx->a;
    b = &maxx->b;
    known_op = true;
  } else if (const auto *lt = expr.as<LTNode>()) {
    a = &lt->a;
    b = &lt->b;
    known_op = true; // cmp -> vcmp/vcmps
  } else if (const auto *len = expr.as<LENode>()) {
    a = &len->a;
    b = &len->b;
    known_op = true;
  } else if (const auto *gt = expr.as<GTNode>()) {
    a = &gt->a;
    b = &gt->b;
    known_op = true;
  } else if (const auto *ge = expr.as<GENode>()) {
    a = &ge->a;
    b = &ge->b;
    known_op = true;
  } else if (const auto *eq = expr.as<EQNode>()) {
    a = &eq->a;
    b = &eq->b;
    known_op = true;
  } else if (const auto *ne = expr.as<NENode>()) {
    a = &ne->a;
    b = &ne->b;
    known_op = true;
  } else if (const auto *andn = expr.as<AndNode>()) {
    a = &andn->a;
    b = &andn->b;
    known_op = true;
  } else if (const auto *orn = expr.as<OrNode>()) {
    a = &orn->a;
    b = &orn->b;
    known_op = true;
  }
  if (known_op) {
    lhs = AnalyzeLaneUseImpl(*a, ctx, reason, metadata_mode);
    rhs = AnalyzeLaneUseImpl(*b, ctx, reason, metadata_mode);
    if (lhs == LaneUse::kUnknown || rhs == LaneUse::kUnknown) {
      if (reason != nullptr && reason->empty()) {
        *reason = "unknown operand in binary expression";
      }
      return LaneUse::kUnknown;
    }
    return (lhs == LaneUse::kUniform && rhs == LaneUse::kUniform)
               ? LaneUse::kUniform
               : LaneUse::kVarying;
  }
  if (const auto *notn = expr.as<NotNode>()) {
    return AnalyzeLaneUseImpl(notn->a, ctx, reason, metadata_mode);
  }
  if (const auto *sel = expr.as<SelectNode>()) {
    // Select(cond, t, f) -> vsel; lane use is varying if any arm is.
    LaneUse c = AnalyzeLaneUseImpl(sel->condition, ctx, reason, metadata_mode);
    LaneUse tv =
        AnalyzeLaneUseImpl(sel->true_value, ctx, reason, metadata_mode);
    LaneUse fv =
        AnalyzeLaneUseImpl(sel->false_value, ctx, reason, metadata_mode);
    if (c == LaneUse::kUnknown || tv == LaneUse::kUnknown ||
        fv == LaneUse::kUnknown) {
      if (reason != nullptr && reason->empty()) {
        *reason = "unknown operand in Select";
      }
      return LaneUse::kUnknown;
    }
    return (c == LaneUse::kUniform && tv == LaneUse::kUniform &&
            fv == LaneUse::kUniform)
               ? LaneUse::kUniform
               : LaneUse::kVarying;
  }
  if (const auto *cast = expr.as<CastNode>()) {
    return AnalyzeLaneUseImpl(cast->value, ctx, reason, metadata_mode);
  }
  if (reason != nullptr) {
    std::ostringstream oss;
    oss << "unsupported " << (metadata_mode ? "metadata" : "value") << " node `"
        << expr->GetTypeKey() << "` in lane-use analysis";
    *reason = oss.str();
  }
  return LaneUse::kUnknown;
}

} // namespace

ValueInfo AnalyzeComputeValue(const PrimExpr &expr, LaneUseContext *ctx) {
  if (const auto *var = expr.as<VarNode>()) {
    if (ctx->value_infos) {
      auto it = ctx->value_infos->find(GetRef<Var>(var));
      if (it != ctx->value_infos->end())
        return it->second;
    }
  }
  if (const auto *load = expr.as<BufferLoadNode>()) {
    if (ctx->storage_values) {
      auto it = ctx->storage_values->find(load->buffer->data);
      if (it != ctx->storage_values->end())
        return it->second;
    }
  }
  if (const auto *cast = expr.as<CastNode>())
    return AnalyzeComputeValue(cast->value, ctx);
  auto merge = [](ValueInfo a, ValueInfo b) {
    if (a.IsVmi() || b.IsVmi()) {
      return ValueInfo::Vmi(std::max(a.logical_lanes, b.logical_lanes));
    }
    return ValueInfo{ValueKind::kScalar, 1};
  };
  const PrimExpr *lhs = nullptr;
  const PrimExpr *rhs = nullptr;
  if (const auto *op = expr.as<AddNode>()) {
    lhs = &op->a;
    rhs = &op->b;
  } else if (const auto *op = expr.as<SubNode>()) {
    lhs = &op->a;
    rhs = &op->b;
  } else if (const auto *op = expr.as<MulNode>()) {
    lhs = &op->a;
    rhs = &op->b;
  } else if (const auto *op = expr.as<DivNode>()) {
    lhs = &op->a;
    rhs = &op->b;
  } else if (const auto *op = expr.as<FloorDivNode>()) {
    lhs = &op->a;
    rhs = &op->b;
  } else if (const auto *op = expr.as<FloorModNode>()) {
    lhs = &op->a;
    rhs = &op->b;
  }
  if (const auto *call = expr.as<CallNode>()) {
    if (call->args.size() == 1) {
      return AnalyzeComputeValue(call->args[0], ctx);
    }
    if (call->args.size() == 2 &&
        GetPtoBitOp(GetRef<Call>(call)) != PtoBitOp::kNone) {
      return merge(AnalyzeComputeValue(call->args[0], ctx),
                   AnalyzeComputeValue(call->args[1], ctx));
    }
  }
  if (lhs)
    return merge(AnalyzeComputeValue(*lhs, ctx),
                 AnalyzeComputeValue(*rhs, ctx));
  std::string reason;
  auto use = AnalyzeLaneUse(expr, ctx, &reason);
  ICHECK(use != LaneUse::kUnknown)
      << "PTO value classification failed: " << reason;
  return use == LaneUse::kVarying ? ValueInfo::Vmi(ctx->mapping.lanes)
                                  : ValueInfo{ValueKind::kScalar, 1};
}

LaneUse AnalyzeLaneUse(const PrimExpr &expr, LaneUseContext *ctx,
                       std::string *reason) {
  return AnalyzeLaneUseImpl(expr, ctx, reason, /*metadata_mode=*/false);
}

LaneUse AnalyzeLaneUseMeta(const PrimExpr &expr, LaneUseContext *ctx,
                           std::string *reason) {
  return AnalyzeLaneUseImpl(expr, ctx, reason, /*metadata_mode=*/true);
}

const char *GetPtoUnaryOp(const Call &call) {
  const auto *op = call->op.as<OpNode>();
  return op ? VectorUnaryVmiOpName(op->name) : nullptr;
}

bool CheckPtoValueCall(const Call &call) {
  PtoBitOp bit_op = GetPtoBitOp(call);
  if (bit_op != PtoBitOp::kNone) {
    CheckPtoBitOp(call, bit_op);
    return true;
  }
  if (!GetPtoUnaryOp(call)) {
    return false;
  }
  const std::string name = Downcast<Op>(call->op)->name;
  if (call->args.size() != 1) {
    throw PtoAnalysisError(name + " expects exactly one operand");
  }
  DataType dtype = call->args[0].dtype();
  if (call->dtype != dtype || !IsSupportedVectorUnaryOp(name, dtype)) {
    std::ostringstream oss;
    oss << name << " vectorization does not support " << dtype << " -> "
        << call->dtype << " (per the VMI operation matrix)";
    throw PtoAnalysisError(oss.str());
  }
  return true;
}

bool IsSupportedVectorUnaryOp(const std::string &tirx_op, DataType dtype) {
  // Per the VMI ISA documentation: vabs admits f16/bf16/f32 (the vabs
  // instruction also takes si8/si16/si32, but the frontend abs() lowers
  // integers to Select and returns unsigned inputs unchanged, so integer
  // dtypes never reach this gate); vexp/vln/vsqrt admit f16/f32 only.
  // The op-name admission is the shared VectorUnaryVmiOpName table.
  if (VectorUnaryVmiOpName(tirx_op) == nullptr) {
    return false;
  }
  const int bits = dtype.bits();
  if (tirx_op == "tirx.fabs") {
    return (dtype.is_float() || dtype.is_bfloat16()) &&
           (bits == 16 || bits == 32);
  }
  // The remaining mapped names (exp/log/sqrt).
  return dtype.is_float() && (bits == 16 || bits == 32);
}

bool ExprUsesVar(const PrimExpr &expr, const VarNode *var) {
  bool uses = false;
  PostOrderVisit(expr, [&](const ObjectRef &n) {
    if (n.as<VarNode>() == var) {
      uses = true;
    }
  });
  return uses;
}

bool ContainsRuntimeValue(const PrimExpr &expr) {
  // True when the expression reads any runtime value: a Var (external
  // scalar, serial coordinate, scalar buffer base) or a BufferLoad.
  // A pure IntImm/FloatImm tree is a compile-time constant.
  bool has_runtime = false;
  PostOrderVisit(expr, [&](const ObjectRef &n) {
    if (n.as<VarNode>() != nullptr || n.as<BufferLoadNode>() != nullptr) {
      has_runtime = true;
    }
  });
  return has_runtime;
}

bool IsFloatFamily(DataType dtype) {
  return dtype.is_float() || dtype.is_bfloat16();
}

bool IsSupportedVectorBinaryOp(const std::string &vmi_op, DataType dtype) {
  // Operation-level matrix (per the VMI ISA documentation):
  //   vmul requires i16/i32/f16/bf16/f32; vadd/vsub additionally allow i8;
  //   vdiv admits f16/f32 only (no bf16, no integers).
  int bits = dtype.bits();
  const bool is_int_like = dtype.is_int() || dtype.is_uint();
  const bool is_fp_like =
      (dtype.is_float() || dtype.is_bfloat16()) && (bits == 16 || bits == 32);
  if (vmi_op == "vmul") {
    return (is_int_like && (bits == 16 || bits == 32)) || is_fp_like;
  }
  if (vmi_op == "vadd" || vmi_op == "vsub") {
    return (is_int_like && (bits == 8 || bits == 16 || bits == 32)) ||
           is_fp_like;
  }
  if (vmi_op == "vmax" || vmi_op == "vmin")
    return is_fp_like;
  if (vmi_op == "vdiv") {
    return dtype.is_float() && (bits == 16 || bits == 32);
  }
  if (vmi_op == "vfloordiv" || vmi_op == "vfloormod") {
    return IsPtoIndexDType(dtype);
  }
  return false;
}

namespace {

/*! Cast attributes read from a Cast node's backend annotations. The
 * frontend writes them in two shapes — TIR Imm objects (the `T.cast` helper
 * lowers `round`/`sat` to StringImm/IntImm) and plain Python values (a raw
 * `annotations={...}` map) — so both spellings are accepted here. */
struct CastAttributes {
  /*! The `round` annotation was present (even when its value is empty). */
  bool has_round_key = false;
  /*! A non-empty rounding mode was requested. An empty string is the
   * frontend's way of saying "use the backend default", which is not the same
   * as an unusable mode. */
  bool has_round_mode = false;
  std::string round; // raw spelling, e.g. "rn", "R", "A"
  bool has_sat = false;
  bool sat = true;
  bool has_rbits = false;
};

/*! Read the attribute hints of a Cast. Returns false with *why set when an
 * annotation uses a shape this version cannot read; an unknown key is
 * itself a rejection, because silently ignoring an attribute would accept
 * a conversion whose semantics were never checked. */
bool ReadCastAttributes(const ffi::Map<ffi::String, ffi::Any> &annotations,
                        CastAttributes *attrs, std::string *why) {
  for (const auto &[key_ref, value] : annotations) {
    const std::string key = key_ref;
    if (key == "round") {
      attrs->has_round_key = true;
      if (const auto *imm = value.as<StringImmNode>()) {
        attrs->round = imm->value;
      } else if (auto str = value.as<ffi::String>()) {
        attrs->round = std::string(*str);
      } else {
        *why = "annotation `round` must be a string";
        return false;
      }
      attrs->has_round_mode = !attrs->round.empty();
    } else if (key == "sat") {
      attrs->has_sat = true;
      if (const auto *imm = value.as<IntImmNode>()) {
        // Only 0/1 are accepted, in the TIR Imm form as well.
        if (imm->value != 0 && imm->value != 1) {
          *why = "annotation `sat` must be a boolean (got " +
                 std::to_string(imm->value) + ")";
          return false;
        }
        attrs->sat = imm->value != 0;
      } else if (auto flag = value.as<bool>()) {
        attrs->sat = *flag;
      } else if (auto flag = value.as<int64_t>()) {
        // Only 0/1 are accepted: any other integer is a malformed flag, not a
        // saturating request.
        if (*flag != 0 && *flag != 1) {
          *why = "annotation `sat` must be a boolean (got " +
                 std::to_string(*flag) + ")";
          return false;
        }
        attrs->sat = *flag != 0;
      } else {
        *why = "annotation `sat` must be a boolean";
        return false;
      }
    } else if (key == "rbits") {
      attrs->has_rbits = true;
    } else {
      *why = "unknown cast annotation `" + key + "`";
      return false;
    }
  }
  return true;
}

/*! Normalize a rounding spelling to the canonical PTO token. The frontend
 * aliases are PTX-style; the canonical tokens are the ones the FP8
 * conversion policy reasons about. Returns false with *why set for
 * aliases with no PTO meaning (stochastic rounding) and for tokens no
 * conversion in this version accepts. */
bool NormalizeCastRounding(const std::string &raw, std::string *token,
                           std::string *why) {
  if (raw == "rn") {
    *token = "R";
    return true;
  }
  if (raw == "rz") {
    *token = "Z";
    return true;
  }
  if (raw == "rp") {
    *token = "C";
    return true;
  }
  if (raw == "rm") {
    *token = "F";
    return true;
  }
  if (raw == "rs") {
    *why = "stochastic rounding (round=\"rs\") is not supported";
    return false;
  }
  static const std::set<std::string> kCanonical = {"R", "A", "H",
                                                   "Z", "C", "F"};
  if (kCanonical.count(raw) != 0) {
    *token = raw;
    return true;
  }
  *why = "rounding mode \"" + raw + "\" has no PTO vcvt meaning";
  return false;
}

/*! Rounding tokens the FP8 narrowing policy accepts. The conversion
 * instructions the FP8 target exposes are FP32 <-> FP8; the matrix is the
 * intersection of the PTODSL vcvt interface with that support, so the source
 * is always float32 and only the rounding modes below are expressible. */
const std::set<std::string> &Fp8AllowedRoundings() {
  static const std::set<std::string> kAllowed = {"R", "A", "H", "Z"};
  return kAllowed;
}

bool IsVmiLogicalLaneCount(int64_t lanes) {
  return lanes == 64 || lanes == 128 || lanes == 256;
}

std::string CastContextName(const CastContext &context) {
  if (context.value_kind == CastValueKind::kScalar) {
    return "scalar";
  }
  std::ostringstream oss;
  oss << "VMI, lanes=" << context.logical_lanes;
  return oss.str();
}

bool IsHalfOrBfloat16(DataType dtype) {
  return (dtype.is_float() && dtype.bits() == 16) || dtype.is_bfloat16();
}

/*! Identity attribute policy shared by the scalar and VMI branches: no
 * conversion exists, so sat=True (the default) is normalized away, while
 * sat=False or a non-empty rounding mode would claim a conversion semantic
 * that does not exist.  An explicit empty round key means "no request". */
bool ValidateIdentityCastAttributes(const CastAttributes &attrs,
                                    const std::string &pair, std::string *why) {
  if (attrs.has_sat && !attrs.sat) {
    *why = pair + ": an identity cast does not accept sat=False";
    return false;
  }
  if (attrs.has_round_mode) {
    *why = pair + ": an identity cast does not accept a rounding mode";
    return false;
  }
  return true;
}

/*! Integer attribute policy: rounding modes are never accepted; an explicit
 * saturation flag is accepted only for narrowing and only as sat=False, which
 * the direct NOSAT lowering already honours. */
bool ValidateIntegerCastAttributes(const CastAttributes &attrs, bool narrowing,
                                   const std::string &pair, std::string *why) {
  if (attrs.has_round_mode) {
    *why = pair + ": integer conversions do not accept a rounding mode";
    return false;
  }
  if (attrs.has_sat) {
    if (!narrowing) {
      *why = pair + ": this integer conversion does not accept an explicit "
                    "saturation flag";
      return false;
    }
    if (attrs.sat) {
      *why = pair + ": integer narrowing accepts only sat=False (NOSAT), not "
                    "sat=True";
      return false;
    }
  }
  return true;
}

Optional<CastPlan> PlanScalarValueCast(const CastNode *cast,
                                       const CastAttributes &attrs,
                                       const std::string &pair,
                                       std::string *reason) {
  const DataType from = cast->value.dtype();
  const DataType to = cast->dtype;
  auto reject = [&](const std::string &why) {
    if (reason != nullptr) {
      *reason = why;
    }
    return Optional<CastPlan>();
  };
  ScalarCastPlan keep;
  keep.action = ScalarCastAction::kKeepInput;
  ScalarCastPlan rebuild;
  rebuild.action = ScalarCastAction::kRebuildCast;
  if (from == to) {
    std::string why;
    if (!ValidateIdentityCastAttributes(attrs, pair, &why)) {
      return reject(why);
    }
    return CastPlan{keep};
  }
  const bool from_int = from.is_int() || from.is_uint();
  const bool to_int = to.is_int() || to.is_uint();
  if (from_int && to_int) {
    std::string why;
    if (!ValidateIntegerCastAttributes(attrs, to.bits() < from.bits(), pair,
                                       &why)) {
      return reject(why);
    }
    // Scalar integer casts keep the ordinary pto.cast/trunc lowering; the
    // validated sat=False marker is already honoured and is normalized away.
    return CastPlan{rebuild};
  }
  if (IsSupportedFloat8(from) || IsSupportedFloat8(to)) {
    return reject(pair + ": non-identity FP8 conversions have no scalar "
                         "lowering inside SIMD_VF; use a lane-varying value "
                         "so the cast takes the VMI vcvt path");
  }
  const bool from_float = from.is_float() || from.is_bfloat16();
  const bool to_float = to.is_float() || to.is_bfloat16();
  if (from_float && to_float) {
    if (attrs.has_round_mode) {
      return reject(pair + ": this conversion does not accept a rounding mode");
    }
    const bool from_f32 = from.is_float() && from.bits() == 32;
    const bool to_f32 = to.is_float() && to.bits() == 32;
    if (!from_f32 && to_f32) {
      if (attrs.has_sat) {
        return reject(pair + ": this direction does not accept a saturation "
                             "mode");
      }
      return CastPlan{rebuild};
    }
    if (from_f32 && !to_f32) {
      if (attrs.has_sat && attrs.sat) {
        return reject(pair + ": this conversion accepts only sat=False "
                             "(non-saturating)");
      }
      return CastPlan{rebuild};
    }
    return reject(pair + ": cross-family f16 <-> bf16 narrowing has no "
                         "validated rounding semantics; convert through "
                         "float32 explicitly");
  }
  return reject(pair + ": no scalar conversion exists for this dtype pair");
}

Optional<CastPlan>
PlanVectorValueCast(const CastNode *cast, const CastContext &context,
                    const CastAttributes &attrs, const std::string &round_token,
                    const std::string &pair, std::string *reason) {
  const DataType from = cast->value.dtype();
  const DataType to = cast->dtype;
  auto reject = [&](const std::string &why) {
    if (reason != nullptr) {
      *reason = why;
    }
    return Optional<CastPlan>();
  };
  if (from == to) {
    std::string why;
    if (!ValidateIdentityCastAttributes(attrs, pair, &why)) {
      return reject(why);
    }
    return CastPlan{VectorCastPlan{}};
  }
  const bool from_int = from.is_int() || from.is_uint();
  const bool to_int = to.is_int() || to.is_uint();
  if (from_int && to_int) {
    std::string why;
    if (!ValidateIntegerCastAttributes(attrs, to.bits() < from.bits(), pair,
                                       &why)) {
      return reject(why);
    }
    VectorCastPlan plan;
    if (to.bits() < from.bits()) {
      plan.steps.push_back(
          {VectorCastOp::kIntegerTruncate, from, to, "", "NOSAT"});
      return CastPlan{plan};
    }
    if (to.bits() == from.bits()) {
      plan.steps.push_back(
          {VectorCastOp::kIntegerReinterpret, from, to, "", ""});
      return CastPlan{plan};
    }
    if (from.is_int() == to.is_int()) {
      plan.steps.push_back({VectorCastOp::kIntegerExtend, from, to, "", ""});
      return CastPlan{plan};
    }
    // Cross-sign widening: extend with the source signedness to the target
    // width, then re-view the same-width result as the target signedness.
    const DataType intermediate =
        from.is_int() ? DataType::Int(to.bits()) : DataType::UInt(to.bits());
    plan.steps.push_back(
        {VectorCastOp::kIntegerExtend, from, intermediate, "", ""});
    plan.steps.push_back(
        {VectorCastOp::kIntegerReinterpret, intermediate, to, "", ""});
    return CastPlan{plan};
  }
  const bool from_fp8 = IsSupportedFloat8(from);
  const bool to_fp8 = IsSupportedFloat8(to);
  if (from_fp8 || to_fp8) {
    if (from_fp8 && to_fp8) {
      return reject(pair + ": conversions between the two FP8 types are not "
                           "supported (they are not a reinterpretation)");
    }
    VectorCastPlan plan;
    if (from_fp8 && to.is_float() && to.bits() == 32) {
      // Exact widening: a real conversion with no options.  An explicit hint
      // here cannot be honoured, so it is rejected rather than dropped.
      if (attrs.has_round_key) {
        return reject(pair + ": this direction does not accept a rounding "
                             "mode");
      }
      if (attrs.has_sat) {
        return reject(pair + ": this direction does not accept a saturation "
                             "mode");
      }
      plan.steps.push_back(
          {VectorCastOp::kNonIntegerConvert, from, to, "", ""});
      return CastPlan{plan};
    }
    if (to_fp8) {
      if (!(from.is_float() && from.bits() == 32)) {
        return reject(pair + ": the FP8 conversion instructions take a "
                             "float32 source; convert to float32 explicitly "
                             "first");
      }
      const std::set<std::string> &allowed = Fp8AllowedRoundings();
      const std::string token = attrs.has_round_mode ? round_token : "R";
      if (allowed.count(token) == 0) {
        std::ostringstream oss;
        oss << pair << ": rounding mode " << token;
        if (attrs.has_round_mode && attrs.round != token) {
          oss << " (from round=\"" << attrs.round << "\")";
        }
        oss << " is not supported by the FP8 conversion policy";
        return reject(oss.str());
      }
      plan.steps.push_back({VectorCastOp::kNonIntegerConvert, from, to, token,
                            (attrs.has_sat && !attrs.sat) ? "NOSAT" : "SAT"});
      return CastPlan{plan};
    }
    return reject(pair + ": FP8 -> FP16/BF16 and FP8 <-> integer are out of "
                         "scope; convert through float32 explicitly");
  }
  const bool from_float = from.is_float() || from.is_bfloat16();
  const bool to_float = to.is_float() || to.is_bfloat16();
  if (from_float && to_float) {
    if (attrs.has_round_mode) {
      return reject(pair + ": this conversion does not accept a rounding mode");
    }
    const bool from_f32 = from.is_float() && from.bits() == 32;
    const bool to_f32 = to.is_float() && to.bits() == 32;
    VectorCastPlan plan;
    if (!from_f32 && to_f32) {
      if (attrs.has_sat) {
        return reject(pair + ": this direction does not accept a saturation "
                             "mode");
      }
      // A real conversion: empty options are not a no-op.
      plan.steps.push_back(
          {VectorCastOp::kNonIntegerConvert, from, to, "", ""});
      return CastPlan{plan};
    }
    if (from_f32 && !to_f32) {
      if (attrs.has_sat && attrs.sat) {
        return reject(pair + ": this conversion accepts only sat=False "
                             "(NOSAT)");
      }
      // Pin R/NOSAT so the generated vcvt never depends on PTODSL defaults.
      plan.steps.push_back(
          {VectorCastOp::kNonIntegerConvert, from, to, "R", "NOSAT"});
      return CastPlan{plan};
    }
    return reject(pair + ": cross-family f16 <-> bf16 narrowing has no "
                         "validated rounding semantics; convert through "
                         "float32 explicitly");
  }
  return reject(pair + ": float <-> integer conversions need per-combination "
                       "rounding/overflow validation");
}

} // namespace

Optional<CastPlan> PlanValueCast(const CastNode *cast,
                                 const CastContext &context,
                                 std::string *reason) {
  DataType from = cast->value.dtype();
  DataType to = cast->dtype;
  auto reject = [&](const std::string &why) {
    if (reason != nullptr) {
      *reason = why;
    }
    return Optional<CastPlan>();
  };
  std::ostringstream pair_os;
  pair_os << "`" << from << " -> " << to << "` (" << CastContextName(context)
          << ")";
  const std::string pair = pair_os.str();
  // The explicit kind is authoritative; lanes only validate it.  Inferring
  // the kind from lanes would turn a lane-1 VMI value into a scalar and a
  // scalar request with vector lanes into an implicit VMI fallback.
  if (context.value_kind == CastValueKind::kScalar) {
    if (context.logical_lanes != 1) {
      return reject(pair +
                    ": a scalar cast context must have exactly one "
                    "logical lane, got " +
                    std::to_string(context.logical_lanes));
    }
  } else {
    if (context.logical_lanes == 1) {
      return reject(pair +
                    ": casting a one-lane VMI value (such as a finalized "
                    "reducer result) is unsupported");
    }
    if (!IsVmiLogicalLaneCount(context.logical_lanes)) {
      return reject(pair +
                    ": a VMI cast context requires logical lanes "
                    "64/128/256, got " +
                    std::to_string(context.logical_lanes));
    }
  }
  // Dtype legality is a precondition for every action plan,
  // including identity — unsupported types (int24/FP64/...) must not slip
  // through the same-type fast path.  The Parallel value path is
  // element-wise, so both sides must be lane-1 elements.
  if (from.lanes() != 1 || to.lanes() != 1) {
    return reject(pair + ": value casts require lane-1 element operands");
  }
  if (!IsSupportedElementDType(from) || !IsSupportedElementDType(to)) {
    return reject(pair + " is outside the supported element dtype set");
  }

  // Attributes are parsed for every Cast, not just the FP8 ones: the
  // expression whitelist deliberately does not look inside a cast's
  // annotations (attributes are not computation values), so an annotation the
  // selected conversion cannot honour has to be rejected here — for every
  // direction — or it would be dropped silently.
  CastAttributes attrs;
  std::string why;
  if (!ReadCastAttributes(cast->annotations, &attrs, &why)) {
    return reject(pair + ": " + why);
  }
  if (attrs.has_rbits) {
    return reject(pair + ": the rbits operand (stochastic rounding) is not "
                         "supported");
  }
  std::string round_token;
  if (attrs.has_round_mode &&
      !NormalizeCastRounding(attrs.round, &round_token, &why)) {
    return reject(pair + ": " + why);
  }

  if (context.value_kind == CastValueKind::kScalar) {
    return PlanScalarValueCast(cast, attrs, pair, reason);
  }
  return PlanVectorValueCast(cast, context, attrs, round_token, pair, reason);
}

std::optional<std::string> FindUnsupportedExprNode(const PrimExpr &expr) {
  WhitelistVisitor visitor;
  visitor(expr);
  return visitor.unsupported;
}

// ---------------------------------------------------------------------------
// Relational kinds shared with VectorizeParallelToPTO / VerifyParallelToPTO
// ---------------------------------------------------------------------------

namespace {

template <typename Node>
bool TakeRelational(const PrimExpr &e, PrimExpr *a, PrimExpr *b) {
  const Node *n = e.as<Node>();
  if (n == nullptr) {
    return false;
  }
  if (a != nullptr) {
    *a = n->a;
  }
  if (b != nullptr) {
    *b = n->b;
  }
  return true;
}

const RelationalKind kRelationalTable[6] = {
    {TakeRelational<EQNode>, "eq",
     [](const PrimExpr &a, const PrimExpr &b) { return a == b; }},
    {TakeRelational<NENode>, "ne",
     [](const PrimExpr &a, const PrimExpr &b) { return a != b; }},
    {TakeRelational<LTNode>, "lt",
     [](const PrimExpr &a, const PrimExpr &b) { return a < b; }},
    {TakeRelational<LENode>, "le",
     [](const PrimExpr &a, const PrimExpr &b) { return a <= b; }},
    {TakeRelational<GTNode>, "gt",
     [](const PrimExpr &a, const PrimExpr &b) { return a > b; }},
    {TakeRelational<GENode>, "ge",
     [](const PrimExpr &a, const PrimExpr &b) { return a >= b; }},
};

} // namespace

const RelationalKind *MatchRelational(const PrimExpr &e, PrimExpr *a,
                                      PrimExpr *b) {
  for (const RelationalKind &kind : kRelationalTable) {
    if (kind.take(e, a, b)) {
      return &kind;
    }
  }
  return nullptr;
}

const RelationalKind *RelationalByName(const char *cmp) {
  for (const RelationalKind &kind : kRelationalTable) {
    if (std::strcmp(kind.cmp, cmp) == 0) {
      return &kind;
    }
  }
  return nullptr;
}

bool IsSafelyReevaluableValue(const PrimExpr &value) {
  class Checker : public tirx::ExprVisitor {
  public:
    bool safe = true;

    // A load observes memory; re-expanding the bind would re-read it at a later
    // program point. A call may read hidden state or have an effect. Neither is
    // substitutable, so the bind var stays an opaque SSA snapshot.
    void VisitExpr_(const tirx::BufferLoadNode *) final { safe = false; }
    void VisitExpr_(const tirx::CallNode *) final { safe = false; }
  };
  Checker checker;
  checker(value);
  return checker.safe;
}

bool IsProvablyTruePredicate(const PrimExpr &predicate,
                             arith::Analyzer *analyzer,
                             const Map<Var, PrimExpr> &bind_env) {
  // Resolve the pass's scalar Bind values first: a `Bind`-defined bool is a
  // known value to both passes, and Verify already checked those references.
  class BindSubstituter : public tirx::ExprMutator {
  public:
    explicit BindSubstituter(const Map<Var, PrimExpr> &env) : env_(env) {}
    PrimExpr VisitExpr_(const VarNode *op) final {
      auto it = env_.find(GetRef<Var>(op));
      return it != env_.end() ? (*it).second : GetRef<PrimExpr>(op);
    }

  private:
    const Map<Var, PrimExpr> &env_;
  };
  PrimExpr value = predicate;
  if (!bind_env.empty()) {
    // A Bind value may itself reference another bound var; a couple of rounds
    // resolve the shallow chains these passes admit without a fixpoint search.
    for (int round = 0; round < 3; ++round) {
      BindSubstituter substituter(bind_env);
      PrimExpr next = substituter(std::move(value));
      if (next.same_as(value)) {
        break;
      }
      value = std::move(next);
    }
  }
  // No explicit Simplify here: CanProve folds the expression itself, and what
  // decides a range-only predicate is the facts bound into the analyzer (a loop
  // bound makes `_t < extent` provable), not an extra rewrite.
  return analyzer->CanProve(value);
}

// Handle-valued Binds: `Bind(q, A.data)` makes q a pointer the lane analysis
// cannot follow. Record the target Vars only; resolving their origin was
// unsound (a handle BufferLoad holds a pointer value, not the storage that
// contains it) and let two admission bypasses through.
void CollectHandleBindTargets(const Stmt &stmt,
                              std::set<const VarNode *> *out) {
  class Collector : public tirx::StmtExprVisitor {
  public:
    explicit Collector(std::set<const VarNode *> *out) : out_(out) {}
    void VisitStmt_(const tirx::BindNode *op) final {
      if (op->value.dtype().is_handle()) {
        out_->insert(op->var.get());
      }
      tirx::StmtExprVisitor::VisitStmt_(op);
    }

  private:
    std::set<const VarNode *> *out_;
  };
  Collector collector(out);
  collector(stmt);
}

StorageOrigins CollectStorageOrigins(const Stmt &body,
                                     const Map<Var, Buffer> &buffer_map,
                                     bool no_alias,
                                     const Array<Var> &non_restrict_params) {
  StorageOrigins origins;
  origins.no_alias = no_alias;
  for (const auto &kv : buffer_map) {
    origins.parameters.insert(kv.second->data.get());
  }
  auto add_non_restrict = [&origins, &buffer_map](const Var &var) {
    if (!var.defined()) {
      return;
    }
    origins.non_restrict.insert(var.get());
    // The frontend may name the handle parameter or its data Var.
    if (auto buffer = buffer_map.Get(var)) {
      origins.non_restrict.insert(buffer.value()->data.get());
    }
  };
  for (const Var &var : non_restrict_params) {
    add_non_restrict(var);
  }

  class Collector : public tirx::StmtExprVisitor {
  public:
    explicit Collector(StorageOrigins *out) : out_(out) {}
    void VisitStmt_(const tirx::SBlockNode *op) final {
      for (const Buffer &buffer : op->alloc_buffers) {
        out_->allocations.insert(buffer->data.get());
      }
      auto it = op->annotations.find(tl::attr::kNonRestrictParams);
      if (it != op->annotations.end()) {
        if (const auto *arr = (*it).second.as<ArrayObj>()) {
          for (const Var &var : Downcast<Array<Var>>((*it).second)) {
            if (var.defined()) {
              out_->non_restrict.insert(var.get());
            }
          }
        }
      }
      tirx::StmtExprVisitor::VisitStmt_(op);
    }

    void VisitStmt_(const tirx::AllocBufferNode *op) final {
      out_->allocations.insert(op->buffer->data.get());
      tirx::StmtExprVisitor::VisitStmt_(op);
    }

  private:
    StorageOrigins *out_;
  };
  Collector collector(&origins);
  collector(body);
  // Block annotations also name the handle parameter; map them to the data Var
  // the conflict test compares, the same way the attribute path does.
  for (const VarNode *var : std::set<const VarNode *>(origins.non_restrict)) {
    if (auto buffer = buffer_map.Get(GetRef<Var>(var))) {
      origins.non_restrict.insert(buffer.value()->data.get());
    }
  }
  return origins;
}

bool MayConflictStorage(const BufferNode *lhs, const BufferNode *rhs,
                        const StorageOrigins &origins) {
  const VarNode *lhs_var = lhs->data.get();
  const VarNode *rhs_var = rhs->data.get();
  if (lhs_var == rhs_var) {
    return true;
  }
  // Two real allocations are distinct storage by construction.
  if (origins.allocations.count(lhs_var) != 0 &&
      origins.allocations.count(rhs_var) != 0) {
    return false;
  }
  // Two parameters are distinct only under an explicit no-alias contract, and
  // even then a non-restrict parameter may alias any other storage.
  if (origins.parameters.count(lhs_var) != 0 &&
      origins.parameters.count(rhs_var) != 0) {
    if (!origins.no_alias) {
      return true;
    }
    return origins.non_restrict.count(lhs_var) != 0 ||
           origins.non_restrict.count(rhs_var) != 0;
  }
  // An allocation against a parameter, or a pointer with no origin we can name,
  // stays conservatively conflicting.
  return true;
}

bool CollectStores(const Stmt &stmt,
                   std::vector<const BufferStoreNode *> *out) {
  if (const auto *store = stmt.as<BufferStoreNode>()) {
    out->push_back(store);
    return true;
  }
  if (pto::IsNoOpStmt(stmt)) {
    return true;
  }
  if (const auto *seq = stmt.as<SeqStmtNode>()) {
    for (const Stmt &s : seq->seq) {
      if (!CollectStores(s, out)) {
        return false;
      }
    }
    return true;
  }
  return false;
}

} // namespace pto
} // namespace tl
} // namespace tvm
