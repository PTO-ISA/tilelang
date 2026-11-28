// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file vectorize_parallel_to_pto.cc
 * \brief VectorizeParallelToPTO: rewrite verified T.Parallel loops inside
 * PTO SIMD_VF blocks into logical VMI code.
 *
 * Consumes the pto_parallel_loop_layout fragment written by
 * LayoutInference and verified by VerifyParallelToPTO. Per parallel unit:
 *   1. invert the lane fragment through the shared helper (the inverse
 *      layout's own Forward result is the single source of truth);
 *   2. build one chunk loop q in [0, ceil(E/L)), mapping the logical
 *      index to q * L + lane;
 *   3. classify every access via the shared MemoryAccess analysis
 *      (continuous -> vload/vstore via tl.access_ptr(start, L, rw);
 *      lane-uniform load -> scalar BufferLoad; indexed load/store and
 *      lane-uniform store -> masked gather/scatter with element offsets);
 *   4. convert value expressions (value roles tracked by result dtype);
 *      every VMI with a return value is bound through EmitBind before use;
 *   5. delete the consumed layout annotation, keep the SIMD_VF block
 *      annotations, and leave no unconverted Parallel behind.
 *
 * Supported shapes: 1D parallel units and two-layer adjacent Parallel nests
 * (2D — the nest converts as one unit with the layout
 * annotation on its outermost For); sequential units allowed; static
 * extents with full or tail masks. A third parallel layer or any other
 * unsupported structure is rejected by Verify.
 *
 * Non-divisible extents use a per-chunk mask in the same loop. Continuous
 * vload is unmasked: the caller must provide initialized readable padding
 * for every complete logical vector. Indexed gather and scatter consume
 * the per-chunk mask directly to limit memory accesses to active lanes.
 */

#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <sstream>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "buffer_version.h"
#include "parallel_to_pto_utils.h"
#include "transform/common/storage_size.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;
namespace pto_analysis = ::tvm::tl::pto;

namespace {

DataType VectorDType(DataType elem, int64_t lanes) {
  return elem.with_lanes(static_cast<int>(lanes));
}

const Op &VmiOp(const std::string &tail) { return Op::Get("tl.vmi." + tail); }

// This version keeps local DCE in Vectorize so
// its directly emitted TIR/PTODSL is cleaner and easier to inspect.
// Cleanup may later move to PTOAS canonicalization/DCE to reduce
// TileLang-specific code and maintenance; clean-output checks would then
// target the optimized PTOAS IR. The emitted IR must be valid and preserve
// program semantics even before DCE.
//
// Dead vector-chain elimination on one converted chunk body. The
// conversion can leave Bind statements whose Var is never referenced:
// source binds consumed only through the address alias environment
// (bind_env_) never use their converted Var, and an unused source bind
// strands its whole value chain (vci/vbrc/vadd/vload).
//
// The traversal is a scope-aware *reverse liveness* pass — a
// dead Bind must not survive just because it was wrapped in an AttrStmt
// or appears as the only statement of a body. The rules are:
//   1. A SeqStmt is processed back to front; `live` always holds the
//      vars everything *after* the current statement needs.
//   2. A Bind's LHS var is a definition, never a use. When the result is
//      not live and the RHS is discardable, the Bind is deleted and its
//      RHS inputs do NOT become live (so the dead chain collapses in one
//      pass); otherwise it is kept and its RHS inputs join `live`.
//   3. An AttrStmt body is cleaned in its own nested scope with a fresh
//      liveness set; only the vars actually referenced inside propagate
//      outward. Attribute node/value references are always uses, and the
//      attribute structure and evaluation position are preserved.
//   4. A bare Bind follows the same rules; when everything dies the
//      remains are a legal no-op (Evaluate(0)).
// Nothing is merged, reordered or otherwise rescheduled;
// referenced loads stay exactly where they were (vecadd keeps its three
// vloads), and Bind-free (frontend-inlined) bodies are untouched.
class DeadBindElim {
public:
  Stmt operator()(const Stmt &stmt) {
    std::set<const VarNode *> live;
    Stmt out = Run(stmt, &live);
    return out.defined() ? out : NoOp();
  }

private:
  static Stmt NoOp() { return Evaluate(make_zero(DataType::Int(32))); }

  static void CollectExprVars(const PrimExpr &expr,
                              std::set<const VarNode *> *live) {
    PostOrderVisit(expr, [&](const ObjectRef &n) {
      if (const auto *v = n.as<VarNode>()) {
        live->insert(v);
      }
    });
  }

  static void CollectStmtVars(const Stmt &stmt,
                              std::set<const VarNode *> *live) {
    PostOrderVisit(stmt, [&](const ObjectRef &n) {
      if (const auto *v = n.as<VarNode>()) {
        live->insert(v);
      }
    });
  }

  static bool DiscardableRhs(const PrimExpr &rhs) {
    // Positive "allowed to discard" set: a Call may disappear
    // only when it is one of the tl.vmi.* ops this pass actually emits as
    // a Bind RHS and that is pure/read-only by construction — vload/vgather
    // (reads) and arithmetic, select or conversion ops (pure), enumerated from
    // the ConvertUnit/VmiOp call sites — or the pure tl.access_ptr address
    // descriptor nested inside a load operand (already whitelisted as a
    // pure index intrinsic by IsSupportedAddressExpr). vstore never
    // appears as a Bind RHS; any other Call — call_extern, other tl.* or
    // an unknown op — keeps its Bind. Plain UB BufferLoads are
    // discardable: they are ordinary reads with no write effect (the
    // scalar "sload" binds). That "ordinary" premise is enforced by
    // VerifyParallelToPTO, not assumed here: accesses to
    // tirx.volatile-annotated allocations and volatile_scope attributes
    // are rejected there, and opaque calls are already excluded
    // from addresses.
    static const std::set<std::string> kDiscardable = {
        "tl.vmi.vload",     "tl.vmi.vgather",
        "tl.vmi.vbrc",      "tl.vmi.vci",
        "tl.vmi.vadd",      "tl.vmi.vsub",
        "tl.vmi.vmul",      "tl.vmi.vdiv",
        "tl.vmi.vfloordiv", "tl.vmi.vfloormod",
        "tl.vmi.vabs",      "tl.vmi.vexp",
        "tl.vmi.vln",       "tl.vmi.vsqrt",
        "tl.vmi.vcvt",      "tl.vmi.vinterpret_cast",
        "tl.vmi.vcmp",      "tl.vmi.vcmps",
        "tl.vmi.vsel",      "tl.vmi.vand",
        "tl.vmi.vor",       "tl.vmi.vnot",
        "tl.vmi.vshl",      "tl.vmi.vshr",
        "tl.vmi.vshls",     "tl.vmi.vshrs",
        "tl.vmi.vxor",      "tl.access_ptr",
    };
    bool discardable = true;
    PostOrderVisit(rhs, [&](const ObjectRef &n) {
      if (const auto *call = n.as<CallNode>()) {
        const auto *op = call->op.as<OpNode>();
        const std::string name = op != nullptr ? op->name : "";
        if (kDiscardable.count(name) == 0 &&
            pto_analysis::GetPtoBitOp(GetRef<Call>(call)) ==
                pto_analysis::PtoBitOp::kNone) {
          discardable = false;
        }
      }
    });
    return discardable;
  }

  // Returns the kept statement, or an *undefined* Stmt when the statement
  // was deleted (only a Bind can be deleted). `live` is threaded from
  // later statements back to earlier ones.
  Stmt Run(const Stmt &stmt, std::set<const VarNode *> *live) {
    if (const auto *bind = stmt.as<BindNode>()) {
      if (live->count(bind->var.get()) == 0 && DiscardableRhs(bind->value)) {
        // Dead and safe to drop: the RHS inputs stay out of `live`, so a
        // chain feeding only this Bind collapses in the same pass.
        return Stmt();
      }
      // Kept: classic liveness — subtract the definition, add the uses.
      live->erase(bind->var.get());
      CollectExprVars(bind->value, live);
      return stmt;
    }
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      Array<Stmt> kept;
      for (auto it = seq->seq.rbegin(); it != seq->seq.rend(); ++it) {
        Stmt sub = Run(*it, live);
        if (sub.defined()) {
          kept.push_back(sub);
        }
      }
      if (kept.empty()) {
        return NoOp();
      }
      // std::reverse does not accept the ffi Array iterators on this
      // toolchain; re-append in reverse instead.
      Array<Stmt> forward;
      for (auto it = kept.rbegin(); it != kept.rend(); ++it) {
        forward.push_back(*it);
      }
      return SeqStmt::Flatten(forward);
    }
    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      // Nested scope: clean the body with a fresh liveness set, then
      // propagate only the references that survive to its entry (external
      // uses; vars defined and used only inside never escape because the
      // converted Vars are unique objects). Attribute node/value
      // references are always uses and keep the structure.
      std::set<const VarNode *> body_live;
      Stmt body = Run(attr->body, &body_live);
      if (!body.defined()) {
        body = NoOp();
      }
      live->insert(body_live.begin(), body_live.end());
      if (auto ne = attr->node.try_cast<ObjectRef>()) {
        if (const auto *pe = ne.value().as<PrimExprNode>()) {
          CollectExprVars(GetRef<PrimExpr>(pe), live);
        }
      }
      CollectExprVars(attr->value, live);
      if (body.same_as(attr->body)) {
        return stmt;
      }
      return AttrStmt(attr->node, attr->attr_key, attr->value, body);
    }
    // Any other statement is always kept; all its Var occurrences are
    // uses.
    CollectStmtVars(stmt, live);
    return stmt;
  }
};

// Follow a finalize destination backwards from its consumers to find writes
// that must leave a runtime loop. The entering value survives zero trips.
class ReducerLoopResults : public StmtExprMutator {
public:
  explicit ReducerLoopResults(Buffer slot) : slot_(std::move(slot)) {}
  bool NeedsInitialValue() const { return has_carry_ && live_; }

private:
  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    if (op->buffer->data.same_as(slot_->data))
      live_ = true;
    return StmtExprMutator::VisitExpr_(op);
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    if (op->buffer->data.same_as(slot_->data)) {
      live_ = false;
      ++writes_;
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const SeqStmtNode *op) final {
    Array<Stmt> seq = op->seq;
    for (size_t i = seq.size(); i > 0; --i)
      seq.Set(i - 1, VisitStmt(seq[i - 1]));
    return SeqStmt::Flatten(seq);
  }

  Stmt VisitStmt_(const IfThenElseNode *op) final {
    const bool after = live_;
    Stmt then_case = VisitStmt(op->then_case);
    const bool then_live = live_;
    live_ = after;
    Optional<Stmt> else_case;
    if (op->else_case.defined())
      else_case = VisitStmt(op->else_case.value());
    live_ |= then_live;
    return IfThenElse(VisitExpr(op->condition), then_case, else_case);
  }

  Stmt VisitStmt_(const ForNode *op) final {
    const bool after = live_;
    const size_t before = writes_;
    For loop = Downcast<For>(StmtExprMutator::VisitStmt_(op));
    live_ |= after;
    if (after && writes_ != before) {
      Array<Var> slots;
      if (auto old = loop->annotations.Get(attr::kPtoReducerLoopResults))
        slots = Downcast<Array<Var>>(old.value());
      slots.push_back(slot_->data);
      loop.CopyOnWrite()->annotations.Set(attr::kPtoReducerLoopResults, slots);
      has_carry_ = true;
    }
    return loop;
  }

  Buffer slot_;
  size_t writes_{0};
  bool live_{false};
  bool has_carry_{false};
};

class VectorizeParallelToPTOImpl : public StmtExprMutator {
public:
  static PrimFunc Substitute(PrimFunc f) {
    VectorizeParallelToPTOImpl mutator;
    for (const Var &parameter : f->params) {
      mutator.parameters_.insert(parameter);
    }
    mutator.storage_.Build(f->body);
    mutator.storage_.ValidateOwnerUses(f->body);
    PrimFuncNode *fptr = f.CopyOnWrite();
    fptr->body = mutator.VisitStmt(f->body);
    class RemoveOld : public StmtMutator {
    public:
      explicit RemoveOld(const pto::ParallelStorageQuery &q) : q(q) {}
      Stmt VisitStmt_(const SBlockNode *op) final {
        SBlock b = Downcast<SBlock>(StmtMutator::VisitStmt_(op));
        Array<Buffer> kept;
        for (const Buffer &buf : b->alloc_buffers)
          if (!q.IsPrivate(buf->data))
            kept.push_back(buf);
        b.CopyOnWrite()->alloc_buffers = kept;
        return b;
      }
      Stmt VisitStmt_(const AllocBufferNode *op) final {
        if (q.IsPrivate(op->buffer->data))
          return Evaluate(0);
        return GetRef<Stmt>(op);
      }
      const pto::ParallelStorageQuery &q;
    };
    fptr->body = RemoveOld(mutator.storage_)(fptr->body);
    return GetRef<PrimFunc>(fptr);
  }

private:
  // AllocBuffer is a flat definition visible to later siblings. Every other
  // statement restores the owner environment on leaving its lexical scope.
  Stmt VisitStmt(const Stmt &stmt) final {
    if (stmt.as<AllocBufferNode>()) {
      return StmtExprMutator::VisitStmt(stmt);
    }
    const size_t allocation_mark = active_allocations_.size();
    Stmt result = StmtExprMutator::VisitStmt(stmt);
    active_allocations_.resize(allocation_mark);
    return result;
  }

  Stmt VisitStmt_(const SBlockNode *op) final {
    for (const Buffer &buffer : op->alloc_buffers) {
      active_allocations_.push_back(buffer);
    }
    if (op->name_hint != "SIMD_VF") {
      return StmtExprMutator::VisitStmt_(op);
    }
    // Shared region scan: the same implementation Verify uses, so
    // skip/reject/convert decisions cannot drift apart.
    const Stmt &region = op->body;
    pto_analysis::PtoRegionScan scan = pto_analysis::ScanPtoRegion(region);
    if (!scan.has_any_parallel && !scan.has_direct_reduce && !scan.has_fill) {
      // Pure hand-written VMI region (possibly with control flow): skipped
      // by both passes; no lanes annotation needed.
      return StmtExprMutator::VisitStmt_(op);
    }
    ICHECK(!scan.has_unsupported_wrapper)
        << "[VectorizeParallelToPTO] internal error: unsupported wrapper "
           "structure passed Verify: "
        << scan.issue;

    auto saved_reducers = reducers_;
    auto saved_buffers = vf_buffers_;
    auto saved_bind = bind_env_;
    auto saved_values = value_map_;
    auto saved_infos = value_infos_;
    auto saved_masks = full_masks_;
    auto saved_prelude = mask_prelude_;
    int64_t saved_lanes = lanes_;
    reducers_ = pto::CollectReducerStates(GetRef<SBlock>(op), storage_);
    lanes_ = reducers_.lanes;
    vf_buffers_.clear();
    full_masks_.clear();
    mask_prelude_.clear();
    for (const auto &[data, info] : reducers_.storage_values) {
      if (!info.IsVmi())
        continue;
      Buffer old = storage_.Allocation(data);
      vf_buffers_.emplace(
          old->data,
          decl_buffer(old->shape, old->dtype.with_lanes(info.logical_lanes),
                      old->name + "_v",
                      info.logical_lanes == 1 ? "local.var" : "local"));
    }
    Stmt body = ConvertRegion(region);
    std::unordered_set<const VarNode *> visited_results;
    for (const auto &site : reducers_.finalizes) {
      const Buffer &slot = vf_buffers_.at(site->result->buffer->data);
      if (!visited_results.insert(slot->data.get()).second)
        continue;
      ReducerLoopResults carry(slot);
      body = carry(body);
      if (carry.NeedsInitialValue()) {
        // vreg declares a type. The runtime loop needs an entering SSA value.
        auto zero = ToVmiWidth(make_zero(slot->dtype), 1, &mask_prelude_);
        mask_prelude_.push_back(BufferStore(slot, zero, {Integer(0)}));
      }
    }
    const auto residual = pto_analysis::ScanPtoRegion(body);
    ICHECK(!residual.has_direct_reduce && !residual.has_fill)
        << "[VectorizeParallelToPTO] direct tileops "
           "must be standalone Evaluate(Call) statements";
    if (!vf_buffers_.empty()) {
      ICHECK(!UsesVar(body,
                      [&](const VarNode *var) {
                        return vf_buffers_.count(GetRef<Var>(var));
                      }))
          << "[VectorizeParallelToPTO] lowered body still references a "
             "replaced reducer allocation";
    }
    auto wrap = [&](Stmt inner) {
      inner = SeqStmt::Flatten(mask_prelude_, inner);
      for (const auto &[old, replacement] : vf_buffers_) {
        inner = AttrStmt(
            replacement->data, attr::kVmiAllocKind,
            StringImm(replacement->dtype.lanes() == 1 ? "data" : "data_array"),
            inner);
      }
      return inner;
    };
    if (const auto *scope = body.as<AttrStmtNode>();
        scope && scope->attr_key == "tl.simdvf_scope")
      body = AttrStmt(scope->node, scope->attr_key, scope->value,
                      wrap(scope->body));
    else
      body = wrap(body);
    Array<Buffer> allocations;
    for (const auto &old : op->alloc_buffers) {
      auto it = vf_buffers_.find(old->data);
      allocations.push_back(it == vf_buffers_.end() ? old : it->second);
    }
    auto annotations = op->annotations;
    annotations.erase(attr::kPtoReducerStates);
    Stmt result =
        SBlock(op->iter_vars, op->reads, op->writes, op->name_hint, body,
               op->init, allocations, op->match_buffers, annotations);
    reducers_ = std::move(saved_reducers);
    vf_buffers_ = std::move(saved_buffers);
    bind_env_ = std::move(saved_bind);
    value_map_ = std::move(saved_values);
    value_infos_ = std::move(saved_infos);
    full_masks_ = std::move(saved_masks);
    mask_prelude_ = std::move(saved_prelude);
    lanes_ = saved_lanes;
    return result;
  }

  Var GetFullMask(int64_t lanes) {
    auto it = full_masks_.find(lanes);
    if (it != full_masks_.end())
      return it->second;
    Var mask("full_mask_" + std::to_string(lanes), DataType::Bool(lanes));
    mask_prelude_.push_back(
        Bind(mask, Call(mask.dtype(), VmiOp("create_mask"), {Integer(lanes)},
                        {{"size", Integer(lanes)}})));
    full_masks_.emplace(lanes, mask);
    return mask;
  }

  pto::ValueInfo Info(const PrimExpr &value) const {
    if (const auto *var = value.as<VarNode>()) {
      auto it = value_infos_.find(GetRef<Var>(var));
      if (it != value_infos_.end())
        return it->second;
    }
    return value.dtype().lanes() > 1 ? pto::ValueInfo::Vmi(value.dtype())
                                     : pto::GetValueInfo(value, value_infos_);
  }

  PrimExpr ToVmiWidth(const PrimExpr &source, int64_t width, Array<Stmt> *pre) {
    auto info = Info(source);
    const auto adaptation = pto::PlanWidthAdaptation(info, width);
    if (adaptation == pto::WidthAdaptation::kKeep)
      return source;
    pto::CheckBroadcastDType(source.dtype().element_of(),
                             "PTO value broadcast");
    Map<String, ObjectRef> attrs{{"size", Integer(width)}};
    if (adaptation == pto::WidthAdaptation::kBroadcastOneLane)
      attrs.Set("group", Integer(1));
    return EmitBind(
        "broadcast",
        Call(source.dtype().with_lanes(width), VmiOp("vbrc"), {source}, attrs),
        pre);
  }

  PrimExpr ReadRegionValue(const PrimExpr &source, Array<Stmt> *pre) {
    if (const auto *load = source.as<BufferLoadNode>()) {
      auto it = vf_buffers_.find(load->buffer->data);
      if (it != vf_buffers_.end()) {
        Var value = EmitBind("result_snapshot",
                             BufferLoad(it->second, load->indices), pre);
        value_infos_[value] = pto::ValueInfo::Vmi(it->second->dtype);
        return value;
      }
      return EmitBind("scalar_snapshot", source, pre);
    }
    if (const auto *var = source.as<VarNode>()) {
      if (auto mapped = value_map_.Get(GetRef<Var>(var)))
        return mapped.value();
    }
    return source;
  }

  Stmt ConvertPtoFinalize(const PtoFinalizeReducerOp &op) {
    pto::AnalyzePtoFinalize(op, reducers_);
    Array<Stmt> out;
    Buffer acc = vf_buffers_.at(op->accumulator->buffer->data);
    auto value =
        EmitBind("partial_snapshot", BufferLoad(acc, {Integer(0)}), &out);
    const char *names[] = {"vcadd", "vcmax", "vcmin"};
    Map<String, ObjectRef> attrs{{"group", Integer(1)}};
    if (op->op == ReducerV2OpType::kSum)
      attrs.Set("reassoc", Bool(true));
    PrimExpr result = EmitBind("reduced_one",
                               Call(acc->dtype.element_of(),
                                    VmiOp(names[static_cast<int>(op->op)]),
                                    {value, GetFullMask(lanes_)}, attrs),
                               &out);
    if (op->seed.defined()) {
      auto seed = ReadRegionValue(
          BufferLoad(op->seed.value()->buffer, {Integer(0)}), &out);
      seed = ToVmiWidth(seed, 1, &out);
      const char *combine[] = {"vadd", "vmax", "vmin"};
      result = EmitBind("seed_combine",
                        Call(result.dtype(),
                             VmiOp(combine[static_cast<int>(op->op)]),
                             {result, seed, GetFullMask(1)}),
                        &out);
    }
    out.push_back(BufferStore(vf_buffers_.at(op->result->buffer->data), result,
                              {Integer(0)}));
    return SeqStmt::Flatten(out);
  }

  Stmt LowerDirectReduce(const Call &call, int64_t lanes) {
    pto_analysis::PtoDirectReducePlan plan =
        pto_analysis::PlanPtoDirectReduce(call, analyzer_, buffer_versions_);
    bool absolute = plan.operation == "abssum" || plan.operation == "absmax";
    bool sum = plan.operation == "sum" || plan.operation == "abssum";
    bool maximum = plan.operation == "max" || plan.operation == "absmax";
    auto i32 = [](int64_t value) { return IntImm(DataType::Int(32), value); };
    DataType vector_dtype = plan.accumulator_dtype.with_lanes(lanes);
    DataType mask_dtype = DataType::Bool(lanes);
    Buffer accumulator =
        tirx::decl_buffer({i32(1)}, vector_dtype, "reduce_acc", "local.var");
    Var chunk("reduce_chunk", DataType::Int(32));
    Var output("reduce_output", DataType::Int(32));
    Var reduction("reduce_index", DataType::Int(32));
    Var full_mask("reduce_domain", mask_dtype);
    Var active_mask("reduce_active", mask_dtype);
    auto mask = [&](const PrimExpr &active) {
      return Call(mask_dtype, VmiOp("create_mask"), {active},
                  {{"size", i32(lanes)}});
    };
    auto pointer = [&](const BufferRegion &region,
                       const Array<PrimExpr> &indices, int64_t extent, int rw) {
      return Call(DataType::Handle(), tl::access_ptr(),
                  {BufferLoad(region->buffer, indices), i32(extent), i32(rw)});
    };
    auto source_indices = [&](const PrimExpr &out, const PrimExpr &red) {
      Array<PrimExpr> indices;
      for (const Range &range : plan.source->region) {
        indices.push_back(range->min);
      }
      indices.Set(plan.source_prefix + plan.reduction_axis, red);
      if (plan.logical_rank == 2) {
        indices.Set(plan.source_prefix + plan.output_axis, out);
      }
      return indices;
    };
    auto destination_indices = [&](const PrimExpr &out) {
      Array<PrimExpr> indices;
      for (const Range &range : plan.destination->region) {
        indices.push_back(range->min);
      }
      indices.Set(plan.destination_output_axis, out);
      return indices;
    };
    auto load = [&](const PrimExpr &out, const PrimExpr &red,
                    Array<Stmt> *statements) {
      DataType src_dtype = plan.source->buffer->dtype;
      int64_t footprint =
          ((lanes + 2048 / src_dtype.bits() - 1) / (2048 / src_dtype.bits())) *
          (2048 / src_dtype.bits());
      PrimExpr value = EmitBind(
          "reduce_load",
          Call(src_dtype.with_lanes(lanes), VmiOp("vload"),
               {pointer(plan.source, source_indices(out, red), footprint, 1),
                i32(0)},
               {{"size", i32(lanes)}}),
          statements);
      if (src_dtype != plan.accumulator_dtype) {
        value =
            EmitBind("reduce_cast",
                     Call(vector_dtype, VmiOp("vcvt"), {value},
                          {{"to_dtype",
                            StringImm(TvmDtypeName(plan.accumulator_dtype))}}),
                     statements);
      }
      if (absolute && !plan.accumulator_dtype.is_uint()) {
        value =
            EmitBind("reduce_abs", Call(vector_dtype, VmiOp("vabs"), {value}),
                     statements);
      }
      return value;
    };
    auto read_accumulator = [&]() { return BufferLoad(accumulator, {i32(0)}); };
    const char *fold_name = sum ? "vadd" : maximum ? "vmax" : "vmin";
    auto seed = [&](const PrimExpr &first, const PrimExpr &active,
                    Array<Stmt> *statements) -> PrimExpr {
      if (plan.operation != "absmax") {
        return first;
      }
      // Match the common Ascend/CUDA reducer's zero-initialized partial
      PrimExpr zero = EmitBind("reduce_zero",
                               Call(vector_dtype, VmiOp("vbrc"),
                                    {make_zero(plan.accumulator_dtype)},
                                    {{"size", i32(lanes)}}),
                               statements);
      return EmitBind("reduce_seed",
                      Call(vector_dtype, VmiOp("vmax"), {zero, first, active}),
                      statements);
    };
    Array<Stmt> body;
    if (plan.reduce_across_lanes) {
      PrimExpr first = load(output, i32(0), &body);
      first = seed(first, full_mask, &body);
      body.push_back(BufferStore(accumulator, first, {i32(0)}));
      auto accumulate = [&](const PrimExpr &red, const PrimExpr &active,
                            bool keep_inactive) {
        Array<Stmt> update;
        PrimExpr next = load(output, red, &update);
        PrimExpr folded = EmitBind("reduce_fold",
                                   Call(vector_dtype, VmiOp(fold_name),
                                        {read_accumulator(), next, active}),
                                   &update);
        if (keep_inactive) {
          folded = EmitBind("reduce_merge",
                            Call(vector_dtype, VmiOp("vsel"),
                                 {active, folded, read_accumulator()}),
                            &update);
        }
        update.push_back(BufferStore(accumulator, folded, {i32(0)}));
        return SeqStmt::Flatten(update);
      };
      int64_t full_chunks = plan.reduction_extent / lanes;
      int64_t tail_lanes = plan.reduction_extent % lanes;
      // The seed already covers the first chunk. Complete chunks need no
      // remaining-lane calculation or merge with the previous accumulator.
      if (full_chunks > 1) {
        body.push_back(For(chunk, i32(1), i32(full_chunks - 1),
                           ForKind::kSerial,
                           accumulate(chunk * i32(lanes), full_mask, false)));
      }
      if (full_chunks > 0 && tail_lanes != 0) {
        body.push_back(Bind(active_mask, mask(i32(tail_lanes))));
        body.push_back(accumulate(i32(full_chunks * lanes), active_mask, true));
      }
      ffi::Map<ffi::String, ObjectRef> annotations;
      if (sum && plan.accumulator_dtype.is_float()) {
        annotations.Set("reassoc", IntImm(DataType::Bool(), 1));
      }
      const char *reduce_name = sum ? "vcadd" : maximum ? "vcmax" : "vcmin";
      PrimExpr result =
          EmitBind("reduce_result",
                   Call(plan.accumulator_dtype, VmiOp(reduce_name),
                        {read_accumulator(), full_mask}, annotations),
                   &body);
      if (!plan.clear) {
        PrimExpr previous = EmitBind(
            "reduce_previous",
            Call(plan.accumulator_dtype, VmiOp("vload"),
                 {pointer(plan.destination, destination_indices(output), 1, 1),
                  i32(0)},
                 {{"size", i32(1)}}),
            &body);
        PrimExpr one = EmitBind("reduce_one",
                                Call(DataType::Bool(), VmiOp("create_mask"),
                                     {i32(1)}, {{"size", i32(1)}}),
                                &body);
        // Merge the previous output only after the cross-lane reduction;
        // seeding every lane with it would count it VL times for sum.
        result = EmitBind("reduce_update",
                          Call(plan.accumulator_dtype, VmiOp(fold_name),
                               {previous, result, one}),
                          &body);
      }
      body.push_back(Evaluate(Call(
          DataType::Void(), VmiOp("vstore"),
          {result, pointer(plan.destination, destination_indices(output), 1, 2),
           i32(0)})));
      Stmt output_loop = For(output, i32(0), i32(plan.output_extent),
                             ForKind::kSerial, SeqStmt::Flatten(body));
      Stmt compute = SeqStmt::Flatten(
          Bind(full_mask, mask(i32(std::min(plan.reduction_extent, lanes)))),
          output_loop);
      return SeqStmt::Flatten(AllocBuffer(accumulator),
                              AttrStmt(accumulator->data, "tl.vmi.alloc_kind",
                                       StringImm("data"), compute));
    }
    body.push_back(
        Bind(active_mask, mask(i32(plan.output_extent) - chunk * i32(lanes))));
    PrimExpr first = load(chunk * i32(lanes), i32(0), &body);
    first = seed(first, active_mask, &body);
    body.push_back(BufferStore(accumulator, first, {i32(0)}));
    Array<Stmt> update;
    PrimExpr next = load(chunk * i32(lanes), reduction, &update);
    PrimExpr folded = EmitBind("reduce_fold",
                               Call(vector_dtype, VmiOp(fold_name),
                                    {read_accumulator(), next, active_mask}),
                               &update);
    update.push_back(BufferStore(accumulator, folded, {i32(0)}));
    if (plan.reduction_extent > 1) {
      body.push_back(For(reduction, i32(1), i32(plan.reduction_extent - 1),
                         ForKind::kSerial, SeqStmt::Flatten(update)));
    }
    if (!plan.clear) {
      int64_t physical_lanes = 2048 / plan.accumulator_dtype.bits();
      int64_t footprint =
          ((lanes + physical_lanes - 1) / physical_lanes) * physical_lanes;
      PrimExpr previous = EmitBind(
          "reduce_previous",
          Call(vector_dtype, VmiOp("vload"),
               {pointer(plan.destination,
                        destination_indices(chunk * i32(lanes)), footprint, 1),
                i32(0)},
               {{"size", i32(lanes)}}),
          &body);
      PrimExpr combined =
          EmitBind("reduce_update",
                   Call(vector_dtype, VmiOp(fold_name),
                        {previous, read_accumulator(), active_mask}),
                   &body);
      body.push_back(BufferStore(accumulator, combined, {i32(0)}));
    }
    body.push_back(Evaluate(
        Call(DataType::Void(), VmiOp("vstore"),
             {read_accumulator(),
              pointer(plan.destination, destination_indices(chunk * i32(lanes)),
                      lanes, 2),
              i32(0), active_mask})));
    // Full chunks and the tail use the same vector reduction. Loads may read
    // complete physical lanes; active_mask limits the fold and output store.
    int64_t chunks = (plan.output_extent + lanes - 1) / lanes;
    Stmt compute = For(chunk, i32(0), i32(chunks), ForKind::kSerial,
                       SeqStmt::Flatten(body));
    return SeqStmt::Flatten(AllocBuffer(accumulator),
                            AttrStmt(accumulator->data, "tl.vmi.alloc_kind",
                                     StringImm("data"), compute));
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tl::attr::kBufferVersion) {
      BufferVersionMap saved = buffer_versions_;
      for (const auto &[storage, count] : op->node.cast<BufferVersionMap>()) {
        buffer_versions_.Set(storage, count);
      }
      Stmt result = StmtExprMutator::VisitStmt_(op);
      buffer_versions_ = std::move(saved);
      return result;
    }
    if (op->attr_key == tirx::attr::thread_extent) {
      if (const auto *iv = op->node.as<IterVarNode>()) {
        With<arith::ConstraintContext> launch_bounds(
            analyzer_, iv->var >= 0 && iv->var < op->value);
        return StmtExprMutator::VisitStmt_(op);
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const AllocBufferNode *op) final {
    active_allocations_.push_back(op->buffer);
    return GetRef<Stmt>(op);
  }

  bool IsScalarFillParameter(const PrimExpr &value) const {
    if (value.dtype() != DataType::Float(16) &&
        value.dtype() != DataType::Float(32)) {
      return false;
    }
    if (const auto *parameter = value.as<VarNode>()) {
      return parameters_.count(GetRef<Var>(parameter)) != 0;
    }
    if (const auto *conversion = value.as<CastNode>()) {
      return IsScalarFillParameter(conversion->value);
    }
    return false;
  }

  Stmt LowerFill(const Call &call, int64_t lanes) {
    pto_analysis::PtoFillPlan plan =
        pto_analysis::PlanPtoFill(call, analyzer_, buffer_versions_);
    auto i32 = [](int64_t value) { return IntImm(DataType::Int(32), value); };
    Buffer dst = plan.destination->buffer;
    DataType dtype = dst->dtype;
    Array<Stmt> statements;
    // Capture once at the source statement, before any destination store.
    // Constants stay literals so the VMI bridge retains their exact dtype.
    PrimExpr value = plan.value;
    bool scalar_parameter = IsScalarFillParameter(value);
    PrimExpr load_value = value;
    std::vector<DataType> float_casts;
    if (!scalar_parameter) {
      while (const auto *conversion = load_value.as<CastNode>()) {
        if (!(conversion->dtype.is_float() ||
              conversion->dtype.is_bfloat16()) ||
            !(conversion->value.dtype().is_float() ||
              conversion->value.dtype().is_bfloat16())) {
          break;
        }
        float_casts.push_back(conversion->dtype);
        load_value = conversion->value;
      }
    }
    if (!load_value.as<BufferLoadNode>() && !load_value.as<VarNode>()) {
      // Integer casts and constants retain their scalar lowering. Floating
      // UB snapshots and parameters use vector conversions below, preserving
      // every intermediate rounding rather than collapsing a cast chain.
      load_value = value;
      float_casts.clear();
    }
    if (const auto *load = load_value.as<BufferLoadNode>()) {
      // Capture the scalar once before any store; a later store may write
      // the same element from which this uniform value was loaded.
      PrimExpr address = Call(DataType::Handle(), tl::access_ptr(),
                              {GetRef<BufferLoad>(load), i32(1), i32(1)});
      value = EmitBind("fill_value",
                       Call(load_value.dtype(), VmiOp("vload"),
                            {address, i32(0)}, {{"size", i32(1)}}),
                       &statements);
    } else if (!scalar_parameter && !load_value.as<IntImmNode>() &&
               !load_value.as<FloatImmNode>()) {
      value = EmitBind("fill_value", load_value, &statements);
    }
    DataType broadcast_dtype = float_casts.empty() ? dtype : load_value.dtype();
    PrimExpr vector =
        EmitBind("fill_vector",
                 Call(broadcast_dtype.with_lanes(lanes), VmiOp("vbrc"), {value},
                      {{"size", i32(lanes)}}),
                 &statements);
    for (auto conversion = float_casts.rbegin();
         conversion != float_casts.rend(); ++conversion) {
      // Keep the scalar snapshot in its loaded dtype. PTOAS has no complete
      // layout route for narrow one-lane vcvt -> broadcast -> wide store;
      // broadcasting the uniform input before vcvt preserves the same cast
      // per element and uses the validated full-VL conversion route.
      vector =
          EmitBind("fill_cast",
                   Call(conversion->with_lanes(lanes), VmiOp("vcvt"), {vector},
                        {{"to_dtype", StringImm(TvmDtypeName(*conversion))}}),
                   &statements);
    }
    PrimExpr mask = EmitBind("fill_mask",
                             Call(DataType::Bool(lanes), VmiOp("create_mask"),
                                  {i32(lanes)}, {{"size", i32(lanes)}}),
                             &statements);
    Var row("fill_row", DataType::Int(32));
    Var chunk("fill_chunk", DataType::Int(32));
    Array<PrimExpr> indices;
    for (const Range &range : plan.destination->region) {
      indices.push_back(range->min);
    }
    if (plan.outer_axis >= 0) {
      indices.Set(plan.outer_axis, row);
    }
    indices.Set(plan.continuous_axis, chunk * i32(lanes));
    PrimExpr pointer = Call(DataType::Handle(), tl::access_ptr(),
                            {BufferLoad(dst, indices), i32(lanes), i32(2)});
    Stmt body = Evaluate(Call(DataType::Void(), VmiOp("vstore"),
                              {vector, pointer, i32(0), mask}));
    body = For(chunk, i32(0), i32(plan.run_extent / lanes), ForKind::kSerial,
               body);
    if (plan.outer_axis >= 0) {
      body = For(row, i32(0), i32(plan.outer_extent), ForKind::kSerial, body);
    }
    statements.push_back(body);
    return SeqStmt::Flatten(statements);
  }

  Stmt VisitStmt_(const ForNode *op) final {
    pto_analysis::ScopedPtoLoopBounds loop_bounds(analyzer_, GetRef<For>(op));
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt ConvertScopedRegion(const Stmt &body) {
    auto saved_bind = bind_env_;
    auto saved_values = value_map_;
    auto saved_infos = value_infos_;
    Stmt result = ConvertRegion(body);
    bind_env_ = std::move(saved_bind);
    value_map_ = std::move(saved_values);
    value_infos_ = std::move(saved_infos);
    return result;
  }

  Stmt ConvertRegion(const Stmt &stmt) {
    struct AllocationScope {
      std::vector<Buffer> &allocations;
      size_t mark;
      bool restore;
      ~AllocationScope() {
        if (restore)
          allocations.resize(mark);
      }
    } scope{active_allocations_, active_allocations_.size(),
            !stmt.as<AllocBufferNode>()};
    if (const auto *alloc = stmt.as<AllocBufferNode>()) {
      active_allocations_.push_back(alloc->buffer);
      return stmt;
    }
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      Array<Stmt> out;
      for (const auto &child : seq->seq)
        out.push_back(ConvertRegion(child));
      return SeqStmt::Flatten(out);
    }
    if (const auto *loop = stmt.as<ForNode>()) {
      auto saved_bind = bind_env_;
      auto saved_values = value_map_;
      auto saved_infos = value_infos_;
      Stmt result;
      if (loop->kind == ForKind::kParallel &&
          !pto::UsesExplicitSIMD(loop->body)) {
        For unit = GetRef<For>(loop);
        auto mapping = pto::InvertPtoLaneLayout(unit);
        pto::SetPtoIndexStorage(&mapping, storage_.PrivateAllocations(unit),
                                reducers_.storage_values);
        result = ConvertUnit(unit, mapping);
        Array<Stmt> declarations;
        for (const auto &old : storage_.PrivateAllocations(unit))
          declarations.push_back(AllocBuffer(
              promoted_.at(old->data),
              {{attr::kVmiAllocKind,
                String(old.scope() == "local.var" ? "data" : "data_array")}}));
        declarations.push_back(result);
        result = SeqStmt::Flatten(declarations);
      } else {
        pto_analysis::ScopedPtoLoopBounds loop_bounds(analyzer_,
                                                      GetRef<For>(loop));
        For replacement = GetRef<For>(loop);
        replacement.CopyOnWrite()->body = ConvertRegion(loop->body);
        result = replacement;
      }
      bind_env_ = std::move(saved_bind);
      value_map_ = std::move(saved_values);
      value_infos_ = std::move(saved_infos);
      return result;
    }
    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      auto saved_bind = bind_env_;
      auto saved_values = value_map_;
      auto saved_infos = value_infos_;
      auto saved_versions = buffer_versions_;
      if (attr->attr_key == tl::attr::kBufferVersion)
        for (const auto &[data, count] : attr->node.cast<BufferVersionMap>())
          buffer_versions_.Set(data, count);
      Stmt body = ConvertRegion(attr->body);
      buffer_versions_ = std::move(saved_versions);
      bind_env_ = std::move(saved_bind);
      value_map_ = std::move(saved_values);
      value_infos_ = std::move(saved_infos);
      return AttrStmt(attr->node, attr->attr_key, attr->value, body);
    }
    if (const auto *bind = stmt.as<BindNode>()) {
      Array<Stmt> out;
      if (reducers_.accumulators.empty())
        return stmt;
      auto value = ReadRegionValue(bind->value, &out);
      Var replacement = bind->var;
      bind_env_.Set(bind->var, bind->value);
      value_map_.Set(bind->var, replacement);
      value_infos_[replacement] = Info(value);
      out.push_back(Bind(replacement, value));
      return SeqStmt::Flatten(out);
    }
    if (const auto *eval = stmt.as<EvaluateNode>()) {
      if (const auto *call = eval->value.as<CallNode>();
          call && pto::IsPtoFill(GetRef<Call>(call)))
        return LowerFill(GetRef<Call>(call), lanes_);
      if (const auto *call = eval->value.as<CallNode>();
          call && pto::IsPtoDirectReduce(GetRef<Call>(call)))
        return LowerDirectReduce(GetRef<Call>(call), lanes_);
      if (const auto *call = eval->value.as<CallNode>();
          call && call->op.same_as(PtoFinalizeReducerOp::Get()))
        return ConvertPtoFinalize(
            Downcast<PtoFinalizeReducerOp>(ParseOperator(GetRef<Call>(call))));
      return stmt;
    }
    if (const auto *store = stmt.as<BufferStoreNode>()) {
      Array<Stmt> out;
      if (reducers_.accumulators.empty())
        return stmt;
      PrimExpr value = ReadRegionValue(store->value, &out);
      auto data = store->buffer->data;
      bool seed = false;
      for (const auto &site : reducers_.finalizes)
        seed |= site->seed.defined() &&
                site->seed.value()->buffer->data.same_as(data);
      if (seed && Info(value).IsVmi() && !vf_buffers_.count(data))
        vf_buffers_.emplace(
            data, decl_buffer(store->buffer->shape, store->buffer->dtype,
                              store->buffer->name + "_v", "local.var"));
      auto target = vf_buffers_.find(data);
      if (target != vf_buffers_.end()) {
        value = ToVmiWidth(value, target->second->dtype.lanes(), &out);
        out.push_back(BufferStore(target->second, value, store->indices));
      } else if (Info(value).IsVmi()) {
        PrimExpr ptr = Call(DataType::Handle(), tl::access_ptr(),
                            {BufferLoad(store->buffer, store->indices),
                             Integer(1), Integer(2)});
        out.push_back(Evaluate(
            Call(DataType::Void(), VmiOp("vstore"), {value, ptr, Integer(0)})));
      } else
        out.push_back(BufferStore(store->buffer, value, store->indices,
                                  store->predicate));
      return SeqStmt::Flatten(out);
    }
    if (const auto *loop = stmt.as<WhileNode>()) {
      While rewritten = GetRef<While>(loop);
      rewritten.CopyOnWrite()->body = ConvertScopedRegion(loop->body);
      return rewritten;
    }
    if (const auto *branch = stmt.as<IfThenElseNode>()) {
      Stmt yes = ConvertScopedRegion(branch->then_case);
      Optional<Stmt> no;
      if (branch->else_case.defined())
        no = ConvertScopedRegion(branch->else_case.value());
      return IfThenElse(branch->condition, yes, no, branch->span);
    }
    if (const auto *block = stmt.as<SBlockNode>()) {
      auto saved_bind = bind_env_;
      auto saved_values = value_map_;
      auto saved_infos = value_infos_;
      for (const Buffer &buffer : block->alloc_buffers)
        active_allocations_.push_back(buffer);
      SBlock copy = GetRef<SBlock>(block);
      copy.CopyOnWrite()->body = ConvertRegion(block->body);
      bind_env_ = std::move(saved_bind);
      value_map_ = std::move(saved_values);
      value_infos_ = std::move(saved_infos);
      return copy;
    }
    if (const auto *realize = stmt.as<SBlockRealizeNode>())
      return SBlockRealize(realize->iter_values, realize->predicate,
                           Downcast<SBlock>(ConvertRegion(realize->block)));
    return stmt;
  }

  // ------------------------------------------------------------------
  // Per-unit conversion
  // ------------------------------------------------------------------
  Stmt ConvertUnit(const For &loop,
                   const pto_analysis::PtoInverseMapping &mapping) {
    // The inverse mapping comes from the region entry: it is the shared
    // source of truth for lanes, the mask plan and the body conversion
    // (see InvertPtoLaneLayout).
    // Reset per-unit conversion state.
    current_unit_ = loop;
    promoted_.clear();
    for (const Buffer &old : storage_.PrivateAllocations(loop)) {
      storage_.CheckStorageV1(old);
      Buffer b = decl_buffer(old->shape, old->dtype.with_lanes(mapping.lanes),
                             old->name + "_v", old.scope());
      auto *w = b.CopyOnWrite();
      w->strides = old->strides;
      w->elem_offset = old->elem_offset;
      promoted_.emplace(old->data, b);
    }
    tmp_counter_ = 0;
    unit_mapping_ = mapping;
    // Only address consumers may request algebraic scalarization of saved
    // integer coordinates. Ordinary computation uses value classification,
    // without applying the loaded-index dtype contract to its Cast inputs.
    address_vars_.clear();
    std::vector<Bind> definitions;
    std::unordered_set<BufferLoad, ObjectPtrHash, ObjectPtrEqual> address_loads;
    std::function<void(const PrimExpr &)> collect_address_vars =
        [&](const PrimExpr &expr) {
          PostOrderVisit(expr, [&](const ObjectRef &node) {
            if (auto var = node.as<Var>()) {
              address_vars_.insert(*var);
            } else if (auto load = node.as<BufferLoad>();
                       load && address_loads.insert(*load).second) {
              // The ordinary visitor skips view metadata at load sites.
              for (const PrimExpr &offset :
                   (*load)->buffer->ElemOffset((*load)->indices)) {
                collect_address_vars(offset);
              }
            }
          });
        };
    auto collect_access = [&](const Buffer &buffer,
                              const Array<PrimExpr> &indices) {
      for (const PrimExpr &index : indices) {
        collect_address_vars(index);
      }
      // View strides and origins can capture Bind values too.
      for (const PrimExpr &offset : buffer->ElemOffset(indices)) {
        collect_address_vars(offset);
      }
    };
    PostOrderVisit(loop->body, [&](const ObjectRef &node) {
      if (const auto *bind = node.as<BindNode>()) {
        definitions.push_back(GetRef<Bind>(bind));
      } else if (const auto *load = node.as<BufferLoadNode>()) {
        collect_access(load->buffer, load->indices);
      } else if (const auto *store = node.as<BufferStoreNode>()) {
        collect_access(store->buffer, store->indices);
      }
    });
    // In verified SSA every dependency precedes its use.
    for (auto it = definitions.rbegin(); it != definitions.rend(); ++it) {
      if (address_vars_.count((*it)->var)) {
        collect_address_vars((*it)->value);
      }
    }
    Optional<For> inner;
    if (const auto *body_for = loop->body.as<ForNode>()) {
      if (body_for->kind == ForKind::kParallel) {
        inner = GetRef<For>(body_for);
      }
    }
    loop_var_ = mapping.is_2d ? (mapping.select_inner ? mapping.inner_var
                                                      : mapping.outer_var)
                              : loop->loop_var;
    lanes_ = mapping.lanes;
    Var chunk("q", loop_var_.dtype());
    chunk_index_ = chunk;
    PrimExpr unit_bounds = chunk >= make_zero(chunk.dtype()) &&
                           chunk < IntImm(chunk.dtype(), mapping.chunk_count);
    const auto serial = mapping.SerialVar();
    if (serial) {
      const Var &serial_var = serial.value();
      unit_bounds =
          unit_bounds && serial_var >= make_zero(serial_var.dtype()) &&
          serial_var < IntImm(serial_var.dtype(), mapping.SerialExtent());
    }
    With<arith::ConstraintContext> unit_scope(analyzer_, unit_bounds);

    // Decide once from the static extent. A tail-bearing unit keeps the
    // existing element-wise remaining-mask loop, including its full chunks.
    const bool has_tail = mapping.tail_lanes != 0;
    Buffer remaining;
    if (has_tail) {
      remaining =
          tirx::decl_buffer({IntImm(DataType::Int(32), 1)}, DataType::Int(32),
                            "remaining", "local.var");
      mask_var_ = Var("mask", DataType::Bool(static_cast<int>(lanes_)));
    } else {

      mask_var_ = GetFullMask(lanes_);
    }

    Array<Stmt> statements;
    const Stmt &source = inner.defined() ? inner.value()->body : loop->body;
    floor_groups_.clear();
    available_floor_groups_.clear();
    ConvertBody(source, &statements);
    Stmt converted = DeadBindElim()(SeqStmt::Flatten(statements));
    converted = ShareFloorDivMod(converted);
    Array<Stmt> chunk_body;
    if (has_tail) {
      PrimExpr count = BufferLoad(remaining, {make_zero(DataType::Int(32))});
      chunk_body.push_back(
          Bind(mask_var_, Call(mask_var_.dtype(), VmiOp("create_mask"), {count},
                               {{"size", IntImm(DataType::Int(32), lanes_)}})));
      chunk_body.push_back(converted);
      chunk_body.push_back(
          BufferStore(remaining, count - IntImm(DataType::Int(32), lanes_),
                      {make_zero(DataType::Int(32))}));
    } else {
      chunk_body.push_back(converted);
    }
    Stmt body = For(chunk, make_zero(chunk.dtype()),
                    IntImm(chunk.dtype(), mapping.chunk_count),
                    ForKind::kSerial, SeqStmt::Flatten(chunk_body));
    if (has_tail) {
      Stmt init = BufferStore(remaining,
                              IntImm(DataType::Int(32), mapping.logical_extent),
                              {make_zero(DataType::Int(32))});
      body = SeqStmt::Flatten(init, body);
    }
    if (serial) {
      // Reset remaining inside the kept loop so every row/column has the
      // same ceil(E/L) iterations and its own final partial mask.
      const Var &serial_var = serial.value();
      body = For(serial_var, make_zero(serial_var.dtype()),
                 IntImm(serial_var.dtype(), mapping.SerialExtent()),
                 ForKind::kSerial, body);
    }
    return has_tail ? SeqStmt::Flatten(AllocBuffer(remaining), body) : body;
  }

  // Substitute the analysis chunk placeholder with the unified loop's q.
  PrimExpr BindChunkToQ(const PrimExpr &expr) const {
    return pto_analysis::SubstituteVar(expr, unit_mapping_.chunk_var.get(),
                                       chunk_index_);
  }

  PrimExpr SubstituteLaneZero(const PrimExpr &expr) const {
    const Var &lane = unit_mapping_.lane_var;
    return pto_analysis::SubstituteVar(expr, lane.get(),
                                       make_zero(lane.dtype()));
  }

  PrimExpr Broadcast(PrimExpr value, Array<Stmt> *pre) {
    return ToVmiWidth(value, lanes_, pre);
  }

  bool IsLaneUniformAddressExpr(const PrimExpr &expr) {
    // Fixed indices can refer to private lanes or VMI storage. Even a one-lane
    // VMI value still requires value lowering, including through aliases.
    PrimExpr simplified = analyzer_->Simplify(expr);
    bool needs_vmi = false;
    PostOrderVisit(simplified, [&](const ObjectRef &node) {
      if (const auto *load = node.as<BufferLoadNode>()) {
        needs_vmi |= storage_.IsPrivate(load->buffer->data) ||
                     vf_buffers_.count(load->buffer->data);
      }
      if (const auto *var = node.as<VarNode>()) {
        if (auto mapped = value_map_.Get(GetRef<Var>(var)))
          needs_vmi |= Info(mapped.value()).IsVmi();
      }
    });
    if (needs_vmi)
      return false;
    return pto_analysis::IsPtoLaneUniform(simplified, loop_var_, unit_mapping_,
                                          bind_env_);
  }

  PrimExpr ConvertScalarExpr(const PrimExpr &expr) {
    class ScalarRemap : public ExprMutator {
    public:
      explicit ScalarRemap(VectorizeParallelToPTOImpl *parent)
          : parent_(parent) {}
      PrimExpr VisitExpr_(const BufferLoadNode *op) final {
        // A scalar index can itself be a read from a view whose stride or
        // origin uses the Parallel coordinate.
        return parent_->ScalarBufferLoad(op->buffer, op->indices);
      }
      PrimExpr VisitExpr_(const VarNode *op) final {
        auto it = parent_->value_map_.find(GetRef<Var>(op));
        if (it != parent_->value_map_.end()) {
          ICHECK(parent_->Info((*it).second).kind ==
                 pto_analysis::ValueKind::kScalar);
          return (*it).second;
        }
        return GetRef<PrimExpr>(op);
      }

    private:
      VectorizeParallelToPTOImpl *parent_;
    } remap(this);
    PrimExpr scalar = pto_analysis::SubstituteVar(
        expr, loop_var_.get(), BindChunkToQ(unit_mapping_.index_expr));
    scalar = analyzer_->Simplify(SubstituteLaneZero(scalar));
    return analyzer_->Simplify(remap(scalar));
  }

  PrimExpr ScalarBufferLoad(const Buffer &buffer,
                            const Array<PrimExpr> &indices) {
    bool remap_metadata = false;
    auto inspect = [&](const PrimExpr &expr) {
      PostOrderVisit(expr, [&](const ObjectRef &node) {
        if (node.as<BufferLoadNode>()) {
          // The nested load's view may depend on the Parallel coordinate even
          // when its logical indices are constant.
          remap_metadata = true;
        } else if (const auto *var = node.as<VarNode>()) {
          remap_metadata |=
              var == loop_var_.get() || value_map_.count(GetRef<Var>(var));
        }
      });
    };
    for (const PrimExpr &shape : buffer->shape)
      inspect(shape);
    for (const PrimExpr &stride : buffer->strides)
      inspect(stride);
    inspect(buffer->elem_offset);
    bool has_non_scalar_index = false;
    for (const PrimExpr &index : indices) {
      PostOrderVisit(index, [&](const ObjectRef &node) {
        if (auto var = node.as<Var>()) {
          auto mapped = value_map_.find(*var);
          if (mapped != value_map_.end()) {
            auto info = Info((*mapped).second);
            has_non_scalar_index |=
                info.kind != pto_analysis::ValueKind::kScalar;
          }
        }
      });
    }
    if (remap_metadata || has_non_scalar_index) {
      // Simplify the complete address before remapping it. A zero stride
      // can make a lane-varying logical index irrelevant to a scalar address.
      for (const Buffer &owner : active_allocations_) {
        if (owner->data.same_as(buffer->data)) {
          Buffer flat = owner.GetFlattenedBuffer();
          PrimExpr offset = ConvertScalarExpr(buffer->ElemOffset(indices)[0]);
          if (flat->dtype != buffer->dtype) {
            // Scalar and continuous accesses keep the view's element units
            // when it reinterprets the allocation.
            PrimExpr extent = analyzer_->Simplify(floordiv(
                GetBufferStorageSizeBytes(flat), buffer->dtype.bytes()));
            auto *view = flat.CopyOnWrite();
            view->dtype = buffer->dtype;
            view->shape = {extent};
            view->elem_offset = make_zero(offset.dtype());
          }
          return BufferLoad(flat,
                            {analyzer_->Simplify(offset - flat->elem_offset)});
        }
      }
    }
    Array<PrimExpr> scalar_indices = indices.Map(
        [&](const PrimExpr &index) { return ConvertScalarExpr(index); });
    return BufferLoad(buffer, scalar_indices);
  }

  PrimExpr ConvertUint16Offset(const PrimExpr &expr, Array<Stmt> *pre) {
    const DataType dtype = DataType::UInt(16);
    const DataType vector_dtype = VectorDType(dtype, lanes_);
    if (const auto *integer = expr.as<IntImmNode>()) {
      return make_const(dtype, integer->value);
    }
    if (expr.as<SelectNode>()) {
      return cast(dtype, ConvertScalarExpr(expr));
    }
    if (const auto *var = expr.as<VarNode>()) {
      if (var == loop_var_.get() || var == unit_mapping_.lane_var.get()) {
        PrimExpr base =
            var == loop_var_.get()
                ? BindChunkToQ(SubstituteLaneZero(unit_mapping_.index_expr))
                : make_zero(dtype);
        return EmitBind("offset_vci",
                        Call(vector_dtype, VmiOp("vci"),
                             {cast(dtype, analyzer_->Simplify(base))},
                             {{"size", IntImm(DataType::Int(32), lanes_)}}),
                        pre);
      }
      return cast(dtype, ConvertScalarExpr(expr));
    }
    const PrimExpr *lhs;
    const PrimExpr *rhs;
    std::string operation;
    if (const auto *add = expr.as<AddNode>()) {
      lhs = &add->a;
      rhs = &add->b;
      operation = "vadd";
    } else if (const auto *sub = expr.as<SubNode>()) {
      lhs = &sub->a;
      rhs = &sub->b;
      operation = "vsub";
    } else {
      const auto *mul = expr.as<MulNode>();
      ICHECK(mul) << "unsupported proved uint16 offset node "
                  << expr->GetTypeKey();
      lhs = &mul->a;
      rhs = &mul->b;
      operation = "vmul";
    }
    PrimExpr a = ConvertUint16Offset(*lhs, pre);
    PrimExpr b = ConvertUint16Offset(*rhs, pre);
    return EmitBinary(std::move(a), std::move(b), operation, "offset", pre);
  }

  struct IndexedOperands {
    PrimExpr pointer;
    PrimExpr base;
    PrimExpr offsets;
  };

  IndexedOperands LowerIndexed(const pto_analysis::MemoryAccess &access,
                               Array<Stmt> *pre) {
    Buffer allocation = pto_analysis::FindAllocationOwner(
        active_allocations_, access.buffer, "VectorizeParallelToPTO");
    pto_analysis::IndexedMemoryOwner owner =
        pto_analysis::DescribeIndexedMemoryOwner(access.buffer, allocation);
    pto_analysis::PtoIndexedAccessPlan plan =
        pto_analysis::PlanPtoIndexedAccess(access, owner, loop_var_,
                                           unit_mapping_, analyzer_,
                                           buffer_versions_, bind_env_);
    PrimExpr base = ConvertScalarExpr(BindChunkToQ(plan.base));
    PrimExpr offset_expr = BindChunkToQ(plan.offsets);
    PrimExpr offsets = plan.generate_uint16_coordinates
                           ? ConvertUint16Offset(offset_expr, pre)
                           : ConvertExpr(offset_expr, pre);
    offsets = Broadcast(offsets, pre);
    // VMI gather and scatter only access memory for active lanes, so tail
    // offsets can be passed directly. PTODSL normalizes the offset carrier.
    Optional<PrimExpr> version_index;
    if (plan.version_index) {
      version_index = ConvertScalarExpr(plan.version_index.value());
    }
    return {pto_analysis::MakeIndexedMemoryDescriptor(
                owner, access.is_write, version_index, plan.descriptor_extent),
            base, offsets};
  }

  // ------------------------------------------------------------------
  // Statement conversion (pre-binds first, then the statement)
  // ------------------------------------------------------------------
  struct FloorGroup {
    PrimExpr lhs;
    PrimExpr rhs;
    Array<Var> reads;
    Array<Var> results;
  };

  bool HasLocalAllocation(const Var &data) const {
    return std::any_of(
        active_allocations_.begin(), active_allocations_.end(),
        [&](const Buffer &buffer) { return buffer->data.same_as(data); });
  }

  bool TrackFloorOperand(const PrimExpr &expr, Array<Var> *reads) const {
    if (const auto *load = expr.as<BufferLoadNode>()) {
      if (!HasLocalAllocation(load->buffer->data) ||
          load->predicate.defined()) {
        return false;
      }
      for (const PrimExpr &offset : load->buffer->ElemOffset(load->indices)) {
        if (SideEffect(offset) != CallEffectKind::kPure) {
          return false;
        }
      }
      reads->push_back(load->buffer->data);
      return true;
    }
    return SideEffect(expr) == CallEffectKind::kPure;
  }

  static bool SameFloorOperand_(const PrimExpr &lhs, const PrimExpr &rhs) {
    if (const auto *lhs_load = lhs.as<BufferLoadNode>()) {
      const auto *rhs_load = rhs.as<BufferLoadNode>();
      // Structural equality can map different Buffer data vars to each other.
      return rhs_load && lhs_load->buffer.same_as(rhs_load->buffer) &&
             StructuralEqual()(lhs_load->indices, rhs_load->indices);
    }
    return StructuralEqual()(lhs, rhs);
  }

  void TrackFloorResult(const PrimExpr &lhs, const PrimExpr &rhs,
                        const Var &result) {
    Array<Var> reads;
    if (!TrackFloorOperand(lhs, &reads) || !TrackFloorOperand(rhs, &reads)) {
      return;
    }
    for (size_t index : available_floor_groups_) {
      FloorGroup &group = floor_groups_[index];
      if (SameFloorOperand_(group.lhs, lhs) &&
          SameFloorOperand_(group.rhs, rhs)) {
        group.results.push_back(result);
        return;
      }
    }
    available_floor_groups_.push_back(floor_groups_.size());
    floor_groups_.push_back({lhs, rhs, std::move(reads), {result}});
  }

  void InvalidateFloorReads(const Var &data) {
    const bool known_allocation =
        HasLocalAllocation(data) || promoted_.count(data);
    auto invalid = [&](size_t index) {
      const Array<Var> &reads = floor_groups_[index].reads;
      return std::any_of(reads.begin(), reads.end(), [&](const Var &read) {
        return !known_allocation || read.same_as(data);
      });
    };
    available_floor_groups_.erase(
        std::remove_if(available_floor_groups_.begin(),
                       available_floor_groups_.end(), invalid),
        available_floor_groups_.end());
  }

  Stmt ShareFloorDivMod(const Stmt &body) {
    if (std::none_of(
            floor_groups_.begin(), floor_groups_.end(),
            [](const FloorGroup &group) { return group.results.size() > 1; })) {
      return body;
    }
    using BindMap =
        std::unordered_map<Var, Bind, ObjectPtrHash, ObjectPtrEqual>;
    using Replacements =
        std::unordered_map<Var, Stmt, ObjectPtrHash, ObjectPtrEqual>;
    BindMap live_floors;
    PostOrderVisit(body, [&](const ObjectRef &node) {
      const auto *bind = node.as<BindNode>();
      const auto *call = bind ? bind->value.as<CallNode>() : nullptr;
      if (call && (call->op.same_as(VmiOp("vfloordiv")) ||
                   call->op.same_as(VmiOp("vfloormod")))) {
        live_floors.emplace(bind->var, GetRef<Bind>(bind));
      }
    });

    Replacements replacements;
    for (const FloorGroup &group : floor_groups_) {
      std::vector<Bind> binds;
      bool has_quotient = false;
      bool has_remainder = false;
      for (const Var &result : group.results) {
        auto it = live_floors.find(result);
        if (it == live_floors.end()) {
          continue;
        }
        const auto *call = it->second->value.as<CallNode>();
        has_quotient |= call->op.same_as(VmiOp("vfloordiv"));
        has_remainder |= call->op.same_as(VmiOp("vfloormod"));
        binds.push_back(it->second);
      }
      if (!has_quotient || !has_remainder) {
        continue;
      }
      const auto *first = binds.front()->value.as<CallNode>();
      Var pair("divmod_" + std::to_string(tmp_counter_++), first->dtype);
      Bind compute(pair,
                   Call(first->dtype, VmiOp("vfloordivmod"), first->args));
      for (const Bind &bind : binds) {
        const auto *call = bind->value.as<CallNode>();
        const bool is_mod = call->op.same_as(VmiOp("vfloormod"));
        Bind project(bind->var, Call(call->dtype, VmiOp("pair_get"),
                                     {pair, Integer(is_mod ? 1 : 0)}));
        replacements.emplace(bind->var, bind.same_as(binds.front())
                                            ? SeqStmt::Flatten(compute, project)
                                            : Stmt(project));
      }
    }
    if (replacements.empty()) {
      return body;
    }
    Stmt shared = IRTransform(
        body, nullptr,
        ffi::Function::FromTyped([&](const ObjectRef &node) -> ObjectRef {
          const auto *bind = node.as<BindNode>();
          auto it = replacements.find(bind->var);
          return it == replacements.end() ? node : it->second;
        }),
        Array<String>{"tirx.Bind"});
    return DeadBindElim()(shared);
  }

  void ConvertBody(const Stmt &body, Array<Stmt> *out) {
    if (const auto *alloc = body.as<AllocBufferNode>()) {
      ICHECK(storage_.Classify(alloc->buffer, current_unit_) ==
             pto::StorageClass::kPrivate);
      return;
    }
    if (const auto *seq = body.as<SeqStmtNode>()) {
      for (const auto &s : seq->seq) {
        ConvertBody(s, out);
      }
      return;
    }
    if (const auto *bind = body.as<BindNode>()) {
      Array<Stmt> pre;
      PrimExpr value =
          address_vars_.count(bind->var) &&
                  pto_analysis::IsPtoIndexDType(bind->value.dtype()) &&
                  IsLaneUniformAddressExpr(bind->value)
              ? ConvertScalarExpr(bind->value)
              : ConvertExpr(bind->value, &pre);
      // Pre-binds first, then the Bind itself.
      for (const auto &s : pre) {
        out->push_back(s);
      }
      Var new_var(bind->var->name_hint, value.dtype());
      bind_env_.Set(bind->var, bind->value); // analysis environment
      value_map_.Set(bind->var, new_var);    // value environment
      value_infos_[new_var] = Info(value);
      out->push_back(Bind(new_var, value));
      return;
    }
    if (const auto *store = body.as<BufferStoreNode>()) {
      ICHECK(!store->predicate.defined())
          << "[VectorizeParallelToPTO] internal error: predicated store passed "
             "Verify; predicated stores must be rejected by "
             "VerifyParallelToPTO";
      Array<Stmt> pre;
      PrimExpr value = ConvertExpr(store->value, &pre);
      auto kind = storage_.Classify(store->buffer, current_unit_);
      bool accumulator = reducers_.accumulators.count(store->buffer->data);
      if (kind == pto::StorageClass::kPrivate || accumulator) {
        auto slot = storage_.ResolveSlot(store->buffer, store->indices,
                                         analyzer_, bind_env_);
        Buffer target = accumulator ? vf_buffers_.at(store->buffer->data)
                                    : promoted_.at(store->buffer->data);
        value = ToVmiWidth(value, lanes_, &pre);
        if (accumulator && unit_mapping_.tail_lanes != 0) {
          auto old = EmitBind("previous_partial",
                              BufferLoad(target, {Integer(slot.slot)}), &pre);
          value = EmitBind(
              "preserve_partial",
              Call(target->dtype, VmiOp("vsel"), {mask_var_, value, old}),
              &pre);
        }
        ICHECK(value.dtype() == target->dtype)
            << "PTO private Store dtype/lanes mismatch";
        for (const Stmt &stmt : pre)
          out->push_back(stmt);
        InvalidateFloorReads(store->buffer->data);
        out->push_back(
            BufferStore(target, value, {IntImm(DataType::Int(32), slot.slot)}));
        return;
      }
      ICHECK(kind != pto::StorageClass::kExternal)
          << "PTO external ordinary storage write";
      auto access = pto_analysis::AnalyzeBufferAccess(
          store->buffer, store->indices, /*is_write=*/true, loop_var_,
          unit_mapping_, bind_env_);
      PrimExpr call;
      if (access.pattern == pto_analysis::AccessPattern::kContinuous ||
          (access.pattern == pto_analysis::AccessPattern::kLaneUniform &&
           unit_mapping_.logical_extent == 1 &&
           is_zero(access.original_offset))) {
        Array<PrimExpr> starts = access.start_indices.Map(
            [&](const PrimExpr &index) { return BindChunkToQ(index); });
        // A single iteration can store an existing one-lane VMI value
        // directly, just like a region-level reducer result writeback.
        const bool point_store =
            unit_mapping_.logical_extent == 1 &&
            Info(value).kind == pto::ValueKind::kOneLaneVmi;
        const int64_t extent = point_store ? 1 : lanes_;
        PrimExpr ptr = Call(DataType::Handle(), tl::access_ptr(),
                            {ScalarBufferLoad(store->buffer, starts),
                             Integer(extent), Integer(2)});
        Array<PrimExpr> args = {ToVmiWidth(value, extent, &pre), ptr,
                                Integer(0)};
        if (!point_store)
          args.push_back(mask_var_);
        call = Call(DataType::Void(), VmiOp("vstore"), args);
      } else {
        value = ToVmiWidth(value, lanes_, &pre);
        IndexedOperands operands = LowerIndexed(access, &pre);
        call = Call(DataType::Void(), VmiOp("vscatter"),
                    {value, operands.pointer, operands.base, operands.offsets,
                     mask_var_});
      }
      for (const Stmt &stmt : pre)
        out->push_back(stmt);
      InvalidateFloorReads(store->buffer->data);
      out->push_back(Evaluate(call));
      return;
    }
    if (const auto *attr = body.as<AttrStmtNode>()) {
      // Verify guarantees the metadata has no lane-dependent semantics
      // (direct or via bind aliases). Rewrite its Var references through
      // value_map_ so already-converted binds are not left dangling
      // The body is converted underneath.
      Array<Stmt> stmts;
      // The body is a nested statement scope — keep the
      // source-bind and value maps scoped with it, so a bind defined
      // inside cannot remap a later sibling's reference (Verify rejects
      // that input; this keeps the transformation state consistent with
      // the unit visitor's scope handling).
      Map<Var, PrimExpr> saved_env = bind_env_;
      Map<Var, Var> saved_map = value_map_;
      available_floor_groups_.clear();
      ConvertBody(attr->body, &stmts);
      available_floor_groups_.clear();
      bind_env_ = std::move(saved_env);
      value_map_ = std::move(saved_map);
      Stmt inner = SeqStmt::Flatten(stmts);
      // Default to the *original* node. Only PrimExpr metadata is
      // remapped; a non-object Any (int/str atom metadata) must forward
      // unchanged — the old code wrote back a default-constructed (empty)
      // node when the try_cast failed.
      ffi::Any node = attr->node;
      if (auto expr = attr->node.try_cast<PrimExpr>()) {
        node = tirx::Substitute(expr.value(), value_map_);
      }
      PrimExpr value = tirx::Substitute(attr->value, value_map_);
      out->push_back(AttrStmt(node, attr->attr_key, value, inner));
      return;
    }
    if (const auto *eval = body.as<EvaluateNode>()) {
      // Convert the expression with the same machinery so no stale loop
      // variable survives; the whitelist is Verify's job.
      Array<Stmt> pre;
      PrimExpr value = ConvertExpr(eval->value, &pre);
      for (const auto &s : pre) {
        out->push_back(s);
      }
      out->push_back(Evaluate(value));
      return;
    }
    if (const auto *loop = body.as<ForNode>()) {
      ICHECK(loop->kind == ForKind::kSerial)
          << "[VectorizeParallelToPTO] internal error: non-serial loop "
             "passed Verify";
      // Loop-shell preservation: a serial loop admitted by Verify is copied
      // through as a scalar shell; the body is converted with the same
      // statement machinery. Binds defined inside stay scoped to the shell
      // (same discipline as the AttrStmt branch). The bounds are copied through
      // verbatim: Verify already rejected a negative constant extent and a
      // non-positive or non-constant step, while a runtime extent is the
      // program's own contract, so the source min/extent/step are kept (a zero
      // trip count is an empty shell).
      Array<Stmt> inner;
      Map<Var, PrimExpr> saved_env = bind_env_;
      Map<Var, Var> saved_map = value_map_;
      available_floor_groups_.clear();
      ConvertBody(loop->body, &inner);
      available_floor_groups_.clear();
      bind_env_ = std::move(saved_env);
      value_map_ = std::move(saved_map);
      // Keep the shell's step (and annotations): the step only changes the
      // sequence of values the induction variable takes, and the shell stays
      // lane-uniform either way.
      out->push_back(For(loop->loop_var, loop->min, loop->extent,
                         ForKind::kSerial, SeqStmt::Flatten(inner),
                         std::nullopt, loop->annotations, loop->step,
                         loop->span));
      return;
    }
    LOG(FATAL) << "[VectorizeParallelToPTO] internal error: unsupported "
                  "statement passed Verify: "
               << body->GetTypeKey();
  }

  // ------------------------------------------------------------------
  // Expression conversion (index vars resolve through bind_env_)
  // ------------------------------------------------------------------
  PrimExpr ConvertExpr(const PrimExpr &expr, Array<Stmt> *pre) {
    if (const auto *load = expr.as<BufferLoadNode>()) {
      if (vf_buffers_.count(load->buffer->data))
        return ReadRegionValue(expr, pre);
      auto kind = storage_.Classify(load->buffer, current_unit_);
      if (kind == pto::StorageClass::kPrivate ||
          kind == pto::StorageClass::kExternal) {
        auto slot = storage_.ResolveSlot(load->buffer, load->indices, analyzer_,
                                         bind_env_);
        Buffer buffer = kind == pto::StorageClass::kPrivate
                            ? promoted_.at(load->buffer->data)
                            : storage_.Allocation(load->buffer->data);
        return EmitBind(
            "local_snapshot",
            BufferLoad(buffer, {IntImm(DataType::Int(32), slot.slot)}), pre);
      }
      auto access = pto_analysis::AnalyzeBufferAccess(
          load->buffer, load->indices, /*is_write=*/false, loop_var_,
          unit_mapping_, bind_env_);
      if (access.pattern == pto_analysis::AccessPattern::kContinuous ||
          (access.pattern == pto_analysis::AccessPattern::kLaneUniform &&
           unit_mapping_.logical_extent == 1 &&
           is_zero(access.original_offset))) {
        Array<PrimExpr> starts = access.start_indices.Map(
            [&](const PrimExpr &index) { return BindChunkToQ(index); });
        PrimExpr ptr =
            Call(DataType::Handle(), tl::access_ptr(),
                 {ScalarBufferLoad(load->buffer, starts),
                  IntImm(DataType::Int(32), static_cast<int>(lanes_)),
                  IntImm(DataType::Int(32), 1)});
        return EmitBind("vload",
                        Call(VectorDType(load->dtype, lanes_), VmiOp("vload"),
                             {ptr, IntImm(DataType::Int(32), 0)},
                             {{"size", IntImm(DataType::Int(32),
                                              static_cast<int>(lanes_))}}),
                        pre);
      }
      if (access.pattern == pto_analysis::AccessPattern::kIndexed) {
        IndexedOperands operands = LowerIndexed(access, pre);
        const bool byte_payload = load->dtype.bits() == 8;
        const DataType result_dtype =
            byte_payload ? (load->dtype.is_int() ? DataType::Int(16)
                                                 : DataType::UInt(16))
                         : load->dtype;
        PrimExpr gathered = EmitBind(
            "gather",
            Call(VectorDType(result_dtype, lanes_), VmiOp("vgather"),
                 {operands.pointer, operands.base, operands.offsets, mask_var_},
                 {{"pmode", StringImm("zero")}}),
            pre);
        if (byte_payload) {
          // Byte gathers zero-extend into B16 even for signed sources. Restore
          // source bits before any user Cast, arithmetic or saved Bind sees it.
          gathered = EmitBind(
              "gather_byte",
              Call(VectorDType(load->dtype, lanes_), VmiOp("vcvt"), {gathered},
                   {{"to_dtype", StringImm(TvmDtypeName(load->dtype))},
                    {"saturate", StringImm("NOSAT")}}),
              pre);
        }
        return gathered;
      }
      // lane-uniform scalar load: keep the BufferLoad (lane bound to 0,
      // chunk bound to q — indices resolve through the bind env
      // inside the shared helper already) and bind it to a scalar Var in
      // the current chunk; later uses (vbrc broadcast, scalar arithmetic,
      // store values) reference that Var. Each textual occurrence binds
      // separately — no cross-access load reuse. The Var dtype is the
      // buffer's own scalar dtype.
      Array<PrimExpr> indices = access.start_indices.Map(
          [&](const PrimExpr &index) { return BindChunkToQ(index); });
      return EmitBind("sload", ScalarBufferLoad(load->buffer, indices), pre);
    }
    if (const auto *var = expr.as<VarNode>()) {
      auto it = value_map_.find(GetRef<Var>(var));
      if (it != value_map_.end()) {
        return (*it).second;
      }
      if (var == loop_var_.get() || var == unit_mapping_.lane_var.get()) {
        // The selected coordinate used as a value becomes a VMI vector.
        PrimExpr base = var == loop_var_.get()
                            ? analyzer_->Simplify(BindChunkToQ(
                                  SubstituteLaneZero(unit_mapping_.index_expr)))
                            : make_zero(var->dtype);
        return EmitBind(
            "vci",
            Call(VectorDType(var->dtype, lanes_), VmiOp("vci"), {base},
                 {{"size",
                   IntImm(DataType::Int(32), static_cast<int>(lanes_))}}),
            pre);
      }
      return GetRef<PrimExpr>(var);
    }
    if (expr.as<IntImmNode>() || expr.as<FloatImmNode>()) {
      return expr;
    }
    if (const auto *add = expr.as<AddNode>()) {
      return ConvertBinary(add->a, add->b, "vadd", pre);
    }
    if (const auto *sub = expr.as<SubNode>()) {
      return ConvertBinary(sub->a, sub->b, "vsub", pre);
    }
    if (const auto *mul = expr.as<MulNode>()) {
      return ConvertBinary(mul->a, mul->b, "vmul", pre);
    }
    if (const auto *div = expr.as<DivNode>()) {
      return ConvertBinary(div->a, div->b, "vdiv", pre);
    }
    if (const auto *div = expr.as<FloorDivNode>()) {
      return ConvertFloor(div->a, div->b, /*is_mod=*/false, pre);
    }
    if (const auto *mod = expr.as<FloorModNode>()) {
      return ConvertFloor(mod->a, mod->b, /*is_mod=*/true, pre);
    }
    if ((expr.as<ModNode>() || expr.as<MinNode>() || expr.as<MaxNode>() ||
         expr.as<SelectNode>()) &&
        pto_analysis::IsPtoIndexDType(expr.dtype())) {
      // Index coordinates (gather/scatter) only. A value Select/Min/Max from
      // control-flow legalization is lane-varying or carries non-index
      // (float/bool) loads, so it must fall through to the value lowering
      // below instead of the scalar coordinate path.
      bool coordinate_uniform = false;
      try {
        coordinate_uniform = IsLaneUniformAddressExpr(expr);
      } catch (const pto_analysis::PtoAnalysisError &) {
        // The lane-uniform proof met a non-index (value) load: the expression
        // is a value choice, not a coordinate.
        coordinate_uniform = false;
      }
      if (coordinate_uniform) {
        PrimExpr coordinate = pto_analysis::SubstituteVar(
            expr, loop_var_.get(), unit_mapping_.index_expr);
        return ConvertScalarExpr(BindChunkToQ(coordinate));
      }
      // Lane-varying int value choices fall through to the vsel lowering.
    }
    if (const auto *call = expr.as<CallNode>()) {
      Call value_call = GetRef<Call>(call);
      auto bit_op = pto_analysis::GetPtoBitOp(value_call);
      if (bit_op != pto_analysis::PtoBitOp::kNone) {
        return ConvertBitOp(value_call, bit_op, pre);
      }
      // Lane-mask boolean combinations come first. They are not unary
      // math, so the strict unary-math mapping below must not see them
      // (it only covers fabs/exp/log/sqrt). The frontend infinity()
      // constant never arrives here: Legalize rewrites it to FloatImm.
      if (const char *mask_op = pto_analysis::GetPtoMaskLogicOp(value_call)) {
        const bool is_not = std::string_view(mask_op) == "vnot";
        ICHECK_EQ(call->args.size(), is_not ? 1U : 2U);
        // Convert each operand exactly once, then dispatch on the converted
        // shape. The call itself carries the spelling the frontend used
        // (tirx.bitwise_and / tir.And / ...), so rebuilding it keeps the
        // scalar form faithful instead of substituting our own node.
        PrimExpr a = ConvertExpr(call->args[0], pre);
        if (is_not) {
          return CombineNot(
              a,
              [&](const PrimExpr &x) {
                return Call(expr.dtype(), call->op, {x});
              },
              pre);
        }
        PrimExpr b = ConvertExpr(call->args[1], pre);
        return CombineBool(
            a, b,
            [&](const PrimExpr &x, const PrimExpr &y) {
              return Call(expr.dtype(), call->op, {x, y});
            },
            mask_op, pre);
      }
      const char *unary = pto_analysis::GetPtoUnaryOp(value_call);
      ICHECK(unary) << "[VectorizeParallelToPTO] call without a VMI mapping: "
                    << call->op;
      PrimExpr a = ConvertExpr(call->args[0], pre);
      if (Info(a).kind == pto::ValueKind::kScalar) {
        return Call(expr.dtype(), call->op, {a});
      }
      auto width = Info(a).logical_lanes;
      return EmitBind(unary,
                      Call(VectorDType(expr.dtype().element_of(), width),
                           VmiOp(unary),
                           {a, width == 1 ? GetFullMask(1) : mask_var_}),
                      pre);
    }
    if (const auto *cast = expr.as<CastNode>()) {
      // Convert the input exactly once, then plan from the explicit shape of
      // that converted input (the original Cast carries the element types).
      PrimExpr value = ConvertExpr(cast->value, pre);
      pto_analysis::CastContext context;
      int64_t cast_lanes = Info(value).logical_lanes;
      if (Info(value).kind == pto::ValueKind::kScalar) {
        context.value_kind = pto_analysis::CastValueKind::kScalar;
        context.logical_lanes = 1;
      } else {
        ICHECK(value.dtype().lanes() == 1 || value.dtype().lanes() == lanes_)
            << "[VectorizeParallelToPTO] internal error: Cast input has "
               "unexpected lanes "
            << value.dtype().lanes() << " (current VL " << lanes_ << ")";
        context.value_kind = pto_analysis::CastValueKind::kVectorVmi;
        context.logical_lanes = cast_lanes;
      }
      ICHECK_EQ(value.dtype().element_of(), cast->value.dtype())
          << "[VectorizeParallelToPTO] internal error: converted Cast input "
             "element type "
          << value.dtype().element_of() << " does not match the original "
          << cast->value.dtype();
      std::string plan_why;
      auto plan = pto_analysis::PlanValueCast(cast, context, &plan_why);
      ICHECK(plan.has_value())
          << "[VectorizeParallelToPTO] internal error: Cast passed Verify "
             "without a conversion plan: "
          << plan_why;
      if (const auto *scalar =
              std::get_if<pto_analysis::ScalarCastPlan>(&plan.value())) {
        ICHECK_EQ(value.dtype().lanes(), 1)
            << "[VectorizeParallelToPTO] internal error: scalar Cast plan "
               "for a vector input";
        if (scalar->action == pto_analysis::ScalarCastAction::kKeepInput) {
          return value;
        }
        // Rebuild the ordinary scalar Cast with the validated annotations and
        // the original span; the scalar lowering itself is unchanged.
        return Cast(cast->dtype, value, scalar->annotations, cast->span);
      }
      const auto &vector = std::get<pto_analysis::VectorCastPlan>(plan.value());
      PrimExpr current = value;
      for (const auto &step : vector.steps) {
        ICHECK_EQ(current.dtype().element_of(), step.from)
            << "[VectorizeParallelToPTO] internal error: Cast step expects "
            << step.from << ", got " << current.dtype().element_of();
        Map<String, ObjectRef> attrs;
        attrs.Set("to_dtype", StringImm(TvmDtypeName(step.to)));
        Call call;
        if (step.op == pto_analysis::VectorCastOp::kIntegerReinterpret) {
          call = Call(VectorDType(step.to, cast_lanes),
                      VmiOp("vinterpret_cast"), {current}, attrs);
        } else {
          if (!step.saturate.empty()) {
            attrs.Set("saturate", StringImm(step.saturate));
          }
          if (!step.rounding.empty()) {
            attrs.Set("rounding", StringImm(step.rounding));
          }
          call = Call(VectorDType(step.to, cast_lanes), VmiOp("vcvt"),
                      {current}, attrs);
        }
        current = EmitBind("cast", call, pre);
      }
      // Zero steps is the proven identity; a final type check keeps every
      // multi-step chain honest (source convert once, dtype/VL continuous).
      ICHECK_EQ(current.dtype(), VectorDType(cast->dtype, cast_lanes))
          << "[VectorizeParallelToPTO] internal error: Cast plan ended at "
          << current.dtype() << ", expected "
          << VectorDType(cast->dtype, cast_lanes);
      return current;
    }
    if (pto_analysis::IsPtoScalarCoordinate(expr, loop_var_, bind_env_,
                                            unit_mapping_.lane_var)) {
      return ConvertScalarExpr(expr);
    }
    if (const auto *andn = expr.as<AndNode>()) {
      PrimExpr a = ConvertExpr(andn->a, pre);
      PrimExpr b = ConvertExpr(andn->b, pre);
      return CombineBool(
          a, b, [](const PrimExpr &x, const PrimExpr &y) { return And(x, y); },
          "vand", pre);
    }
    if (const auto *orn = expr.as<OrNode>()) {
      PrimExpr a = ConvertExpr(orn->a, pre);
      PrimExpr b = ConvertExpr(orn->b, pre);
      return CombineBool(
          a, b, [](const PrimExpr &x, const PrimExpr &y) { return Or(x, y); },
          "vor", pre);
    }
    if (const auto *notn = expr.as<NotNode>()) {
      PrimExpr a = ConvertExpr(notn->a, pre);
      return CombineNot(a, [](const PrimExpr &x) { return Not(x); }, pre);
    }
    if (const auto *sel = expr.as<SelectNode>()) {

      // Select has two lowerings. When the condition and both arms are
      // lane-uniform it stays a scalar Select (codegen emits pto.select), which
      // keeps scalar-only operations such as int8 multiply legal. Otherwise the
      // condition becomes a lane mask, scalar arms are broadcast, and it lowers
      // to one vsel.
      PrimExpr cond = ConvertExpr(sel->condition, pre);
      PrimExpr tval = ConvertExpr(sel->true_value, pre);
      PrimExpr fval = ConvertExpr(sel->false_value, pre);
      if (IsScalarBool(cond) && Info(tval).kind == pto::ValueKind::kScalar &&
          Info(fval).kind == pto::ValueKind::kScalar) {
        // A vector consumer broadcasts this at the combination point.
        return Select(cond, tval, fval);
      }
      if (IsScalarBool(cond)) {
        cond = ScalarBoolToMask(cond, pre);
      }
      tval = ToVmiWidth(tval, lanes_, pre);
      fval = ToVmiWidth(fval, lanes_, pre);
      ICHECK(cond.dtype().lanes() > 1)
          << "[VectorizeParallelToPTO] unsupported Select condition: expected "
             "a lane mask, a lane-uniform comparison, or a scalar bool; got a "
             "scalar `"
          << sel->condition->GetTypeKey()
          << "`. Rewrite the test as a comparison, or make it lane-varying.";
      return EmitBind(
          "vsel", Call(tval.dtype(), VmiOp("vsel"), {cond, tval, fval}), pre);
    }
    PrimExpr rel_a;
    PrimExpr rel_b;
    if (const pto::RelationalKind *rel =
            pto::MatchRelational(expr, &rel_a, &rel_b)) {
      return ConvertCompare(rel_a, rel_b, rel->cmp, pre);
    }
    LOG(FATAL) << "[VectorizeParallelToPTO] internal error: unsupported "
                  "expression passed Verify: "
               << expr->GetTypeKey();
  }

  // A bool with no lanes: a kernel parameter, a value read from a 1-element
  // buffer, or the result of scalar-only arithmetic. It is a valid scalar
  // Select condition; a mask consumer lifts it via ScalarBoolToMask.
  static bool IsScalarBool(const PrimExpr &e) {
    return e.dtype().is_bool() && e.dtype().lanes() == 1;
  }

  // Materialise a scalar bool as a lane mask that is either all-true or
  // all-false, which is exactly the semantics of applying one scalar
  // predicate to a whole vector. A VMI mask must be produced by
  // pto.vmi.create_mask, so the flag becomes the active-lane count: all lanes
  // when it is true, none when it is false. The mask therefore stays in the
  // VMI domain from the start, with no vector operand and no carrier dtype.
  PrimExpr ScalarBoolToMask(const PrimExpr &flag, Array<Stmt> *pre) {
    DataType mask_ty = DataType::Bool(static_cast<int>(lanes_));
    PrimExpr active = EmitBind(
        "pcount",
        Select(flag, IntImm(DataType::Int(32), static_cast<int>(lanes_)),
               IntImm(DataType::Int(32), 0)),
        pre);
    return EmitBind(
        "cmask",
        Call(mask_ty, VmiOp("create_mask"), {active},
             {{"size", IntImm(DataType::Int(32), static_cast<int>(lanes_))}}),
        pre);
  }

  // Boolean EQ/NE over lane masks (a mask with a scalar bool included):
  // vcmp/vcmps take vregs and the VMI element type set has no one-bit bool,
  // so build the truth table out of vand/vor/vnot instead.
  PrimExpr BoolCompareToMask(const PrimExpr &a, const PrimExpr &b,
                             const char *cmp, Array<Stmt> *pre) {
    DataType mask_ty = DataType::Bool(static_cast<int>(lanes_));
    PrimExpr am = AsLaneMask(a, pre);
    PrimExpr bm = AsLaneMask(b, pre);
    PrimExpr na = EmitBind("mnot", Call(mask_ty, VmiOp("vnot"), {am}), pre);
    PrimExpr nb = EmitBind("mnot", Call(mask_ty, VmiOp("vnot"), {bm}), pre);
    if (std::string(cmp) == "eq") {
      PrimExpr keep =
          EmitBind("vand", Call(mask_ty, VmiOp("vand"), {am, bm}), pre);
      PrimExpr drop =
          EmitBind("vand", Call(mask_ty, VmiOp("vand"), {na, nb}), pre);
      return EmitBind("vor", Call(mask_ty, VmiOp("vor"), {keep, drop}), pre);
    }
    PrimExpr left =
        EmitBind("vand", Call(mask_ty, VmiOp("vand"), {am, nb}), pre);
    PrimExpr right =
        EmitBind("vand", Call(mask_ty, VmiOp("vand"), {na, bm}), pre);
    return EmitBind("vor", Call(mask_ty, VmiOp("vor"), {left, right}), pre);
  }

  // Turn a value that may still be a scalar bool into a lane mask. Anything
  // that is already a lane vector (a real mask) passes through untouched.
  PrimExpr AsLaneMask(const PrimExpr &value, Array<Stmt> *pre) {
    if (value.dtype().lanes() > 1) {
      return value;
    }
    if (IsScalarBool(value)) {
      return ScalarBoolToMask(value, pre);
    }
    return value;
  }

  // One dispatch for every boolean combination, over operands that were each
  // converted exactly once. Verify admits the scalar domain, so a fully
  // lane-uniform combination must stay scalar.
  PrimExpr
  CombineBool(const PrimExpr &a, const PrimExpr &b,
              const std::function<PrimExpr(const PrimExpr &, const PrimExpr &)>
                  &scalar_form,
              const char *mask_op, Array<Stmt> *pre) {
    if (a.dtype().lanes() == 1 && b.dtype().lanes() == 1) {
      return scalar_form(a, b);
    }
    PrimExpr am = AsLaneMask(a, pre);
    PrimExpr bm = AsLaneMask(b, pre);
    DataType mask_ty = DataType::Bool(static_cast<int>(lanes_));
    ICHECK(am.dtype().lanes() > 1 && bm.dtype().lanes() > 1)
        << "[VectorizeParallelToPTO] " << mask_op
        << " expects lane masks (a non-bool scalar operand has no lane mask)";
    return EmitBind(mask_op, Call(mask_ty, VmiOp(mask_op), {am, bm}), pre);
  }

  // The unary counterpart of CombineBool (same scalar/mask rule).
  PrimExpr
  CombineNot(const PrimExpr &a,
             const std::function<PrimExpr(const PrimExpr &)> &scalar_form,
             Array<Stmt> *pre) {
    if (a.dtype().lanes() == 1) {
      return scalar_form(a);
    }
    PrimExpr am = AsLaneMask(a, pre);
    DataType mask_ty = DataType::Bool(static_cast<int>(lanes_));
    ICHECK(am.dtype().lanes() > 1)
        << "[VectorizeParallelToPTO] vnot expects a lane mask";
    return EmitBind("vnot", Call(mask_ty, VmiOp("vnot"), {am}), pre);
  }

  // The single compare dispatch. Every entry point (an expression compare and a
  // Select condition) reaches it, so the scalar/mask choice cannot diverge
  // between them: two lane-uniform operands keep the scalar relation, and only
  // a lane-varying operand becomes vcmp/vcmps.
  PrimExpr ConvertCompare(const PrimExpr &lhs, const PrimExpr &rhs,
                          const char *cmp, Array<Stmt> *pre) {
    PrimExpr a = ConvertExpr(lhs, pre);
    PrimExpr b = ConvertExpr(rhs, pre);
    // Comparisons participate in the unit mask. Expand a finalized one-lane
    // data register before choosing the scalar or vector comparison form.
    if (Info(a).IsVmi() && a.dtype().lanes() == 1)
      a = ToVmiWidth(a, lanes_, pre);
    if (Info(b).IsVmi() && b.dtype().lanes() == 1)
      b = ToVmiWidth(b, lanes_, pre);
    const pto::RelationalKind *kind = pto::RelationalByName(cmp);
    ICHECK(kind != nullptr)
        << "[VectorizeParallelToPTO] internal error: unknown cmp " << cmp;
    if (a.dtype().lanes() == 1 && b.dtype().lanes() == 1) {
      // Two lane-uniform operands: fold to the scalar bool with the shared
      // relational table's semantics. This is the domain Verify admitted, so a
      // uniform compare must not be promoted to a one-bit vector.
      return kind->fold(a, b);
    }
    // Boolean EQ/NE over masks (a mask with a scalar bool included):
    // vcmp/vcmps take vregs and the VMI element set has no one-bit bool, so
    // build the truth table out of vand/vor/vnot.
    if ((std::string(cmp) == "eq" || std::string(cmp) == "ne") &&
        a.dtype().is_bool() && b.dtype().is_bool()) {
      return BoolCompareToMask(a, b, cmp, pre);
    }
    DataType mask_ty = DataType::Bool(static_cast<int>(lanes_));
    const bool a_vec = a.dtype().lanes() > 1;
    const bool b_vec = b.dtype().lanes() > 1;
    if (a_vec && !b_vec) {
      return EmitBind(
          "vcmps",
          Call(mask_ty, VmiOp("vcmps"), {a, b, mask_var_, StringImm(cmp)}),
          pre);
    }
    if (!a_vec && b_vec) {
      a = BroadcastToLanes("brc", a, pre);
    }
    return EmitBind(
        "vcmp", Call(mask_ty, VmiOp("vcmp"), {a, b, mask_var_, StringImm(cmp)}),
        pre);
  }

  PrimExpr ConvertBitOp(const Call &call, pto_analysis::PtoBitOp op,
                        Array<Stmt> *pre) {
    using pto_analysis::PtoBitOp;
    PrimExpr value = ConvertExpr(call->args[0], pre);
    pto::ValueInfo value_info = Info(value);
    const bool vector = value_info.IsVmi();
    const DataType dtype = call->dtype;
    if (op == PtoBitOp::kReinterpret) {
      if (dtype == call->args[0].dtype()) {
        return value;
      }
      if (!vector) {
        return Call(dtype, call->op, {value}, call->span);
      }
      return EmitBind("reinterpret",
                      Call(VectorDType(dtype, value_info.logical_lanes),
                           VmiOp("vinterpret_cast"), {value},
                           {{"to_dtype", StringImm(TvmDtypeName(dtype))}}),
                      pre);
    }
    if (op == PtoBitOp::kNot) {
      if (!vector) {
        return Call(dtype, call->op, {value}, call->span);
      }
      int64_t width = value_info.logical_lanes;
      return EmitBind("vnot",
                      Call(VectorDType(dtype, width), VmiOp("vnot"),
                           {value, width == 1 ? GetFullMask(1) : mask_var_}),
                      pre);
    }
    PrimExpr rhs = ConvertExpr(call->args[1], pre);
    const bool rhs_vector = Info(rhs).IsVmi();
    if (!vector && !rhs_vector) {
      return Call(dtype, call->op, {value, rhs}, call->span);
    }
    const char *name;
    if (op == PtoBitOp::kXor) {
      name = "vxor";
    } else if (op == PtoBitOp::kAnd || op == PtoBitOp::kOr) {
      name = op == PtoBitOp::kAnd ? "vand" : "vor";
    } else if (rhs_vector) {
      name = op == PtoBitOp::kShiftLeft ? "vshl" : "vshr";
    } else {
      // The VMI scalar-shift ABI uses an i16 count for all data widths.
      name = op == PtoBitOp::kShiftLeft ? "vshls" : "vshrs";
      int64_t width = value_info.logical_lanes;
      return EmitBind(name,
                      Call(VectorDType(dtype, width), VmiOp(name),
                           {value, cast(DataType::Int(16), rhs),
                            width == 1 ? GetFullMask(1) : mask_var_}),
                      pre);
    }
    return EmitBinary(std::move(value), std::move(rhs), name, name, pre);
  }

  PrimExpr ConvertBinary(const PrimExpr &lhs, const PrimExpr &rhs,
                         const std::string &vmi_name, Array<Stmt> *pre) {
    PrimExpr a = ConvertExpr(lhs, pre);
    PrimExpr b = ConvertExpr(rhs, pre);
    return EmitBinary(std::move(a), std::move(b), vmi_name, vmi_name, pre);
  }

  PrimExpr ConvertFloor(const PrimExpr &lhs, const PrimExpr &rhs, bool is_mod,
                        Array<Stmt> *pre) {
    PrimExpr a = ConvertExpr(lhs, pre);
    PrimExpr b = ConvertExpr(rhs, pre);
    const pto::ValueInfo lhs_info = Info(a);
    if (!lhs_info.IsVmi() && !Info(b).IsVmi()) {
      return is_mod ? floormod(a, b) : floordiv(a, b);
    }

    int shift;
    if (!is_const_power_of_two_integer(b, &shift)) {
      const char *name = is_mod ? "vfloormod" : "vfloordiv";
      Var result = Downcast<Var>(EmitBinary(a, b, name, name, pre));
      TrackFloorResult(lhs, rhs, result);
      return result;
    }

    // VMI shifts signed data arithmetically and unsigned data logically.
    // Masking the low bits gives the floor remainder for either sign.
    if (is_mod) {
      return EmitBinary(
          a, make_const(a.dtype().element_of(), (int64_t{1} << shift) - 1),
          "vand", "floormod", pre);
    }
    return EmitBind(
        "floordiv",
        Call(a.dtype(), VmiOp("vshrs"),
             {a, IntImm(DataType::Int(16), shift),
              lhs_info.logical_lanes == 1 ? GetFullMask(1) : mask_var_}),
        pre);
  }

  PrimExpr EmitBinary(PrimExpr a, PrimExpr b, const std::string &vmi_name,
                      const std::string &result_name, Array<Stmt> *pre) {
    auto lhs_info = Info(a);
    auto rhs_info = Info(b);
    bool a_vec = lhs_info.IsVmi();
    bool b_vec = rhs_info.IsVmi();
    if (!a_vec && !b_vec) {
      // Lane-uniform folds stay scalar with the TIR semantics of the node.
      if (vmi_name == "vadd") {
        return a + b;
      }
      if (vmi_name == "vsub") {
        return a - b;
      }
      if (vmi_name == "vmul") {
        return a * b;
      }
      if (vmi_name == "vdiv")
        return div(a, b);
      ICHECK(false) << "[VectorizeParallelToPTO] internal error: unknown "
                       "binary op "
                    << vmi_name;
    }
    int64_t width = std::max(lhs_info.logical_lanes, rhs_info.logical_lanes);
    a = ToVmiWidth(a, width, pre);
    b = ToVmiWidth(b, width, pre);
    return EmitBind(result_name,
                    Call(a.dtype(), VmiOp(vmi_name),
                         {a, b, width == 1 ? GetFullMask(1) : mask_var_}),
                    pre);
  }

  // TVM dtype name identical to Python str(dtype) (vmi.py ABI):
  // float32 -> "float32", bfloat16 -> "bfloat16", int8 -> "int8", ...
  static std::string TvmDtypeName(DataType dtype) {
    std::ostringstream os;
    if (dtype.is_bfloat16()) {
      return "bfloat16";
    }
    // The FP8 spellings the PTO dtype parser recognizes; both are the TIR
    // dtype names, not the PTO target names.
    if (dtype.is_float8_e4m3fn()) {
      return "float8_e4m3fn";
    }
    if (dtype.is_float8_e5m2()) {
      return "float8_e5m2";
    }
    if (dtype.is_float()) {
      os << "float" << dtype.bits();
    } else if (dtype.is_int()) {
      os << "int" << dtype.bits();
    } else if (dtype.is_uint()) {
      os << "uint" << dtype.bits();
    } else if (dtype.is_bool()) {
      return "bool";
    } else {
      os << dtype;
    }
    return os.str();
  }

  // Broadcast a lane-uniform scalar to the current lane count (vbrc).
  PrimExpr BroadcastToLanes(const char *tag, const PrimExpr &scalar,
                            Array<Stmt> *pre) {
    pto_analysis::CheckBroadcastDType(scalar.dtype(), tag);
    return EmitBind(
        tag,
        Call(VectorDType(scalar.dtype(), lanes_), VmiOp("vbrc"), {scalar},
             {{"size", IntImm(DataType::Int(32), static_cast<int>(lanes_))}}),
        pre);
  }

  Var EmitBind(const std::string &name, const PrimExpr &value,
               Array<Stmt> *pre) {
    Var result(name + "_" + std::to_string(tmp_counter_++), value.dtype());
    pre->push_back(Bind(result, value));
    if (const auto *call = value.as<CallNode>())
      if (const auto *op = call->op.as<OpNode>();
          op && std::string(op->name).rfind("tl.vmi.", 0) == 0)
        value_infos_[result] = pto::ValueInfo::Vmi(value.dtype());
    return result;
  }

  pto::PtoReducerStates reducers_;
  std::unordered_map<Var, Buffer, ObjectPtrHash, ObjectPtrEqual> vf_buffers_;
  pto::ValueInfoMap value_infos_;
  std::map<int64_t, Var> full_masks_;
  Array<Stmt> mask_prelude_;
  pto::ParallelStorageQuery storage_;
  For current_unit_;
  std::unordered_map<Var, Buffer, ObjectPtrHash, ObjectPtrEqual> promoted_;
  arith::Analyzer root_analyzer_;
  arith::Analyzer *analyzer_ = &root_analyzer_;

  // Per-unit state.
  Var loop_var_;
  Var chunk_index_; // emitted chunk loop Var
  pto_analysis::PtoInverseMapping unit_mapping_;
  int64_t lanes_ = 0;
  Var mask_var_;
  int tmp_counter_ = 0;

  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> parameters_;
  std::vector<Buffer> active_allocations_;
  BufferVersionMap buffer_versions_;
  Map<Var, PrimExpr> bind_env_; // analysis env: original var -> def expr
  Map<Var, Var> value_map_;     // value env: original var -> converted var
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> address_vars_;
  std::vector<FloorGroup> floor_groups_;
  std::vector<size_t> available_floor_groups_;
};

} // namespace

namespace transform {

tvm::transform::Pass VectorizeParallelToPTO() {
  auto pass_func = [](PrimFunc f, const IRModule &m,
                      const tvm::transform::PassContext &ctx) {
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    if (!target.defined() || !TargetIsPTO(target.value())) {
      return f;
    }
    return VectorizeParallelToPTOImpl::Substitute(std::move(f));
  };
  return tvm::tirx::transform::CreatePrimFuncPass(
      pass_func, 0, "tl.VectorizeParallelToPTO", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.VectorizeParallelToPTO",
                        VectorizeParallelToPTO);
}

} // namespace transform

} // namespace tl
} // namespace tvm
