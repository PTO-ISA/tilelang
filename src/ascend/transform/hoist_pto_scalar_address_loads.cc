// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>
#include <unordered_set>

#include "parallel_to_pto_utils.h"

namespace tvm::tl {
namespace {
using namespace tirx;
using namespace ffi;
using VarSet = std::unordered_set<const VarNode *>;

// A store through another Buffer over the same data still writes the scalar.
// If its address is exposed elsewhere, an unknown call or a distinct handle
// in the VF could also write it.
class ScalarUses : public StmtExprVisitor {
public:
  VarSet exposed;
  std::unordered_map<const SBlockNode *, VarSet> writes;

private:
  const SBlockNode *vf_ = nullptr;

  void VisitBufferDef(const Buffer &buffer, bool alloc_data) final {
    if (vf_ && alloc_data)
      writes[vf_].insert(buffer->data.get());
    StmtExprVisitor::VisitBufferDef(buffer, alloc_data);
  }
  void VisitStmt_(const AllocBufferNode *op) final {
    if (op->annotations.count(tirx::attr::kVolatile))
      exposed.insert(op->buffer->data.get());
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitExpr_(const VarNode *op) final {
    if (op->dtype.is_handle())
      exposed.insert(op);
  }
  void VisitExpr_(const CallNode *op) final {
    // Intrinsics can interpret a BufferLoad operand as an lvalue (address_of,
    // tl.access_ptr, tl.region, atomics). Treat any direct operand this way.
    for (const auto &arg : op->args)
      if (const auto *load = arg.as<BufferLoadNode>())
        exposed.insert(load->buffer->data.get());
    StmtExprVisitor::VisitExpr_(op);
  }
  void VisitExpr_(const BufferLoadNode *op) final {
    if (op->predicate)
      VisitExpr(*op->predicate);
    StmtExprVisitor::VisitExpr_(op);
  }
  void VisitStmt_(const BufferStoreNode *op) final {
    if (vf_)
      writes[vf_].insert(op->buffer->data.get());
    if (op->predicate)
      VisitExpr(*op->predicate);
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const AttrStmtNode *op) final {
    if (auto expr = op->node.try_cast<PrimExpr>())
      VisitExpr(*expr);
    if (auto buf = op->node.try_cast<Buffer>())
      exposed.insert((*buf)->data.get());
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const SBlockNode *op) final {
    for (const auto &match : op->match_buffers)
      exposed.insert(match->source->buffer->data.get());
    auto saved = vf_;
    if (op->name_hint == "SIMD_VF")
      vf_ = op;
    StmtExprVisitor::VisitStmt_(op);
    vf_ = saved;
  }
};

// Replace reads only in ordinary UB coordinates of converting Parallel units.
// Capture the load itself. Index arithmetic can also depend on VF-local vars.
class AddressRewriter : public StmtExprMutator {
public:
  explicit AddressRewriter(const Map<Buffer, bool> &initialized,
                           const VarSet &writes,
                           const std::vector<For> &units)
      : initialized_(initialized), writes_(writes) {
    for (const auto &unit : units)
      units_.insert(unit.get());
  }
  Array<Stmt> bindings;

private:
  Stmt VisitStmt_(const ForNode *op) final {
    bool saved = in_unit_;
    in_unit_ |= units_.count(op) != 0;
    auto result = StmtExprMutator::VisitStmt_(op);
    in_unit_ = saved;
    return result;
  }
  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    const Buffer &b = op->buffer;
    if (in_index_ && initialized_.Get(b).value_or(false) &&
        !writes_.count(b->data.get()) &&
        !op->predicate && op->indices.size() == 1 && is_zero(op->indices[0])) {
      if (!values_.count(b)) {
        Var value(b->name + "_value", b->dtype);
        bindings.push_back(Bind(value, GetRef<BufferLoad>(op)));
        values_.Set(b, value);
      }
      return values_[b];
    }
    BufferLoad load = GetRef<BufferLoad>(op);
    auto indices = RewriteIndices(b, op->indices);
    if (!indices.same_as(op->indices))
      load.CopyOnWrite()->indices = indices;
    return load;
  }
  Stmt VisitStmt_(const BufferStoreNode *op) final {
    auto value = VisitExpr(op->value);
    auto indices = RewriteIndices(op->buffer, op->indices);
    if (value.same_as(op->value) && indices.same_as(op->indices))
      return GetRef<Stmt>(op);
    return BufferStore(op->buffer, value, indices, op->predicate, op->span);
  }
  Array<PrimExpr> RewriteIndices(const Buffer &b, Array<PrimExpr> indices) {
    bool saved = in_index_;
    in_index_ = in_unit_ && pto::IsOrdinaryUbScope(b.scope());
    indices.MutateByApply([&](PrimExpr e) { return VisitExpr(e); });
    in_index_ = saved;
    return indices;
  }
  bool in_index_ = false;
  bool in_unit_ = false;
  std::unordered_set<const ForNode *> units_;
  Map<Buffer, Var> values_;
  const Map<Buffer, bool> &initialized_;
  const VarSet &writes_;
};

class ScalarAddressHoister : public StmtMutator {
public:
  explicit ScalarAddressHoister(const Stmt &body) {
    uses_(body);
    storage_.Build(body);
  }

private:
  void Register(const Buffer &b) {
    if (b.scope() == "local.var" && b->dtype.is_scalar() &&
        (b->dtype.is_int() || b->dtype.is_uint()) && !b->dtype.is_bool() &&
        b->shape.size() == 1 && is_one(b->shape[0]) && is_zero(b->elem_offset) &&
        !uses_.exposed.count(b->data.get()) && !storage_.IsPrivate(b->data) &&
        storage_.Classify(b) == pto::StorageClass::kExternal)
      initialized_.Set(b, false);
  }

  Stmt VisitStmt_(const SeqStmtNode *op) final {
    // Only allocations and unconditional stores in preceding siblings
    // dominate this statement. Facts from a branch or loop stay in its body.
    auto saved = initialized_;
    auto result = VisitSeqStmt_(op, false, [&](const Stmt &stmt) {
      if (const auto *alloc = stmt.as<AllocBufferNode>()) {
        Register(alloc->buffer);
      } else if (const auto *st = stmt.as<BufferStoreNode>()) {
        if (initialized_.count(st->buffer) && !st->predicate &&
            st->indices.size() == 1 && is_zero(st->indices[0]))
          initialized_.Set(st->buffer, true);
      }
      return VisitStmt(stmt);
    });
    initialized_ = saved;
    return result;
  }
  Stmt VisitStmt_(const SBlockNode *op) final {
    auto block = GetRef<SBlock>(op);
    if (op->name_hint == "SIMD_VF")
      return RewriteVF(block, {});
    if (op->name_hint == "SIMT_VF")
      return block;
    auto saved = initialized_;
    for (const Buffer &b : op->alloc_buffers)
      Register(b);
    auto result = StmtMutator::VisitStmt_(op);
    initialized_ = saved;
    return result;
  }
  Stmt VisitStmt_(const SBlockRealizeNode *op) final {
    if (op->block->name_hint == "SIMD_VF")
      return RewriteVF(op->block, GetRef<SBlockRealize>(op));
    return StmtMutator::VisitStmt_(op);
  }
  Stmt RewriteVF(SBlock block, SBlockRealize realize) {
    Stmt original = realize.defined() ? Stmt(realize) : Stmt(block);
    if (initialized_.empty() || block->init || !block->iter_vars.empty() ||
        (realize.defined() && !realize->iter_values.empty()))
      return original;
    auto scan = pto::ScanPtoRegion(block->body);
    if (scan.units.empty() || scan.has_unsupported_wrapper)
      return original;
    AddressRewriter rewrite(initialized_, uses_.writes[block.get()], scan.units);
    auto body = rewrite(block->body);
    if (rewrite.bindings.empty())
      return original;
    block.CopyOnWrite()->body = body;
    if (realize.defined()) {
      rewrite.bindings.push_back(SBlockRealize({}, const_true(), block, realize->span));
    } else {
      rewrite.bindings.push_back(block);
    }
    auto result = SeqStmt::Flatten(rewrite.bindings);
    if (realize.defined() && !is_one(realize->predicate))
      return IfThenElse(realize->predicate, result);
    return result;
  }
  ScalarUses uses_;
  pto::ParallelStorageQuery storage_;
  Map<Buffer, bool> initialized_;
};
} // namespace

namespace transform {
tvm::transform::Pass HoistPTOScalarAddressLoads() {
  auto pass = [](PrimFunc f, const IRModule &, const tvm::transform::PassContext &) {
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    if (target && TargetIsPTO(*target)) {
      auto body = ScalarAddressHoister(f->body)(f->body);
      if (!body.same_as(f->body))
        f.CopyOnWrite()->body = body;
    }
    return f;
  };
  return tirx::transform::CreatePrimFuncPass(pass, 0, "tl.HoistPTOScalarAddressLoads", {});
}
TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.HoistPTOScalarAddressLoads",
                                   HoistPTOScalarAddressLoads);
}
} // namespace transform
} // namespace tvm::tl
