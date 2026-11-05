#include "ascend/pto_parallel_storage.h"
#include "ascend/layout/pto_index_analysis.h"
#include <limits>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

namespace tvm::tl::pto {
using namespace tirx;
using namespace ffi;

const Buffer &ParallelStorageQuery::Allocation(const Var &data) const {
  auto it = allocations_.find(data);
  ICHECK(it != allocations_.end())
      << "PTO Parallel ordinary storage has no actual allocation: " << data;
  return it->second;
}

void ParallelStorageQuery::Build(const Stmt &body) {
  allocations_.clear();
  owners_.clear();
  vmi::CollectAllocMarkers(body, &markers_, "PTO Parallel storage");
  auto add = [&](const Buffer &b) {
    auto [it, inserted] = allocations_.emplace(b->data, b);
    if (!inserted) {
      arith::Analyzer a;
      const Buffer &old = it->second;
      ICHECK(old->dtype == b->dtype && old.scope() == b.scope() &&
             old->shape.size() == b->shape.size())
          << "PTO Parallel conflicting allocation definitions: " << b->name;
      for (size_t i = 0; i < old->shape.size(); ++i)
        ICHECK(a.CanProveEqual(old->shape[i], b->shape[i]))
            << "PTO Parallel conflicting allocation shape: " << b->name;
      ICHECK(old->strides.size() == b->strides.size() &&
             a.CanProveEqual(old->elem_offset, b->elem_offset))
          << "PTO Parallel conflicting allocation layout: " << b->name;
      for (size_t i = 0; i < old->strides.size(); ++i)
        ICHECK(a.CanProveEqual(old->strides[i], b->strides[i]))
            << "PTO Parallel conflicting allocation strides: " << b->name;
    }
  };
  PostOrderVisit(body, [&](const ObjectRef &n) {
    if (const auto *b = n.as<SBlockNode>())
      for (const Buffer &buf : b->alloc_buffers)
        add(buf);
    if (const auto *a = n.as<AllocBufferNode>())
      add(a->buffer);
  });
  class Collector : public StmtVisitor {
  public:
    explicit Collector(ParallelStorageQuery *q) : q(q) {}
    void VisitStmt_(const SBlockNode *op) final {
      bool saved = in_vf;
      if (op->name_hint == "SIMD_VF")
        in_vf = true;
      StmtVisitor::VisitStmt_(op);
      in_vf = saved;
    }
    void VisitStmt_(const ForNode *op) final {
      auto found = op->annotations.find(attr::kParallelPrivateAllocs);
      if (found != op->annotations.end()) {
        ICHECK(in_vf && op->kind == ForKind::kParallel && !adjacent_inner)
            << "PTO Parallel private ownership must be on a SIMD_VF unit root";
        auto vars = (*found).second.try_cast<Array<Var>>();
        ICHECK(vars.has_value())
            << "PTO Parallel private ownership expects Array<Var>";
        For root = GetRef<For>(op);
        for (const Var &data : *vars) {
          const Buffer &b = q->Allocation(data);
          ICHECK((b.scope() == "local" || b.scope() == "local.var") &&
                 !q->markers_.kinds.count(data))
              << "PTO Parallel private ownership requires ordinary local "
                 "storage: "
              << b->name;
          ICHECK(q->owners_.emplace(data, root).second)
              << "PTO Parallel duplicate private ownership: " << b->name;
        }
      }
      bool saved = adjacent_inner;
      const auto *next = op->body.as<ForNode>();
      adjacent_inner = op->kind == ForKind::kParallel && next &&
                       next->kind == ForKind::kParallel;
      StmtVisitor::VisitStmt_(op);
      adjacent_inner = saved;
    }
    ParallelStorageQuery *q;
    bool in_vf = false, adjacent_inner = false;
  };
  Collector(this)(body);
}

StorageClass ParallelStorageQuery::Classify(const Buffer &b,
                                            const For &unit) const {
  auto owner = owners_.find(b->data);
  if (owner != owners_.end()) {
    ICHECK(unit.defined() && owner->second.same_as(unit))
        << "PTO Parallel private owner violation for buffer " << b->name;
    return StorageClass::kPrivate;
  }
  if (markers_.kinds.count(b->data))
    return StorageClass::kExplicitVMI;
  if (b.scope() == "local" || b.scope() == "local.var")
    return StorageClass::kExternal;
  if (b.scope() == "shared" || b.scope() == "shared.dyn")
    return StorageClass::kUB;
  return StorageClass::kUnsupported;
}

void ParallelStorageQuery::CheckStorageV1(const Buffer &b) const {
  const Buffer &a = Allocation(b->data);
  arith::Analyzer analyzer;
  ICHECK(a->shape.size() == 1 && analyzer.CanProveEqual(a->shape[0], 1))
      << "PTO Parallel ordinary storage v1 requires shape (1,): " << b->name;
  ICHECK(a->dtype.lanes() == 1 && a->dtype == b->dtype &&
         a.scope() == b.scope())
      << "PTO Parallel incompatible ordinary allocation dtype/scope: "
      << b->name;
}

SlotAccess ParallelStorageQuery::ResolveSlot(
    const Buffer &b, const Array<PrimExpr> &indices, arith::Analyzer *analyzer,
    const Map<Var, PrimExpr> &binds) const {
  CheckStorageV1(b);
  const Buffer &a = Allocation(b->data);
  // Accept only equivalent compiler remappings, not arbitrary alias views.
  ICHECK(a->shape.size() == b->shape.size() &&
         a->strides.size() == b->strides.size() &&
         indices.size() == b->shape.size())
      << "PTO Parallel incompatible ordinary access view: " << b->name;
  ICHECK(analyzer->CanProveEqual(a->elem_offset, b->elem_offset))
      << "PTO Parallel incompatible ordinary access offset: " << b->name;
  int64_t capacity = 1;
  for (size_t i = 0; i < a->shape.size(); ++i) {
    PrimExpr extent = analyzer->Simplify(a->shape[i]);
    const auto *e = extent.as<IntImmNode>();
    ICHECK(e && e->value > 0 &&
           capacity <= std::numeric_limits<int64_t>::max() / e->value)
        << "PTO Parallel requires positive static slot capacity: " << b->name;
    capacity *= e->value;
    ICHECK(analyzer->CanProveEqual(a->shape[i], b->shape[i]))
        << "PTO Parallel incompatible ordinary access shape: " << b->name;
  }
  for (size_t i = 0; i < a->strides.size(); ++i)
    ICHECK(analyzer->CanProveEqual(a->strides[i], b->strides[i]))
        << "PTO Parallel incompatible ordinary access strides: " << b->name;
  Array<PrimExpr> resolved;
  for (const PrimExpr &index : indices) {
    auto expanded = ResolveIndexAliases(index, binds);
    ICHECK(expanded.defined() && IsSupportedAddressExpr(expanded.value()))
        << "PTO Parallel ordinary slot must be a constant coordinate, not a "
           "loaded value: "
        << b->name;
    resolved.push_back(analyzer->Simplify(expanded.value()));
  }
  auto offsets = b->ElemOffset(resolved);
  ICHECK_EQ(offsets.size(), 1U);
  PrimExpr offset = analyzer->Simplify(offsets[0] - a->elem_offset);
  const auto *slot = offset.as<IntImmNode>();
  ICHECK(slot && slot->value >= 0 && slot->value < capacity)
      << "PTO Parallel ordinary slot must be constant and in bounds: "
      << b->name << " index=" << indices << " slot=" << offset;
  return {capacity, slot->value};
}

Array<Buffer> ParallelStorageQuery::PrivateAllocations(const For &unit) const {
  Array<Buffer> result;
  auto found = unit->annotations.find(attr::kParallelPrivateAllocs);
  if (found != unit->annotations.end())
    for (const Var &data : (*found).second.cast<Array<Var>>())
      result.push_back(Allocation(data));
  return result;
}

void ParallelStorageQuery::ValidateOwnerUses(const Stmt &body) const {
  class Uses : public StmtExprVisitor {
  public:
    explicit Uses(const ParallelStorageQuery *q) : q(q) {}
    void VisitStmt_(const ForNode *op) final {
      For saved = unit;
      if (op->kind == ForKind::kParallel && !unit.defined())
        unit = GetRef<For>(op);
      StmtExprVisitor::VisitStmt_(op);
      unit = saved;
    }
    void VisitExpr_(const BufferLoadNode *op) final {
      if (q->IsPrivate(op->buffer->data))
        q->Classify(op->buffer, unit);
      StmtExprVisitor::VisitExpr_(op);
      if (op->predicate.defined())
        VisitExpr(op->predicate.value());
    }
    void VisitStmt_(const BufferStoreNode *op) final {
      if (q->IsPrivate(op->buffer->data))
        q->Classify(op->buffer, unit);
      StmtExprVisitor::VisitStmt_(op);
      if (op->predicate.defined())
        VisitExpr(op->predicate.value());
    }
    void VisitExpr_(const VarNode *op) final {
      ICHECK(!q->IsPrivate(GetRef<Var>(op)))
          << "PTO Parallel unsupported private address escape: "
          << op->name_hint;
    }
    const ParallelStorageQuery *q;
    For unit;
  };
  Uses(this)(body);
}
} // namespace tvm::tl::pto
