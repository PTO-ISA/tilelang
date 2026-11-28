/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*! \file materialize_multi_buffer.cc
 *  \brief Lower prepared logical multi-buffer clocks to physical buffer
 *         versions after synchronization insertion.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "./auto_schedule/kernel_rewriter.h"
#include "./auto_schedule/multi_buffer.h"
#include "./auto_schedule/scheduled_tir.h"
#include "ascend/layout/pto_index_analysis.h"
#include "ascend/target_utils.h"
#include "layout/layout.h"
#include "op/builtin.h"
#include "transform/common/attr.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ascend;
using ffi::GetRef;
using ffi::Map;

namespace {

PrimExpr BufferStoragePitch(const Buffer &buffer) {
  if (!buffer->strides.empty())
    return buffer->strides[0] * buffer->shape[0];
  DataType index_dtype =
      buffer->shape.empty() ? DataType::Int(32) : buffer->shape[0].dtype();
  PrimExpr pitch = make_const(index_dtype, 1);
  for (const PrimExpr &extent : buffer->shape)
    pitch = pitch * extent;
  return pitch;
}

PrimExpr BufferStoragePitchBits(const Buffer &buffer) {
  PrimExpr pitch = BufferStoragePitch(buffer);
  int element_bits = buffer->dtype.bits() * buffer->dtype.lanes();
  return pitch * make_const(pitch.dtype(), element_bits);
}

PrimExpr StoragePitchInElements(const PrimExpr &storage_pitch_bits,
                                const Buffer &buffer) {
  int element_bits = buffer->dtype.bits() * buffer->dtype.lanes();
  PrimExpr divisor = make_const(storage_pitch_bits.dtype(), element_bits);
  arith::Analyzer analyzer;
  ICHECK(analyzer.CanProveEqual(indexmod(storage_pitch_bits, divisor), 0))
      << "Automatic multi-buffer storage pitch " << storage_pitch_bits
      << " bits is not divisible by alias element width " << element_bits
      << " for buffer " << buffer->name;
  return analyzer.Simplify(indexdiv(storage_pitch_bits, divisor));
}

Array<PrimExpr> ExplicitStrides(const Buffer &buffer) {
  if (!buffer->strides.empty())
    return buffer->strides;
  if (buffer->shape.empty())
    return {};
  std::vector<PrimExpr> compact(buffer->shape.size());
  PrimExpr stride = make_const(buffer->shape.back().dtype(), 1);
  for (size_t i = buffer->shape.size(); i > 0; --i) {
    compact[i - 1] = stride;
    stride = stride * buffer->shape[i - 1];
  }
  return Array<PrimExpr>(compact.begin(), compact.end());
}

Buffer CreateMultiVersionBuffer(const Buffer &buffer, int num_versions,
                                const PrimExpr &storage_pitch_bits) {
  ObjectPtr<BufferNode> versioned =
      tvm::ffi::make_object<BufferNode>(*(buffer.get()));
  versioned->shape.insert(versioned->shape.begin(), PrimExpr(num_versions));
  Array<PrimExpr> strides{StoragePitchInElements(storage_pitch_bits, buffer)};
  for (const PrimExpr &stride : ExplicitStrides(buffer))
    strides.push_back(stride);
  versioned->strides = std::move(strides);
  return Buffer(versioned);
}

// The scheduler owns a fresh counter allocation. If its address escapes
// anywhere in the kernel, a VF could update it through an unrelated handle.
// TIRx's default visitors omit optional BufferLoad/BufferStore predicates.
class CounterAddressUses : public StmtExprVisitor {
public:
  bool Contains(const Var &storage) const { return addresses_.count(storage); }

protected:
  void VisitExpr_(const VarNode *op) final {
    addresses_.insert(GetRef<Var>(op));
  }
  void VisitExpr_(const BufferLoadNode *op) final {
    if (op->predicate)
      VisitExpr(op->predicate.value());
    StmtExprVisitor::VisitExpr_(op);
  }
  void VisitStmt_(const BufferStoreNode *op) final {
    if (op->predicate)
      VisitExpr(op->predicate.value());
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const SBlockNode *op) final {
    for (const auto &match : op->match_buffers)
      addresses_.insert(match->source->buffer->data);
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->node.as<PrimExprNode>())
      VisitExpr(Downcast<PrimExpr>(op->node));
    StmtExprVisitor::VisitStmt_(op);
  }
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> addresses_;
};

class VersionedBufferRegistry {
public:
  VersionedBufferRegistry(const MultiBufferPlan &plan,
                          const SBlock &kernel_root, bool pto) {
    if (pto)
      counter_addresses_(kernel_root);
    for (const MultiBufferInfo &info : plan.Infos()) {
      if (info.NeedsVersionDimension())
        storage_versions_.emplace(info.storage, info.num_versions);
    }
    CollectAllocationPitches(kernel_root);
    for (const MultiBufferInfo &info : plan.Infos()) {
      if (!info.NeedsVersionDimension())
        continue;
      ICHECK(storage_pitch_bits_.count(info.storage))
          << "Cannot find the physical allocation for automatic multi-buffer "
             "storage "
          << info.storage->name_hint;
    }
  }

  bool CanCaptureCounter(const Var &storage) const {
    return !counter_addresses_.Contains(storage);
  }

  Buffer Resolve(const Buffer &buffer) {
    if (expanded_buffers_.count(buffer))
      return buffer;
    auto exact = buffer_remap_.find(buffer);
    if (exact != buffer_remap_.end())
      return exact->second;
    auto storage = storage_versions_.find(buffer->data);
    if (storage == storage_versions_.end())
      return buffer;
    Buffer expanded = CreateMultiVersionBuffer(
        buffer, storage->second, storage_pitch_bits_.at(buffer->data));
    buffer_remap_.emplace(buffer, expanded);
    expanded_buffers_.insert(expanded);
    return expanded;
  }

private:
  void RegisterAllocation(const Buffer &buffer) {
    if (!storage_versions_.count(buffer->data))
      return;
    PrimExpr pitch_bits = BufferStoragePitchBits(buffer);
    auto [it, inserted] = storage_pitch_bits_.emplace(buffer->data, pitch_bits);
    ICHECK(inserted || ffi::StructuralEqual()(it->second, pitch_bits))
        << "Automatic multi-buffer storage " << buffer->data->name_hint
        << " has inconsistent physical allocation pitches";
  }

  void CollectAllocationPitches(const SBlock &kernel_root) {
    class Collector : public StmtVisitor {
    public:
      explicit Collector(VersionedBufferRegistry *registry)
          : registry_(registry) {}

      Map<Var, PrimExpr> layout_pitch_bits;

    private:
      void VisitStmt_(const SBlockNode *op) final {
        for (const Buffer &buffer : op->alloc_buffers)
          registry_->RegisterAllocation(buffer);
        if (auto value = op->annotations.Get(attr::kLayoutMap)) {
          for (const auto &[buffer, layout] :
               value.value().cast<Map<Buffer, Layout>>()) {
            if (!registry_->storage_versions_.count(buffer->data) ||
                layout.as<FragmentNode>())
              continue;
            PrimExpr bits =
                make_const(buffer->DefaultIndexType(),
                           buffer->dtype.bits() * buffer->dtype.lanes());
            for (const PrimExpr &extent : layout->OutputShape())
              bits = bits * extent;
            arith::Analyzer analyzer;
            bits = analyzer.Simplify(bits);
            if (auto previous = layout_pitch_bits.Get(buffer->data))
              ICHECK(analyzer.CanProveEqual(previous.value(), bits))
                  << "Automatic multi-buffer storage "
                  << buffer->data->name_hint
                  << " has inconsistent layout storage pitches";
            layout_pitch_bits.Set(buffer->data, bits);
          }
        }
        StmtVisitor::VisitStmt_(op);
      }

      void VisitStmt_(const AllocBufferNode *op) final {
        registry_->RegisterAllocation(op->buffer);
        StmtVisitor::VisitStmt_(op);
      }

      VersionedBufferRegistry *registry_;
    } collector(this);
    collector(kernel_root);
    // Layout lowering sizes the allocation from its physical output shape.
    // Ordinary views must use the same version pitch without inheriting the
    // annotated view's within-version index mapping.
    for (const auto &[storage, bits] : collector.layout_pitch_bits) {
      auto allocation = storage_pitch_bits_.find(storage);
      if (allocation != storage_pitch_bits_.end())
        allocation->second = bits;
    }
  }

  CounterAddressUses counter_addresses_;
  std::unordered_map<Var, int, ObjectPtrHash, ObjectPtrEqual> storage_versions_;
  std::unordered_map<Var, PrimExpr, ObjectPtrHash, ObjectPtrEqual>
      storage_pitch_bits_;
  std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual>
      buffer_remap_;
  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> expanded_buffers_;
};

// Inspect the logical VF before adding version indices. A prepared counter is
// allocated and initialized by the scheduler outside the task. If its storage
// already occurs in the VF (including an alias or a pointer argument), keep
// the original load at the use site.
class VFVersionSlots : public CounterAddressUses {
public:
  bool has_parallel{false};
  Array<Stmt> bindings;

  PrimExpr Capture(const Var &counter, const PrimExpr &index) {
    if (!has_parallel || Contains(counter))
      return index;
    auto &slots = by_counter_[counter];
    for (const Bind &binding : slots) {
      if (ffi::StructuralEqual()(binding->value, index))
        return binding->var;
    }
    Var slot("version_slot", index.dtype());
    Bind binding(slot, index);
    slots.push_back(binding);
    bindings.push_back(binding);
    return slot;
  }

private:
  void VisitStmt_(const ForNode *op) final {
    has_parallel |= op->kind == ForKind::kParallel;
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitBufferUse(const Buffer &buffer) final {
    addresses_.insert(buffer->data);
  }
  void VisitBufferDef(const Buffer &buffer, bool alloc_data) final {
    addresses_.insert(buffer->data);
    StmtExprVisitor::VisitBufferDef(buffer, alloc_data);
  }

  std::unordered_map<Var, std::vector<Bind>, ObjectPtrHash, ObjectPtrEqual>
      by_counter_;
};

class MultiBufferAccessRewriter : public StmtExprMutator {
public:
  MultiBufferAccessRewriter(const MultiBufferPlan &plan,
                            VersionedBufferRegistry *buffers,
                            const MultiBufferBroadcastFill &broadcast_storages,
                            bool pto)
      : plan_(plan), buffers_(buffers), pto_(pto) {
    for (const Var &storage : broadcast_storages) {
      const MultiBufferInfo *info = plan_.Find(storage);
      ICHECK(info != nullptr)
          << "Broadcast initialization refers to storage outside the prepared "
             "multi-buffer plan: "
          << storage;
      if (info->NeedsVersionDimension())
        broadcast_storages_.insert(storage);
    }
  }

  ffi::Any RewriteAttributeNode(const ffi::String &attr_key,
                                const ffi::Any &node) {
    if (!CanRewriteMultiBufferAttrNode(attr_key, node))
      return node;
    return VisitExpr(Downcast<PrimExpr>(node));
  }

  PrimExpr RewriteGuardExpression(const PrimExpr &expression) {
    return VisitExpr(expression);
  }

private:
  bool IsBroadcastFill(const Buffer &buffer) const {
    return broadcast_storages_.count(buffer->data) != 0;
  }

  PrimExpr VersionIndex(const MultiBufferInfo &info,
                        const Buffer &versioned) const {
    if (broadcast_storages_.count(info.storage))
      return make_zero(versioned->shape[0].dtype());
    PrimExpr iteration =
        info.UsesCounter()
            ? PrimExpr(BufferLoad(info.counter, {IntImm(DataType::Int(32), 0)}))
            : CalculateIterationCount(info.owners.front().loop);
    PrimExpr version =
        indexmod(iteration, IntImm(iteration.dtype(), info.num_versions));
    DataType index_dtype = versioned->shape[0].dtype();
    if (version.dtype() != index_dtype)
      version = cast(index_dtype, version);
    if (vf_slots_ && info.UsesCounter() &&
        pto::IsOrdinaryUbScope(versioned.scope()) &&
        buffers_->CanCaptureCounter(info.counter->data))
      return vf_slots_->Capture(info.counter->data, version);
    return version;
  }

  Stmt RewriteVF(const SBlock &block, const SBlockRealizeNode *realize) {
    VFVersionSlots slots;
    slots(block);
    VFVersionSlots *previous = vf_slots_;
    vf_slots_ = &slots;
    SBlock rewritten =
        Downcast<SBlock>(StmtExprMutator::VisitStmt_(block.get()));
    vf_slots_ = previous;
    if (!realize)
      return SeqStmt::Flatten(slots.bindings, rewritten);

    auto values = realize->iter_values.Map(
        [this](const PrimExpr &value) { return VisitExpr(value); });
    PrimExpr pred = VisitExpr(realize->predicate);
    if (slots.bindings.empty())
      return SBlockRealize(values, pred, rewritten, realize->span);
    Stmt body = SeqStmt::Flatten(
        slots.bindings,
        SBlockRealize(values, Bool(true), rewritten, realize->span));
    return is_one(pred) ? body : IfThenElse(pred, body);
  }

  Stmt VisitStmt_(const SBlockNode *op) final {
    if (pto_ && op->name_hint == "SIMD_VF")
      return RewriteVF(GetRef<SBlock>(op), nullptr);
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const SBlockRealizeNode *op) final {
    if (pto_ && op->block->name_hint == "SIMD_VF")
      return RewriteVF(op->block, op);
    return StmtExprMutator::VisitStmt_(op);
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    const MultiBufferInfo *info = plan_.Find(load->buffer);
    if (!info || !info->NeedsVersionDimension())
      return load;
    Buffer versioned = buffers_->Resolve(load->buffer);
    auto *node = load.CopyOnWrite();
    node->buffer = versioned;
    node->indices.insert(node->indices.begin(), VersionIndex(*info, versioned));
    return load;
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    const MultiBufferInfo *info = plan_.Find(store->buffer);
    if (!info || !info->NeedsVersionDimension())
      return store;
    Buffer versioned = buffers_->Resolve(store->buffer);
    auto *node = store.CopyOnWrite();
    node->buffer = versioned;
    node->indices.insert(node->indices.begin(), VersionIndex(*info, versioned));
    return store;
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    Stmt rewritten = StmtExprMutator::VisitStmt_(op);
    if (!CanRewriteMultiBufferAttrNode(op->attr_key, op->node))
      return rewritten;
    AttrStmt attribute = Downcast<AttrStmt>(rewritten);
    attribute.CopyOnWrite()->node =
        RewriteAttributeNode(op->attr_key, op->node);
    return attribute;
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    static const Op &region_op = region();
    if (op->op.same_as(region_op) && op->args.size() >= 2) {
      if (const auto *load = op->args[0].as<BufferLoadNode>()) {
        const MultiBufferInfo *info = plan_.Find(load->buffer);
        if (info != nullptr && info->NeedsVersionDimension()) {
          bool broadcast = IsBroadcastFill(load->buffer);
          Array<PrimExpr> args;
          args.push_back(VisitExpr(op->args[0]));
          args.push_back(VisitExpr(op->args[1]));
          args.push_back(broadcast ? buffers_->Resolve(load->buffer)->shape[0]
                                   : PrimExpr(IntImm(DataType::Int(32), 1)));
          for (size_t i = 2; i < op->args.size(); ++i)
            args.push_back(VisitExpr(op->args[i]));
          return Call(op->dtype, op->op, args, op->annotations, op->span);
        }
      }
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  const MultiBufferPlan &plan_;
  VersionedBufferRegistry *buffers_;
  bool pto_;
  VFVersionSlots *vf_slots_{nullptr};
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> broadcast_storages_;
};

void RewriteGuards(IRStructure *node, MultiBufferAccessRewriter *rewriter) {
  GuardList guards;
  guards.reserve(node->GetGuards().size());
  for (const auto &guard : node->GetGuards()) {
    if (guard->IsCondition()) {
      const auto *condition = static_cast<const ConditionGuard *>(guard.get());
      guards.push_back(std::make_unique<ConditionGuard>(
          rewriter->RewriteGuardExpression(condition->condition)));
      continue;
    }
    const auto *attribute = static_cast<const AttributeGuard *>(guard.get());
    guards.push_back(std::make_unique<AttributeGuard>(
        rewriter->RewriteAttributeNode(attribute->key, attribute->node),
        attribute->key, attribute->value, attribute->span));
  }
  node->SetGuards(std::move(guards));
}

void RewriteScheduledTree(std::vector<std::shared_ptr<IRStructure>> *nodes,
                          const MultiBufferPlan &plan,
                          VersionedBufferRegistry *buffers, bool pto) {
  for (const auto &node : *nodes) {
    MultiBufferBroadcastFill broadcast_storages;
    if (node->IsTask()) {
      auto *task = static_cast<TaskNode *>(node.get());
      broadcast_storages = GetMultiBufferBroadcastFills(task);
      MultiBufferAccessRewriter rewriter(plan, buffers, broadcast_storages,
                                         pto);
      RewriteGuards(task, &rewriter);
      task->stmt = rewriter(task->stmt);
      if (!broadcast_storages.empty())
        ClearMultiBufferBroadcastFills(task);
      continue;
    }
    auto *control = static_cast<ControlNode *>(node.get());
    MultiBufferAccessRewriter rewriter(plan, buffers, broadcast_storages, pto);
    RewriteGuards(control, &rewriter);
    RewriteScheduledTree(&control->children, plan, buffers, pto);
    For loop = control->control;
    auto *for_node = loop.CopyOnWrite();
    for_node->annotations.erase(kMultiBufferCounterMap);
    for_node->annotations.erase(kStorageEpochGuardMap);
    for_node->annotations.erase(kWriteCounterUpdateGuardMap);
    control->control = std::move(loop);
  }
}

Layout ExpandLayoutToBuffer(const Buffer &layout_buffer,
                            const Buffer &target_buffer, const Layout &layout) {
  if (!layout.defined() ||
      target_buffer->shape.size() < layout_buffer->shape.size()) {
    return Layout();
  }
  size_t leading_ndim =
      target_buffer->shape.size() - layout_buffer->shape.size();
  if (leading_ndim == 0)
    return layout;
  Array<PrimExpr> leading_shape;
  for (size_t i = 0; i < leading_ndim; ++i)
    leading_shape.push_back(target_buffer->shape[i]);
  return layout->Expand(leading_shape);
}

class AllocationRewriter : public StmtMutator {
public:
  explicit AllocationRewriter(VersionedBufferRegistry *buffers)
      : buffers_(buffers) {}

private:
  Stmt VisitStmt_(const AllocBufferNode *op) final {
    Buffer rewritten = buffers_->Resolve(op->buffer);
    if (rewritten.same_as(op->buffer))
      return StmtMutator::VisitStmt_(op);
    return AllocBuffer(rewritten, op->annotations, op->span);
  }

  Stmt VisitStmt_(const SBlockNode *op) final {
    Stmt body = VisitStmt(op->body);
    Array<Buffer> alloc_buffers;
    bool buffers_changed = false;
    for (const Buffer &buffer : op->alloc_buffers) {
      Buffer rewritten = buffers_->Resolve(buffer);
      buffers_changed |= !rewritten.same_as(buffer);
      alloc_buffers.push_back(rewritten);
    }

    Map<String, Any> annotations = op->annotations;
    bool layout_changed = false;
    if (auto value = op->annotations.Get(attr::kLayoutMap)) {
      if (auto layout_map = value.value().as<Map<Buffer, Layout>>()) {
        Map<Buffer, Layout> rewritten_layouts = layout_map.value();
        // Expand each view from its own layout. Data-Var lookup is ambiguous
        // when several canonical matrix views share a versioned allocation.
        for (const auto &[buffer, layout] : layout_map.value()) {
          Buffer expanded = buffers_->Resolve(buffer);
          if (!expanded.same_as(buffer)) {
            rewritten_layouts.Set(
                expanded, ExpandLayoutToBuffer(buffer, expanded, layout));
            layout_changed = true;
          }
        }
        if (layout_changed)
          annotations.Set(attr::kLayoutMap, rewritten_layouts);
      }
    }

    SBlock rewritten = GetRef<SBlock>(op);
    SBlockNode *node = rewritten.CopyOnWrite();
    node->body = std::move(body);
    if (buffers_changed)
      node->alloc_buffers = std::move(alloc_buffers);
    if (layout_changed)
      node->annotations = std::move(annotations);
    return rewritten;
  }

  VersionedBufferRegistry *buffers_;
};

void AddBufferVersion(BufferVersionMap *versions, const Var &storage,
                      int num_versions) {
  if (num_versions <= 1)
    return;
  if (auto existing = versions->Get(storage)) {
    ICHECK_EQ(existing.value(), num_versions)
        << "Conflicting physical buffer version counts for " << storage;
  } else {
    versions->Set(storage, num_versions);
  }
}

void RemoveCounterTasks(
    std::vector<std::shared_ptr<IRStructure>> *nodes, ControlNode *parent,
    const std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>
        &counter_storages) {
  std::vector<std::shared_ptr<IRStructure>> retained;
  retained.reserve(nodes->size());
  for (const std::shared_ptr<IRStructure> &node : *nodes) {
    if (node->IsTask()) {
      auto *task = static_cast<TaskNode *>(node.get());
      auto counter = std::find_if(
          counter_storages.begin(), counter_storages.end(),
          [&](const Var &storage) { return task->TouchesStorage(storage); });
      if (counter != counter_storages.end()) {
        // Only the generated initialization/advance store may disappear.
        // A reader or a compound task can carry work that must survive.
        Stmt body = task->stmt;
        if (const auto *marker = body.as<AttrStmtNode>();
            marker && marker->attr_key == attr::kAscendTask) {
          body = marker->body;
        }
        while (const auto *guard = body.as<IfThenElseNode>()) {
          if (guard->else_case.defined())
            break;
          body = guard->then_case;
        }
        const auto *store = body.as<BufferStoreNode>();
        ICHECK(store && store->buffer->data.same_as(*counter))
            << "Cannot elide single-version counter " << (*counter)->name_hint
            << ": expected a task that only stores to this counter";
        continue;
      }
    } else {
      auto *control = static_cast<ControlNode *>(node.get());
      for (const Var &storage : counter_storages) {
        ICHECK(!control->task->TouchesStorage(storage))
            << "Cannot elide single-version counter " << storage->name_hint
            << ": its value is used by loop control";
      }
      RemoveCounterTasks(&control->children, control, counter_storages);
    }
    node->SetIndex(retained.size());
    node->SetParent(parent);
    retained.push_back(node);
  }
  *nodes = std::move(retained);
}

void ElideUnitVersionCounterGroups(ScheduledTIR *scheduled_tir,
                                   const MultiBufferPlan &plan) {
  std::unordered_set<int> versioned_counter_groups;
  for (const MultiBufferInfo &info : plan.Infos()) {
    if (info.UsesCounter() &&
        (info.NeedsVersionDimension() || info.write_counter))
      versioned_counter_groups.insert(info.counter_group_id);
  }

  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> counter_storages;
  for (const MultiBufferInfo &info : plan.Infos()) {
    if (info.UsesCounter() &&
        !versioned_counter_groups.count(info.counter_group_id))
      counter_storages.insert(info.counter->data);
  }
  if (counter_storages.empty())
    return;

  RemoveCounterTasks(&scheduled_tir->tree, nullptr, counter_storages);
  Array<Buffer> alloc_buffers;
  for (const Buffer &buffer :
       scheduled_tir->metadata.kernel_root->alloc_buffers) {
    if (!counter_storages.count(buffer->data))
      alloc_buffers.push_back(buffer);
  }
  SBlock kernel_root = scheduled_tir->metadata.kernel_root;
  kernel_root.CopyOnWrite()->alloc_buffers = std::move(alloc_buffers);
  scheduled_tir->metadata.kernel_root = std::move(kernel_root);
}

SBlock MaterializeKernel(ScheduledTIR scheduled_tir, bool pto) {
  MultiBufferPlan plan = ReadMultiBufferPlan(
      scheduled_tir.tree, scheduled_tir.metadata.buffer_versions,
      L0StorageGroups(CollectL0SFBindings(scheduled_tir.metadata.kernel_root)));
  ElideUnitVersionCounterGroups(&scheduled_tir, plan);
  VersionedBufferRegistry buffers(plan, scheduled_tir.metadata.kernel_root,
                                  pto);
  RewriteScheduledTree(&scheduled_tir.tree, plan, &buffers, pto);

  BufferVersionMap physical_versions;
  for (const auto &[storage, num_versions] :
       scheduled_tir.metadata.manual_buffer_versions) {
    AddBufferVersion(&physical_versions, storage, num_versions);
  }
  for (const MultiBufferInfo &info : plan.Infos())
    AddBufferVersion(&physical_versions, info.storage, info.num_versions);
  scheduled_tir.metadata.buffer_versions = {};
  scheduled_tir.metadata.manual_buffer_versions = std::move(physical_versions);

  Stmt serialized = EncodeScheduledTIR(std::move(scheduled_tir));
  serialized = AllocationRewriter(&buffers)(std::move(serialized));
  const auto *root = serialized.as<SBlockNode>();
  ICHECK(root) << "Multi-buffer materialization must preserve tilelang_root";
  return GetRef<SBlock>(root);
}

} // namespace

tvm::transform::Pass MaterializeMultiBuffer() {
  using namespace tirx::transform;
  auto pass_func = [](PrimFunc func, const IRModule &,
                      const tvm::transform::PassContext &) -> PrimFunc {
    auto target = func->GetAttr<Target>(tvm::attr::kTarget);
    bool pto = target.defined() && TargetIsPTO(target.value());
    return RewriteTilelangKernels(
        std::move(func), "MaterializeMultiBuffer",
        [pto](const TilelangKernelContext &context) {
          ScheduledTIR scheduled_tir =
              DecodeScheduledTIR(context.root, context.outer_ctx);
          return MaterializeKernel(std::move(scheduled_tir), pto);
        });
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.MaterializeMultiBuffer", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.MaterializeMultiBuffer",
                        MaterializeMultiBuffer);
}

} // namespace tl
} // namespace tvm
