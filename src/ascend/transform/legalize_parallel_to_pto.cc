/*!
 * \file legalize_parallel_to_pto.cc
 * \brief LegalizeParallelToPTO: safe scalar rewrites for the control-flow
 * forms VerifyParallelToPTO admitted in a converting PTO SIMD_VF region.
 *
 * Pipeline position: `VerifyParallelToPTO -> LegalizeParallelToPTO ->
 * VectorizeParallelToPTO` (target-gated on PTO). The pass rewrites the
 * scalar parallel TIR and produces scalar parallel TIR: lane-varying
 * If/Then/Else become either a value choice (both arms write the same
 * destination) or a predicated store (one-sided branch; nested conditions
 * conjoin their predicates), Min/Max become comparisons + Select, and the
 * tl/tir.infinity() constant becomes FloatImm, so the vectorizer only
 * needs vcmp/vcmps + vsel + a masked vstore. A branch that writes a buffer
 * and then reads it back (or reads it and then overwrites it) is emitted in
 * its source order with the branch predicate, so the access sequence the
 * source implies is preserved; ordering inside one vecscope is the backend's
 * job (PTOAS analyses intra-vecscope UB dependencies and inserts mem_bar), so
 * this pass neither rejects those shapes nor inserts synchronisation itself.
 * It never emits VMI and never calls a reducer.
 * A shape whose condition semantics cannot be preserved by the rewrite is
 * reported instead of being silently widened.
 *
 * The lane-uniform store specialisation is intentionally NOT part of this
 * pass: Verify already rejects lane-uniform stores, and this pass fails
 * defensively on that shape instead of materialising a reducer.
 *
 * This pass descends only into converting SIMD_VF blocks (those containing
 * a `T.Parallel` unit — the same predicate Verify uses); non-converting
 * regions are returned unchanged, so their Min/Max keep their code shape.
 */

#include <tvm/runtime/logging.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <limits>
#include <set>
#include <sstream>
#include <vector>

#include "parallel_to_pto_utils.h"
#include "support/check.h"

#include "../target_utils.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

[[noreturn]] void Fail(const std::string &msg) {
  LOG(FATAL) << "[LegalizeParallelToPTO] " << msg;
}

bool SameAccess(const BufferStoreNode *a, const BufferStoreNode *b) {
  if (!a->buffer->data.same_as(b->buffer->data)) {
    return false;
  }
  if (a->indices.size() != b->indices.size()) {
    return false;
  }
  for (size_t i = 0; i < a->indices.size(); ++i) {
    if (!ffi::StructuralEqual()(a->indices[i], b->indices[i])) {
      return false;
    }
  }
  return true;
}

// Collect a flat list of BufferStores from a stmt (single store or Seq of
// stores). Nested Seq allowed; anything that is neither store nor no-op aborts
// collection.
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

// True when \p call is one of the boolean mask-combine ops that
// VectorizeParallelToPTO lowers to the VMI op \p vmi_name ("vand" / "vor" /
// "vnot"): the shared name table carries every admitted spelling
// (tirx./tir. bitwise_* and tir.And|Or|Not).
bool IsMaskLogicCall(const CallNode *call, const char *vmi_name) {
  if (const auto *op = call->op.as<OpNode>()) {
    const char *mapped = pto::MaskLogicVmiOpName(op->name);
    return mapped != nullptr && std::string(mapped) == vmi_name;
  }
  return false;
}

// Expand And/Or/Not in the Select condition into nested Selects:
//   select(a&b, t, f) => select(a, select(b, t, f), f)
//   select(a|b, t, f) => select(a, t, select(b, t, f))
//   select(~a, t, f)  => select(a, f, t)
// (Boolean *values* under T.Parallel stay mask logic and are converted by
// VectorizeParallelToPTO to vand/vor/vnot.)
PrimExpr ExpandPredSelect(const PrimExpr &cond, const PrimExpr &tval,
                          const PrimExpr &fval) {
  if (const auto *andn = cond.as<AndNode>()) {
    return ExpandPredSelect(andn->a, ExpandPredSelect(andn->b, tval, fval),
                            fval);
  }
  if (const auto *orn = cond.as<OrNode>()) {
    return ExpandPredSelect(orn->a, tval, ExpandPredSelect(orn->b, tval, fval));
  }
  if (const auto *notn = cond.as<NotNode>()) {
    return ExpandPredSelect(notn->a, fval, tval);
  }
  if (const auto *call = cond.as<CallNode>()) {
    if (IsMaskLogicCall(call, "vand") && call->args.size() == 2) {
      return ExpandPredSelect(
          call->args[0], ExpandPredSelect(call->args[1], tval, fval), fval);
    }
    if (IsMaskLogicCall(call, "vor") && call->args.size() == 2) {
      return ExpandPredSelect(call->args[0], tval,
                              ExpandPredSelect(call->args[1], tval, fval));
    }
    if (IsMaskLogicCall(call, "vnot") && call->args.size() == 1) {
      return ExpandPredSelect(call->args[0], fval, tval);
    }
  }
  return Select(cond, tval, fval);
}

class ControlFlowMutator : public StmtExprMutator {
public:
  Stmt VisitStmt_(const SBlockNode *op) final {
    if (op->name_hint != pto::kSimdVfBlockName) {
      // Enclosing blocks (root/tilelang_root/VECTOR/...) are traversed so the
      // SIMD_VF region is reached, but `in_simd_vf_` stays false: nothing
      // outside a converting region is rewritten.
      return StmtExprMutator::VisitStmt_(op);
    }
    // Only the converting regions are this pass's subject: a SIMD_VF block
    // that holds a T.Parallel unit. A SIMD_VF block without T.Parallel
    // (hand-written VMI) is returned unchanged, so its Min/Max keep their
    // code shape.
    if (!pto::RegionContainsParallel(op->body)) {
      return GetRef<Stmt>(op);
    }
    bool saved = in_simd_vf_;
    in_simd_vf_ = true;
    Stmt body = StmtExprMutator::VisitStmt(op->body);
    in_simd_vf_ = saved;
    if (body.same_as(op->body)) {
      return GetRef<Stmt>(op);
    }
    SBlock block = GetRef<SBlock>(op);
    block.CopyOnWrite()->body = body;
    return block;
  }

  Stmt VisitStmt_(const ForNode *op) final {
    if (!in_simd_vf_ || op->kind != ForKind::kParallel) {
      return StmtExprMutator::VisitStmt_(op);
    }

    Var saved_var = parallel_var_;
    bool saved_in = in_parallel_;
    parallel_var_ = op->loop_var;
    in_parallel_ = true;

    Stmt body = StmtExprMutator::VisitStmt(op->body);

    For loop = GetRef<For>(op);
    if (!body.same_as(op->body)) {
      loop.CopyOnWrite()->body = body;
    }

    parallel_var_ = saved_var;
    in_parallel_ = saved_in;
    return loop;
  }

  Stmt VisitStmt_(const IfThenElseNode *op) final {
    if (!in_parallel_) {
      return StmtExprMutator::VisitStmt_(op);
    }
    Stmt then_case = VisitStmt(op->then_case);
    Optional<Stmt> else_case;
    if (op->else_case.defined()) {
      else_case = VisitStmt(op->else_case.value());
    }
    return RewriteParallelIf(op->condition, then_case, else_case);
  }

  // Min/Max -> Select so Vectorize only needs cmp + vsel. Rewritten only
  // inside a T.Parallel unit in a converting region: scalar expressions in
  // sibling serial code within the same SIMD_VF keep their Min/Max shape.
  PrimExpr VisitExpr_(const MinNode *op) final {
    if (!in_parallel_) {
      return StmtExprMutator::VisitExpr_(op);
    }
    PrimExpr a = VisitExpr(op->a);
    PrimExpr b = VisitExpr(op->b);
    return Select(a < b, a, b);
  }
  PrimExpr VisitExpr_(const MaxNode *op) final {
    if (!in_parallel_) {
      return StmtExprMutator::VisitExpr_(op);
    }
    PrimExpr a = VisitExpr(op->a);
    PrimExpr b = VisitExpr(op->b);
    return Select(a > b, a, b);
  }

  // The frontend infinity() sentinel is a compile-time +inf constant of its
  // dtype; rewrite it to FloatImm so Vectorize's constant handling covers it
  // without a frontend-op-name case.
  PrimExpr VisitExpr_(const CallNode *op) final {
    if (in_parallel_) {
      if (const auto *op_node = op->op.as<OpNode>()) {
        const std::string &name = op_node->name;
        if (name == "tl.infinity" || name == "tir.infinity") {
          return FloatImm(op->dtype,
                          std::numeric_limits<double>::infinity());
        }
      }
    }
    return StmtExprMutator::VisitExpr_(op);
  }

private:
  bool IndexDependsOnParallelVar(const Array<PrimExpr> &indices) const {
    if (!parallel_var_.defined()) {
      return false;
    }
    Var pv = parallel_var_;
    for (const PrimExpr &idx : indices) {
      if (tirx::UsesVar(idx, [pv](const VarNode *v) {
            return pv.same_as(GetRef<Var>(v));
          })) {
        return true;
      }
    }
    return false;
  }

  Stmt RewriteParallelIf(const PrimExpr &condition, const Stmt &then_case,
                         const Optional<Stmt> &else_case) {
    std::vector<const BufferStoreNode *> then_stores;
    std::vector<const BufferStoreNode *> else_stores;
    const bool then_ok = CollectStores(then_case, &then_stores);
    const bool else_defined = else_case.defined();
    const bool else_ok =
        !else_defined || CollectStores(else_case.value(), &else_stores);

    if (!then_ok) {
      Fail("If then-branch under T.Parallel must be BufferStore(s) or no-op "
           "(Select-rewritable); unsupported stmt shape (Verify admits the "
           "same shapes)");
    }
    if (else_defined && !else_ok) {
      Fail("If else-branch under T.Parallel must be BufferStore(s) or no-op "
           "(Select-rewritable); unsupported stmt shape (Verify admits the "
           "same shapes)");
    }

    const bool then_noop = then_stores.empty();
    const bool else_noop = !else_defined || else_stores.empty();
    if (then_noop && else_noop) {
      return Evaluate(0);
    }

    PrimExpr cond = VisitExpr(condition);
    Array<Stmt> out_stores;

    auto check_lane_varying = [&](const Buffer &buf,
                                  const Array<PrimExpr> &indices) {
      if (IndexDependsOnParallelVar(indices)) {
        return;
      }
      std::ostringstream oss;
      oss << "lane-uniform store into buffer `" << buf->name
          << "` under T.Parallel is not supported at this stage; make "
             "the store lane-varying or hoist the computation out of "
             "the SimdVF region";
      Fail(oss.str());
    };

    // Rewrite one store as a predicated store. A nested one-sided branch
    // already carries its own predicate; the enclosing condition has to be
    // conjoined with it, never dropped.
    auto emit_predicated_store = [&](const BufferStoreNode *store,
                                     const PrimExpr &arm_pred) {
      check_lane_varying(store->buffer, store->indices);
      PrimExpr value = VisitExpr(store->value);
      PrimExpr pred = arm_pred;
      if (store->predicate.defined()) {
        pred = And(pred, VisitExpr(store->predicate.value()));
      }
      out_stores.push_back(
          BufferStore(store->buffer, value, store->indices, pred));
    };

    // Rewrite a pair of stores that write one address into a value choice.
    // Valid only when neither side is itself predicated: a predicated store
    // only writes the lanes its predicate selects, which a Select over two
    // values cannot express.
    auto emit_value_choice = [&](const BufferStoreNode *th,
                                 const BufferStoreNode *el) {
      check_lane_varying(th->buffer, th->indices);
      PrimExpr then_v = VisitExpr(th->value);
      PrimExpr else_v = VisitExpr(el->value);
      out_stores.push_back(BufferStore(
          th->buffer, ExpandPredSelect(cond, then_v, else_v), th->indices));
    };

    // An arm whose stores depend on each other only keeps its meaning if the
    // arm's write order is preserved: a later store may read what an earlier
    // one wrote (RAW), or overwrite what an earlier one read (WAR). Both are
    // accepted here: the arms are emitted in their source order with the
    // branch predicate, and the backend orders the resulting UB accesses
    // (PTOAS analyses intra-vecscope dependencies and inserts mem_bar). This
    // pass only has to produce the correct access sequence, not to insert
    // synchronisation itself.
    auto has_intra_arm_dependency =
        [](const std::vector<const BufferStoreNode *> &stores) {
          std::set<const BufferNode *> written;
          std::set<const BufferNode *> read;
          class Reads : public StmtExprVisitor {
          public:
            explicit Reads(std::set<const BufferNode *> *out) : out_(out) {}
            void VisitExpr_(const BufferLoadNode *op) final {
              out_->insert(op->buffer.get());
              StmtExprVisitor::VisitExpr_(op);
            }

          private:
            std::set<const BufferNode *> *out_;
          };
          for (const BufferStoreNode *store : stores) {
            std::set<const BufferNode *> store_reads;
            Reads probe(&store_reads);
            probe(store->value);
            for (const PrimExpr &idx : store->indices) {
              probe(idx);
            }
            // RAW: this store reads what an earlier store of the arm wrote.
            for (const BufferNode *buf : store_reads) {
              if (written.count(buf) != 0) {
                return true;
              }
            }
            // WAR: this store overwrites what an earlier store of the arm read.
            if (read.count(store->buffer.get()) != 0) {
              return true;
            }
            written.insert(store->buffer.get());
            read.insert(store_reads.begin(), store_reads.end());
          }
          return false;
        };

    const bool arms_predicated = [&] {
      for (const BufferStoreNode *s : then_stores) {
        if (s->predicate.defined()) return true;
      }
      for (const BufferStoreNode *s : else_stores) {
        if (s->predicate.defined()) return true;
      }
      return false;
    }();

    const bool arms_ordered = has_intra_arm_dependency(then_stores) ||
                              has_intra_arm_dependency(else_stores);


    if (then_noop) {
      for (const BufferStoreNode *el : else_stores) {
        emit_predicated_store(el, Not(cond));
      }
    } else if (else_noop) {
      for (const BufferStoreNode *th : then_stores) {
        emit_predicated_store(th, cond);
      }
    } else if (arms_predicated || arms_ordered) {
      // Two different reasons not to merge the arms into value choices:
      //
      //  * a nested one-sided branch already left a predicate on a store, and
      //    a Select over two values cannot express "write only these lanes";
      //  * a store reads what an earlier store of the same arm wrote, so the
      //    arm's write order carries meaning.
      //
      // Both are handled by emitting each arm in its source order, gated by
      // the branch condition: the complementary predicates keep the arms
      // exclusive, and the source order of an arm is preserved. The UB
      // read-after-write this can leave inside one vecscope is ordered by the
      // backend's vecscope memory barriers (pto.mem_bar "VST_VLD"), which run
      // after vecscope inference; the same guarantee the pipeline now relies
      // on for every other round-trip buffer.
      for (const BufferStoreNode *th : then_stores) {
        emit_predicated_store(th, cond);
      }
      for (const BufferStoreNode *el : else_stores) {
        emit_predicated_store(el, Not(cond));
      }
    } else {
      // Neither arm is predicated and no arm depends on its own write order,
      // so stores that write one address can be merged into a value choice.
      // An unmatched store only exists on one side of that address and is
      // gated by that side's condition.
      std::vector<bool> else_used(else_stores.size(), false);
      for (const BufferStoreNode *th : then_stores) {
        const BufferStoreNode *matched = nullptr;
        for (size_t j = 0; j < else_stores.size(); ++j) {
          if (!else_used[j] && SameAccess(th, else_stores[j])) {
            matched = else_stores[j];
            else_used[j] = true;
            break;
          }
        }
        if (matched != nullptr) {
          emit_value_choice(th, matched);
        } else {
          emit_predicated_store(th, cond);
        }
      }
      for (size_t j = 0; j < else_stores.size(); ++j) {
        if (!else_used[j]) {
          emit_predicated_store(else_stores[j], Not(cond));
        }
      }
    }

    if (out_stores.empty()) {
      return Evaluate(0);
    }
    if (out_stores.size() == 1) {
      return out_stores[0];
    }
    return SeqStmt(out_stores);
  }

  bool in_simd_vf_ = false;
  bool in_parallel_ = false;
  Var parallel_var_;
};

PrimFunc LowerControlFlow(PrimFunc f) {
  auto *n = f.CopyOnWrite();
  n->body = ControlFlowMutator()(std::move(n->body));
  return f;
}

} // namespace

namespace transform {

tvm::transform::Pass LegalizeParallelToPTO() {
  auto pass_func = [](PrimFunc f, const IRModule &,
                      const tvm::transform::PassContext &) {
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    if (!target.defined() || !TargetIsPTO(target.value())) {
      return f;
    }
    return LowerControlFlow(std::move(f));
  };
  return tvm::tirx::transform::CreatePrimFuncPass(
      pass_func, 0, "tl.LegalizeParallelToPTO", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LegalizeParallelToPTO",
                        LegalizeParallelToPTO);
}

} // namespace transform

} // namespace tl
} // namespace tvm
