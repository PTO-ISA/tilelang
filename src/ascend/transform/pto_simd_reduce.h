// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.
#ifndef TVM_TL_ASCEND_TRANSFORM_PTO_SIMD_REDUCE_H_
#define TVM_TL_ASCEND_TRANSFORM_PTO_SIMD_REDUCE_H_

#include "../../op/reduce.h"
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

namespace tvm::tl::pto {

// T.reduce is a collective between Parallel units, not an elementwise
// expression inside one. Retain its original regions, including view strides,
// version indices and partial destinations, until VMI addresses are legalized.
inline ReduceOp ParseSimdReduce(const tirx::CallNode *call) {
  ReduceOp op(call->args, call->annotations);
  ICHECK(op->src.scope() == "shared" || op->src.scope() == "shared.dyn")
      << "PTO SIMD reduce requires a UB source";
  ICHECK(op->dst.scope() == "shared" || op->dst.scope() == "shared.dyn")
      << "PTO SIMD reduce requires a UB destination";
  ICHECK(op->type->IsSum() || op->type->IsAbsSum() || op->type->IsMax() ||
         op->type->IsAbsMax() || op->type->IsMin())
      << "PTO SIMD reduce supports sum, abssum, max, absmax and min";
  ICHECK(!op->nan_propagate)
      << "PTO SIMD reduce does not support nan_propagate";
  ICHECK(op->dst->dtype == DataType::Float(32) ||
         op->dst->dtype == DataType::Int(32))
      << "PTO SIMD reduce requires float32 or int32 accumulation";
  ICHECK(op->src->dtype == op->dst->dtype ||
         (op->dst->dtype == DataType::Float(32) &&
          (op->src->dtype.is_float16() || op->src->dtype.is_bfloat16())))
      << "PTO SIMD reduce supports matching types or FP16/BF16 to FP32";
  ICHECK_GE(op->dim, 0);
  ICHECK_LT(op->dim, op->srcRegion_->region.size());
  for (const Range &r : op->srcRegion_->region) {
    ICHECK(r->extent.as<IntImmNode>() && r->extent.as<IntImmNode>()->value > 0)
        << "PTO SIMD reduce requires positive static region extents";
  }
  arith::Analyzer analyzer;
  PrimExpr src_count = 1, dst_count = 1;
  for (size_t i = 0; i < op->srcRegion_->region.size(); ++i) {
    if (i != static_cast<size_t>(op->dim))
      src_count *= op->srcRegion_->region[i]->extent;
  }
  for (const Range &r : op->dstRegion_->region)
    dst_count *= r->extent;
  ICHECK(analyzer.CanProveEqual(src_count, dst_count))
      << "PTO SIMD reduce destination must contain one element per reduced row";
  return op;
}

class SimdReduceLowerer : public tirx::StmtExprMutator {
public:
  explicit SimdReduceLowerer(int64_t lanes) : lanes_(lanes) {}

private:
  Stmt VisitStmt_(const tirx::EvaluateNode *stmt) final {
    const auto *call = stmt->value.as<tirx::CallNode>();
    if (!call || !call->op.same_as(ReduceOp::Get()))
      return GetRef<Stmt>(stmt);
    ReduceOp op = ParseSimdReduce(call);
    const auto &src = op->srcRegion_->region;
    const auto &dst = op->dstRegion_->region;
    const int64_t extent = src[op->dim]->extent.as<IntImmNode>()->value;
    const DataType dt = op->dst->dtype;
    const DataType vt = dt.with_lanes(lanes_);
    const DataType mt = DataType::Bool(lanes_);
    const bool sum = op->type->IsSum() || op->type->IsAbsSum();
    const bool maximum = op->type->IsMax() || op->type->IsAbsMax();
    const bool absolute = op->type->IsAbsMax() || op->type->IsAbsSum();
    const std::string fold = sum ? "vadd" : (maximum ? "vmax" : "vmin");
    const std::string cross = sum ? "vcadd" : (maximum ? "vcmax" : "vcmin");
    Map<String, ObjectRef> sizes{{"size", IntImm(DataType::Int(32), lanes_)}};
    auto vop = [](const std::string &name) {
      return Op::Get("tl.vmi." + name);
    };
    Array<Stmt> body;
    auto bind = [&](const std::string &name, PrimExpr value) {
      Var var(name, value.dtype());
      body.push_back(tirx::Bind(var, value));
      return var;
    };
    auto mask = [&](int64_t n) {
      return bind("reduce_mask", Call(mt, vop("create_mask"),
                                      {IntImm(DataType::Int(32), n)}, sizes));
    };
    Array<PrimExpr> src_index;
    Array<Var> row_vars;
    Array<PrimExpr> row_extents;
    PrimExpr linear_row = 0;
    for (size_t i = 0; i < src.size(); ++i) {
      if (i == static_cast<size_t>(op->dim)) {
        src_index.push_back(src[i]->min);
      } else {
        Var row("reduce_row", DataType::Int(32));
        row_vars.push_back(row);
        row_extents.push_back(src[i]->extent);
        src_index.push_back(src[i]->min + row);
        linear_row = linear_row * src[i]->extent + row;
      }
    }
    Array<PrimExpr> dst_index;
    PrimExpr quotient = linear_row;
    for (int i = static_cast<int>(dst.size()) - 1; i >= 0; --i) {
      dst_index.insert(dst_index.begin(),
                       dst[i]->min + floormod(quotient, dst[i]->extent));
      quotient = floordiv(quotient, dst[i]->extent);
    }
    auto ptr = [](const Buffer &buffer, const Array<PrimExpr> &indices,
                  int64_t size, int rw) {
      return Call(DataType::Handle(), tl::access_ptr(),
                  {BufferLoad(buffer, indices), IntImm(DataType::Int(32), size),
                   IntImm(DataType::Int(32), rw)});
    };
    arith::Analyzer analyzer;
    Array<PrimExpr> next_index = src_index;
    next_index.Set(op->dim, next_index[op->dim] + 1);
    PrimExpr stride = analyzer.Simplify(op->src->ElemOffset(next_index)[0] -
                                        op->src->ElemOffset(src_index)[0]);
    Var first_mask = mask(std::min(extent, lanes_));
    PrimExpr acc;
    for (int64_t begin = 0; begin < extent; begin += lanes_) {
      Array<PrimExpr> index = src_index;
      index.Set(op->dim, index[op->dim] + IntImm(DataType::Int(32), begin));
      Var active =
          begin == 0 ? first_mask : mask(std::min(lanes_, extent - begin));
      const DataType source_vt = op->src->dtype.with_lanes(lanes_);
      PrimExpr loaded;
      if (analyzer.CanProveEqual(stride, 1)) {
        loaded = bind("reduce_load",
                      Call(source_vt, vop("vload"),
                           {ptr(op->src, index, lanes_, 1), 0}, sizes));
      } else {
        Var indices = bind("reduce_indices", Call(DataType::Int(32, lanes_),
                                                  vop("vci"), {0}, sizes));
        Var bytes = bind("reduce_stride", Call(DataType::Int(32, lanes_),
                                               vop("vbrc"), {stride}, sizes));
        Var offsets =
            bind("reduce_offsets", Call(DataType::Int(32, lanes_), vop("vmul"),
                                        {indices, bytes, active}));
        PrimExpr typed_offsets = offsets;
        if (op->src->dtype.bits() == 16) {
          typed_offsets =
              bind("reduce_indices16",
                   Call(DataType::UInt(16, lanes_), vop("vcvt"), {offsets},
                        {{"to_dtype", StringImm("uint16")}}));
        } else {
          typed_offsets =
              bind("reduce_indices32",
                   Call(DataType::UInt(32, lanes_), vop("vinterpret_cast"),
                        {offsets}, {{"to_dtype", StringImm("uint32")}}));
        }
        loaded =
            bind("reduce_gather",
                 Call(source_vt, vop("vgather"),
                      {ptr(op->src, index, 1, 1), 0, typed_offsets, active}));
      }
      if (op->src->dtype != dt) {
        loaded =
            bind("reduce_cast", Call(vt, vop("vcvt"), {loaded},
                                     {{"to_dtype", StringImm("float32")}}));
      }
      if (absolute)
        loaded = bind("reduce_abs", Call(vt, vop("vabs"), {loaded, active}));
      if (begin == 0) {
        acc = loaded;
      } else {
        Var folded =
            bind("reduce_fold", Call(vt, vop(fold), {acc, loaded, active}));
        acc =
            bind("reduce_merge", Call(vt, vop("vsel"), {active, folded, acc}));
      }
    }
    Map<String, ObjectRef> cross_attrs;
    if (sum)
      cross_attrs.Set("reassoc", IntImm(DataType::Bool(), 1));
    Var reduced = bind("reduce_result",
                       Call(dt, vop(cross), {acc, first_mask}, cross_attrs));
    PrimExpr broadcast = reduced;
    Map<String, ObjectRef> one_size{{"size", IntImm(DataType::Int(32), 1)}};
    Var one = bind("reduce_one",
                   Call(DataType::Bool(), vop("create_mask"), {1}, one_size));
    if (!op->clear) {
      Var old = bind(
          "reduce_old",
          Call(dt, vop("vload"), {ptr(op->dst, dst_index, 1, 1), 0}, one_size));
      broadcast =
          bind("reduce_update", Call(dt, vop(fold), {old, broadcast, one}));
    }
    body.push_back(
        Evaluate(Call(DataType::Void(), vop("vstore"),
                      {broadcast, ptr(op->dst, dst_index, 1, 2), 0, one})));
    Stmt result = SeqStmt::Flatten(body);
    for (int i = static_cast<int>(row_vars.size()) - 1; i >= 0; --i) {
      result = For(row_vars[i], 0, row_extents[i], ForKind::kSerial, result);
    }
    return result;
  }
  int64_t lanes_;
};

} // namespace tvm::tl::pto
#endif
