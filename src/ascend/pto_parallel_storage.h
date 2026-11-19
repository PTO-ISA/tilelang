/*!
 * \file pto_parallel_storage.h
 * \brief Source ownership and constant slots for ordinary Parallel locals.
 */
#ifndef TVM_TL_ASCEND_PTO_PARALLEL_STORAGE_H_
#define TVM_TL_ASCEND_PTO_PARALLEL_STORAGE_H_
#include "ascend/transform/vmi_alloc.h"
#include <tvm/arith/analyzer.h>
#include <tvm/tirx/stmt.h>
#include <unordered_map>

namespace tvm::tl::pto {
enum class StorageClass {
  kPrivate,
  kExternal,
  kUB,
  kExplicitVMI,
  kUnsupported
};
struct SlotAccess {
  int64_t capacity;
  int64_t slot;
};
class ParallelStorageQuery {
public:
  void Build(const tirx::Stmt &body);
  StorageClass Classify(const tirx::Buffer &buffer,
                        const tirx::For &unit = {}) const;
  SlotAccess ResolveSlot(const tirx::Buffer &buffer,
                         const ffi::Array<PrimExpr> &indices,
                         arith::Analyzer *analyzer,
                         const ffi::Map<tirx::Var, PrimExpr> &binds = {}) const;
  void ValidateOwnerUses(const tirx::Stmt &body) const;
  void CheckStorageV1(const tirx::Buffer &buffer) const;
  ffi::Array<tirx::Buffer> PrivateAllocations(const tirx::For &unit) const;
  bool IsPrivate(const tirx::Var &data) const { return owners_.count(data); }
  const tirx::Buffer &Allocation(const tirx::Var &data) const;

private:
  using Buffers = std::unordered_map<tirx::Var, tirx::Buffer,
                                     ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;
  using Owners = std::unordered_map<tirx::Var, tirx::For, ffi::ObjectPtrHash,
                                    ffi::ObjectPtrEqual>;
  Buffers allocations_;
  Owners owners_;
  vmi::AllocMarkers markers_;
};
} // namespace tvm::tl::pto
#endif
