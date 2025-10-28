#include "Analysis/ScopeIdAllocation.h"

namespace mlir {
namespace triton::proton {

#define DEBUG_TYPE "proton-scope-id-allocation"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

void ScopeIdAllocation::run() {
  llvm::StringMap<size_t> nameToIdMap;
  llvm::StringMap<size_t> activeScopes;
  ScopeId id = 0;

  funcOp->walk<WalkOrder::PreOrder>([&](RecordOp recordOp) {
    auto name = recordOp.getName();
    LDBG("Processing RecordOp: " << recordOp);
    if (recordOp.getIsStart()) {
      if (activeScopes.contains(name)) {
        mlir::emitError(recordOp.getLoc(), "The scope name '")
            << name << "' is already open";
      } else {
        if (!nameToIdMap.contains(name)) {
          nameToIdMap[name] = id;
          idToNameMap[id] = name;
          LDBG("Assigning new scope id " << id << " to name '" << name << "'");
          id++;
        }
        opToIdMap[recordOp] = nameToIdMap[name];
        activeScopes[name] = nameToIdMap[name];
      }
    } else {
      if (activeScopes.contains(name)) {
        opToIdMap[recordOp] = activeScopes[name];
        activeScopes.erase(name);
      } else {
        mlir::emitError(recordOp.getLoc(), "The scope name '")
            << name << "' was not opened or already closed";
      }
    }
  });

  if (activeScopes.size() > 0) {
    for (auto &[name, _] : activeScopes) {
      mlir::emitError(funcOp->getLoc(), "Scope name '")
          << name << "' was opened but never closed";
    }
  }
  // Note: scopeParentIds is intentionally left empty (no hierarchy tracking)
}

ModuleScopeIdAllocation::ModuleScopeIdAllocation(ModuleOp moduleOp)
    : CallGraph<ScopeIdAllocation>(moduleOp) {
  ScopeIdAllocation::ScopeId funcScopeId = 0;
  walk<WalkOrder::PreOrder, WalkOrder::PostOrder>(
      // Pre-order edge walk callback
      [](CallOpInterface callOp, FunctionOpInterface funcOp) {},
      // Post-order node walk callback
      [&](FunctionOpInterface funcOp) {
        if (funcMap.contains(funcOp)) {
          return;
        }
        auto iter = funcMap.try_emplace(funcOp, ScopeIdAllocation(funcOp));
        funcScopeIdMap[funcOp] = funcScopeId;
        funcScopeId += iter.first->second.getNumScopes();
      });
  // Precompute per-function scope id mappings
  for (auto [funcOp, offset] : funcScopeIdMap) {
    // Names
    auto names = funcMap.lookup(funcOp).getScopeIdNames();
    for (auto &p : names)
      p.first += offset;
    scopeIdNames[funcOp] = std::move(names);
    // Parents - intentionally left empty, no hierarchy tracking
    scopeIdParents[funcOp] = ScopeIdAllocation::ScopeIdParent{};
  }
}

ScopeIdAllocation::ScopeId
ModuleScopeIdAllocation::getOpScopeId(Operation *op) const {
  auto funcOp = op->getParentOfType<triton::FuncOp>();
  auto funcOffset = funcScopeIdMap.lookup(funcOp);
  return funcMap.lookup(funcOp).getOpScopeId(op) + funcOffset;
}

ScopeIdAllocation::ScopeIdName
ModuleScopeIdAllocation::getScopeIdNames(triton::FuncOp funcOp) const {
  return scopeIdNames.lookup(funcOp);
}

ScopeIdAllocation::ScopeIdName
ModuleScopeIdAllocation::getScopeIdNames() const {
  ScopeIdAllocation::ScopeIdName combined;
  for (auto &entry : scopeIdNames)
    combined.insert(combined.end(), entry.second.begin(), entry.second.end());
  return combined;
}

ScopeIdAllocation::ScopeIdParent
ModuleScopeIdAllocation::getScopeIdParents(triton::FuncOp funcOp) const {
  return scopeIdParents.lookup(funcOp);
}

ScopeIdAllocation::ScopeIdParent
ModuleScopeIdAllocation::getScopeIdParents() const {
  ScopeIdAllocation::ScopeIdParent combined;
  for (auto &entry : scopeIdParents)
    combined.insert(combined.end(), entry.second.begin(), entry.second.end());
  return combined;
}

} // namespace triton::proton
} // namespace mlir
