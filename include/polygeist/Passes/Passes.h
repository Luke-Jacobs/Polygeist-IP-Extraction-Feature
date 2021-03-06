#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Pass/Pass.h"
#include <memory>
namespace mlir {
namespace polygeist {
/* Add constructor for IPregion pass here */
std::unique_ptr<OperationPass<FuncOp>> createMem2RegPass();
std::unique_ptr<OperationPass<FuncOp>> createLoopRestructurePass();
std::unique_ptr<OperationPass<FuncOp>> replaceAffineCFGPass();
std::unique_ptr<Pass> createCanonicalizeForPass();
std::unique_ptr<Pass> createRaiseSCFToAffinePass();
std::unique_ptr<Pass> createCPUifyPass(std::string);
std::unique_ptr<Pass> createBarrierRemovalContinuation();
std::unique_ptr<OperationPass<FuncOp>> detectReductionPass();
std::unique_ptr<OperationPass<FuncOp>> createRemoveTrivialUsePass();
std::unique_ptr<Pass> createParallelLowerPass();
std::unique_ptr<Pass> createConvertPolygeistToLLVMPass(const LowerToLLVMOptions &options);
std::unique_ptr<Pass> createExtractIPPass(unsigned startLine, unsigned endLine);
} // namespace polygeist
} // namespace mlir

void fully2ComposeAffineMapAndOperands(
    mlir::AffineMap *map, llvm::SmallVectorImpl<mlir::Value> *operands);
bool isValidIndex(mlir::Value val);

namespace mlir {
// Forward declaration from Dialect.h
template <typename ConcreteDialect>
void registerDialect(DialectRegistry &registry);

namespace scf {
class SCFDialect;
} // end namespace scf

namespace memref {
class MemRefDialect;
} // end namespace memref

class AffineDialect;
class StandardOpsDialect;
namespace LLVM {
class LLVMDialect;
}

#define GEN_PASS_CLASSES
#include "polygeist/Passes/Passes.h.inc"

} // end namespace mlir
