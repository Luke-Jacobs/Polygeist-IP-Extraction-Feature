//===- IPRegionPass.cpp - Insert ipregion operation around IP ops ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Insert ipregion around IP code (specified by ipdef and ipend pragmas)
//
//===----------------------------------------------------------------------===//

#include "polygeist/Passes/Passes.h"
#include "polygeist/Ops.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Analysis/Liveness.h"
#include <queue>
#include <vector>
#include <iostream>  // TODO REMOVE for debugging

using namespace mlir;

namespace {

struct ExtractIPPass : public ExtractIPPassBase<ExtractIPPass> {

  unsigned startLine, endLine;

  static void printValueVec(std::vector<Value> inVec) {
    for (auto inVal : inVec) {
      llvm::errs() << "VALUE: " << inVal << "\n";
      auto inValUses = inVal.getUses();
      // auto ownerBlk = inVal.getParentBlock();
      // llvm::errs() << "Owner Block: " << ownerBlk << "\n";
      // llvm::errs() << "Dumping owner block: \n";
      // ownerBlk->dump();
      for (auto &use : inValUses) {
        llvm::errs() << "\tUse: " << *use.getOwner() << "\n";
      }
    }
  }

  /* 
  * Helper function to find if Operation is in or contains an IP 
  */
  static bool containsIP(mlir::Block &blk) {
    for (Operation &op : blk.getOperations()) {
      for (Region &reg : op.getRegions()) {
        for (Block &blk : reg.getBlocks())
          if (containsIP(blk))
            return true;
      }
    }
    return false;
  }

  /*
  * Helper function to find line number associated with Operation
  */
  static long getLineNumberOfOp(mlir::Operation &op) {
    using namespace mlir;

    FileLineColLoc lineData = op.getLoc().dyn_cast<FileLineColLoc>();
    if (lineData != NULL) {
      return lineData.getLine();
    } else {
      return -1;
    }
  }

  /* 
  * Performs the annotation of MLIR operations so that an ipregion operation can be created around them
  */
  void annotateBlkWithIPAttr(mlir::Block &blk, mlir::OpBuilder &builder, bool printOperation = false) {
    for (Operation &op : blk) {
      long opLineNumber = getLineNumberOfOp(op);
      if (opLineNumber > 0) {
        if ((unsigned)opLineNumber > startLine && (unsigned)opLineNumber < endLine) {
          // Make sure to exclude return operations from the IP- this is because this leaves no return operation for the host function after splicing
          if (mlir::isa<ReturnOp>(op)) {
            op.setAttr("inIP", builder.getBoolAttr(false));
            continue;  
          }
          op.setAttr("inIP", builder.getBoolAttr(true));
          // auto ipregion = builder.create<::mlir::polygeist::IpRegionOp>(builder.getUnknownLoc()); // This is a long comment
        } else {
          op.setAttr("inIP", builder.getBoolAttr(false));  // Make sure that we also label the operations that are not within the IP, but have a line number attached
        }
      }
    }

    for (mlir::Operation &op : blk.getOperations()) {
      for (mlir::Region &reg : op.getRegions()) {
        for (mlir::Block &blk_inner : reg.getBlocks()) {
          annotateBlkWithIPAttr(blk_inner, builder);
        }
      }
    }
  }

  /* 
  * Generate an IP function with BFS recursion through MLIR. Once the first {inIP = true} is found, capture it and every other operation (regardless of its line number status) 
  * until we hit an {inIP = false}. 
  */
  static bool captureIPOperations(mlir::ModuleOp &module, mlir::FuncOp hostOp, mlir::Block *blk, mlir::OpBuilder &builder) {
    mlir::Attribute attr;
    std::queue<Block *> captureTargetQueue;
    captureTargetQueue.push(blk);

    while (!captureTargetQueue.empty()) {
      // llvm::errs() << "----- TAKING A BLOCK FROM QUEUE -----\n";

      /* Take a block off the queue (BFS order) */
      blk = captureTargetQueue.front();
      captureTargetQueue.pop();

      for (auto blkOpIter = blk->begin(); blkOpIter != blk->end(); ++blkOpIter) {
        mlir::Operation &op = *blkOpIter;

        // FIXME Due to the scrambled nature of how MLIR pushes constants to the top of the function, 
        // a good rule of thumb (in order to extract the IP region at the right starting position) is
        // to ignore starting the IP block with a constant.
        if (mlir::isa<arith::ConstantOp>(op))
          continue;  
        if (mlir::isa<mlir::LLVM::UndefOp>(op))  // The undef operation is grouped with constants at the tops of functions <- TODO fix the constant-grouping problem
          continue;
        if (mlir::isa<mlir::memref::AllocaOp>(op))
          continue;

        if ((attr = op.getAttr("inIP")) != NULL && attr == builder.getBoolAttr(true)) {
          /* Now that we have found the start of the IP, search for an {inIP = false} attribute or the end of the block */

          // Now the task is to find the end of the IP in the block
          mlir::Block::iterator startOfIP = blkOpIter;
          mlir::Block::iterator startOfRemainder = blk->end();  // this iterator location keeps track of the first operation of the remainder block
          Block *startOfIPBlk = blk->splitBlock(startOfIP);

          // Continue iterating: Add all operations below this that are in the same block that don't have {inIP = false} in them
          for (auto ipOpIter = startOfIPBlk->begin(); ipOpIter != startOfIPBlk->end(); ++ipOpIter) {
            mlir::Operation &opAfterIPStart = *ipOpIter;
            // llvm::errs() << "Looking at operation: " << opAfterIPStart << "\n";
            bool isAmbigousOp = (attr = opAfterIPStart.getAttr("inIP")) == NULL;
            bool isExcludedOp = attr == builder.getBoolAttr(false);
            bool isRetOp = isa<ReturnOp>(opAfterIPStart);
            if (isAmbigousOp)
              startOfRemainder = ipOpIter;  // We have found the end of the IP (the iter to the first operation in this block that is not in the IP)
            if (isExcludedOp || isRetOp) {  // Both excluded operations and the return operation triggers the end-of-IP logic
              // If the startOfRemainder iterator location has not been updated by encountering an ambiguous operation, update it here at this excluded operation
              if (startOfRemainder == blk->end())
                startOfRemainder = ipOpIter;
              break;
            }
            // If we have encountered an operation that has been annotated explicitly as being in the IP
            if (!isAmbigousOp && !isExcludedOp) {
              startOfRemainder = blk->end();  // Reset the remainder block start because we have not yet found it
            }
          }

          /* Temporary fix for affine.yield errors: reverse the iterator if the last instruction in the IP is an affine.yield. This means
             the original function is left without a yield, which will cause invalid MLIR. */

          Block *remainderBlk = NULL;
          if (startOfRemainder != blk->end()) { // If we have remaining operations, split the block at the start of those remaining operations
            remainderBlk = startOfIPBlk->splitBlock(startOfRemainder);
            // llvm::errs() << "Remainder block: ";
            // remainderBlk->dump();
          }
          Block *ipBlk = startOfIPBlk;

          // llvm::errs() << "<<<<< IP block (" << ipBlk << ") extracted >>>>>\n";
          // ipBlk->dump();

          /* Create a function from the ip block */
          Liveness lv(hostOp);
          auto liveIn = lv.getLiveIn(ipBlk);
          auto liveOut = lv.getLiveOut(ipBlk);
          std::vector<Value> inVec, outVec;

          /* Collect liveIn values as Value and filter any BlockArguments that have their origin in a block 
             inside of the IP function. This method needs testing, but it works for affine.for loops. */
          for (Value val : liveIn) { 
            // llvm::errs() << "Parent op of value: " << *val.getParentBlock()->getParentOp() << "\n";
            if (val.getParentBlock()->getParentOp()->getAttr("inIP") == builder.getBoolAttr(true)) {
              // llvm::errs() << "Skipping value: " << val << "\n";
              continue;  // Skip any BlockArgument that is the argument of a block within an operation in the IP region
            }
            inVec.push_back(val);
          }
          for (Value val : liveOut) { 
            outVec.push_back(val); 
          }

          auto funcType = builder.getFunctionType(ValueRange(inVec).getTypes(), ValueRange(outVec).getTypes());

          // llvm::errs() << "==== IP FUNCTION INPUTS =====\n";
          // printValueVec(inVec);
          // llvm::errs() << "==== IP FUNCTION OUTPUTS =====\n";
          // printValueVec(outVec);

          /* Create the new function with arguments and results specified by funcType */
          auto ipFunc = builder.create<FuncOp>(op.getLoc(), "IP_func", funcType);
          module.push_back(ipFunc);  // Add FuncOp to the module
          
          /* Add the function call */
          builder.setInsertionPointToEnd(blk);  // The location of the function call should be right before the first split
          auto ipCallOp = builder.create<mlir::CallOp>(op.getLoc(), ipFunc, ValueRange(inVec));

          /* Copy the existing IP block operations into the function */
          Block &createIPBlk = ipFunc.body().emplaceBlock();
          auto &createIPBlkOps = createIPBlk.getOperations();
          auto &ipBlkOps = ipBlk->getOperations();
          createIPBlkOps.splice(createIPBlkOps.begin(), ipBlkOps, ipBlkOps.begin(), ipBlkOps.end());
          builder.setInsertionPointToEnd(&createIPBlk);
          builder.create<ReturnOp>(op.getLoc(), ValueRange(outVec));  // Create a return op with values from the liveness output

          // Add arguments to the function region now that we have already added arguments to FuncOp
          ipFunc.body().addArguments(ValueRange(inVec).getTypes());

          /* Replace all uses (within the function) of old values from main with the newly-created argument values */
          for (auto zip : llvm::zip(inVec, ipFunc.getArguments())) {
            auto liveInVal = std::get<0>(zip);
            auto newArgVal = std::get<1>(zip);
            // llvm::errs() << "Comparing (" << liveInVal << ") with (" << newArgVal << ")\n";
            liveInVal.replaceUsesWithIf(newArgVal, [&](OpOperand &operand) { return operand.getOwner()->getParentOfType<FuncOp>() == ipFunc; });
          }

          /* Replace all uses (within the remainder block) of old values with the ip function return values */
          for (auto zip : llvm::zip(outVec, ipCallOp.getResults())) {
            auto liveOutVal = std::get<0>(zip);  // This item represents an old value that is being used by operations in remainderBlk
            auto newArgVal = std::get<1>(zip);  // This is a return value of IP_func(...)
            liveOutVal.replaceUsesWithIf(newArgVal, [&](OpOperand &operand) { return operand.getOwner()->getParentOfType<FuncOp>() == hostOp; });
          }

          ipBlk->erase();

          // std::cout << "<<<<< IP function extracted >>>>>\n";
          // ipFunc.dump();

          /* Print usages of arguments. If the arguments are used by the function (as they should be), all values should be non-zero */
          // std::cout << "===== DEBUG ARGUMENT USAGE COUNT =====\n";
          // for (size_t argI = 0; argI < ipFunc.body().getArguments().size(); argI++) {
              // std::cout << "Usage count for argument " << argI << " is: ";
          //   auto argumentUsageIter = ipFunc.body().getArgument(argI).getUses();
          //   // std::cout << std::distance(argumentUsageIter.begin(), argumentUsageIter.end()) << "\n";
          // }
          
          // std::cout << "<<<<< Third block extracted >>>>>\n";
          // remainderBlk->dump();

          /* Put the two halves together */
          if (remainderBlk) {
            auto &opsInBlk = blk->getOperations();
            auto &opsInRemainderBlk = remainderBlk->getOperations();
            opsInBlk.splice(opsInBlk.end(), opsInRemainderBlk, opsInRemainderBlk.begin(), opsInRemainderBlk.end());
            // builder.mergeBlocks(blk, remainderBlk, blk->getArguments());
            remainderBlk->erase();
          }

          llvm::errs() << "COMPLETE MLIR OUTPUT AFTER IP FUNCTION EXTRACT PASS\n";
          module->dump();

          return true;
        }
      
        /* If we haven't found any IP-annotated operations in this operation, put its child blocks on the queue to traverse into */
        for (mlir::Region &reg : op.getRegions()) {
          for (mlir::Block &childBlk : reg.getBlocks()) {
            captureTargetQueue.push(&childBlk);
          }
        }
      }
    }

    // We weren't able to capture any IP operations
    return false;
  }

  ExtractIPPass(unsigned startLine, unsigned endLine) : startLine(startLine), endLine(endLine) {

  }

  void runOnOperation() override;

};

}; // end anonymous namespace

std::unique_ptr<Pass> polygeist::createExtractIPPass(unsigned startLine, unsigned endLine) {
  return std::make_unique<ExtractIPPass>(startLine, endLine);
}

void ExtractIPPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  mlir::OpBuilder builder(moduleOp.getContext());

  for (auto &unconvertedOp : moduleOp.getOps()) {
    FuncOp funcOp = mlir::dyn_cast<FuncOp>(&unconvertedOp);
    if (funcOp == NULL)
      continue;

    /* Skip IP functions, since they are already captured */
    if (funcOp.getName() == "IP_func")
      continue;

    for (Block &blk : funcOp.body()) {
      annotateBlkWithIPAttr(blk, builder);
    }

    llvm::errs() << "===== POST-ANNOTATION MLIR DUMP =====\n";
    funcOp->dump();

    /* Traverse into the MLIR and find the first instance of the {inIP = true} attribute. */
    mlir::IRRewriter rewriter(funcOp->getContext());
    for (Block &blk : funcOp.body()) {
      /* Search for IP operations until we find one, then extract the IP that starts at that  */
      if (captureIPOperations(moduleOp, funcOp, &blk, builder)) {
        break;
      }
    }
  }
}
