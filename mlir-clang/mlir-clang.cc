//===- mlir-clang.cpp - mlir-clang Driver ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for mlir-clang when built as standalone binary.
//
//===----------------------------------------------------------------------===//

#include <clang/Basic/DiagnosticIDs.h>
#include <clang/Driver/Driver.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/CompilerInvocation.h>
#include <clang/Frontend/FrontendOptions.h>
#include <clang/Frontend/TextDiagnosticBuffer.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <clang/Frontend/Utils.h>

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"
#include "mlir/Conversion/SCFToOpenMP/SCFToOpenMP.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Program.h"
#include <fstream>

#include "polygeist/Ops.h"
#include "Lib/pragmaHandler.h"
#include <iostream>

#include "polygeist/Dialect.h"
#include "polygeist/Passes/Passes.h"

using namespace llvm;

static cl::OptionCategory toolOptions("clang to mlir - tool options");

static cl::opt<bool> CudaLower("cuda-lower", cl::init(false),
                               cl::desc("Add parallel loops around cuda"));

static cl::opt<bool> EmitLLVM("emit-llvm", cl::init(false),
                              cl::desc("Emit llvm"));

static cl::opt<bool> EmitAssembly("S", cl::init(false),
                                  cl::desc("Emit Assembly"));

static cl::opt<bool> Opt0("O0", cl::init(false), cl::desc("Opt level 0"));
static cl::opt<bool> Opt1("O1", cl::init(false), cl::desc("Opt level 1"));
static cl::opt<bool> Opt2("O2", cl::init(false), cl::desc("Opt level 2"));
static cl::opt<bool> Opt3("O3", cl::init(false), cl::desc("Opt level 3"));

static cl::opt<bool> SCFOpenMP("scf-openmp", cl::init(true),
                               cl::desc("Emit llvm"));

static cl::opt<bool> ShowAST("show-ast", cl::init(false), cl::desc("Show AST"));

static cl::opt<bool> ExtractIP("extract-ip", cl::init(false), cl::desc("Extract IPs from C bookmarked by ipdef and ipend pragmas"));

static cl::opt<bool> ImmediateMLIR("immediate", cl::init(false),
                                   cl::desc("Emit immediate mlir"));

static cl::opt<bool> RaiseToAffine("raise-scf-to-affine", cl::init(false),
                                   cl::desc("Raise SCF to Affine"));

static cl::opt<bool> ScalarReplacement("scal-rep", cl::init(true),
                                       cl::desc("Raise SCF to Affine"));

static cl::opt<bool>
    DetectReduction("detect-reduction", cl::init(false),
                    cl::desc("Detect reduction in inner most loop"));

static cl::opt<std::string> Standard("std", cl::init(""),
                                     cl::desc("C/C++ std"));

static cl::opt<std::string> CUDAGPUArch("cuda-gpu-arch", cl::init(""),
                                        cl::desc("CUDA GPU arch"));

static cl::opt<std::string> CUDAPath("cuda-path", cl::init(""),
                                     cl::desc("CUDA Path"));

static cl::opt<std::string> Output("o", cl::init("-"), cl::desc("Output file"));

static cl::list<std::string> inputFileName(cl::Positional, cl::OneOrMore,
                                           cl::desc("<Specify input file>"),
                                           cl::cat(toolOptions));

static cl::opt<std::string> cfunction("function",
                                      cl::desc("<Specify function>"),
                                      cl::init("main"), cl::cat(toolOptions));

static cl::opt<bool> FOpenMP("fopenmp", cl::init(false),
                             cl::desc("Enable OpenMP"));

static cl::opt<std::string> ToCPU("cpuify", cl::init(""),
                                  cl::desc("Convert to cpu"));

static cl::opt<std::string> MArch("march", cl::init(""),
                                  cl::desc("Architecture"));

static cl::opt<std::string> ResourceDir("resource-dir", cl::init(""),
                                        cl::desc("Resource-dir"));

static cl::opt<bool> EarlyVerifier("early-verifier", cl::init(false),
                                   cl::desc("Enable verifier ASAP"));

static cl::opt<bool> Verbose("v", cl::init(false), cl::desc("Verbose"));

static cl::opt<bool>
    showDialects("show-dialects",
                 llvm::cl::desc("Print the list of registered dialects"),
                 llvm::cl::init(false), cl::cat(toolOptions));

static cl::list<std::string> includeDirs("I", cl::desc("include search path"),
                                         cl::cat(toolOptions));

static cl::list<std::string> defines("D", cl::desc("defines"),
                                     cl::cat(toolOptions));

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

class MemRefInsider
    : public mlir::MemRefElementTypeInterface::FallbackModel<MemRefInsider> {};

template <typename T>
struct PtrElementModel
    : public mlir::LLVM::PointerElementTypeInterface::ExternalModel<
          PtrElementModel<T>, T> {};

extern int cc1_main(ArrayRef<const char *> Argv, const char *Argv0,
                    void *MainAddr);
extern int cc1as_main(ArrayRef<const char *> Argv, const char *Argv0,
                      void *MainAddr);
extern int cc1gen_reproducer_main(ArrayRef<const char *> Argv,
                                  const char *Argv0, void *MainAddr);
std::string GetExecutablePath(const char *Argv0, bool CanonicalPrefixes) {
  if (!CanonicalPrefixes) {
    SmallString<128> ExecutablePath(Argv0);
    // Do a PATH lookup if Argv0 isn't a valid path.
    if (!llvm::sys::fs::exists(ExecutablePath))
      if (llvm::ErrorOr<std::string> P =
              llvm::sys::findProgramByName(ExecutablePath))
        ExecutablePath = *P;
    return std::string(ExecutablePath.str());
  }

  // This just needs to be some symbol in the binary; C++ doesn't
  // allow taking the address of ::main however.
  void *P = (void *)(intptr_t)GetExecutablePath;
  return llvm::sys::fs::getMainExecutable(Argv0, P);
}
static int ExecuteCC1Tool(SmallVectorImpl<const char *> &ArgV) {
  // If we call the cc1 tool from the clangDriver library (through
  // Driver::CC1Main), we need to clean up the options usage count. The options
  // are currently global, and they might have been used previously by the
  // driver.
  llvm::cl::ResetAllOptionOccurrences();

  llvm::BumpPtrAllocator A;
  llvm::StringSaver Saver(A);
  llvm::cl::ExpandResponseFiles(Saver, &llvm::cl::TokenizeGNUCommandLine, ArgV,
                                /*MarkEOLs=*/false);
  StringRef Tool = ArgV[1];
  void *GetExecutablePathVP = (void *)(intptr_t)GetExecutablePath;
  if (Tool == "-cc1")
    return cc1_main(makeArrayRef(ArgV).slice(1), ArgV[0], GetExecutablePathVP);
  if (Tool == "-cc1as")
    return cc1as_main(makeArrayRef(ArgV).slice(2), ArgV[0],
                      GetExecutablePathVP);
  if (Tool == "-cc1gen-reproducer")
    return cc1gen_reproducer_main(makeArrayRef(ArgV).slice(2), ArgV[0],
                                  GetExecutablePathVP);
  // Reject unknown tools.
  llvm::errs() << "error: unknown integrated tool '" << Tool << "'. "
               << "Valid tools include '-cc1' and '-cc1as'.\n";
  return 1;
}

int emitBinary(char *Argv0, const char *filename,
               SmallVectorImpl<char *> &LinkArgs, bool LinkOMP) {

  using namespace clang;
  using namespace clang::driver;
  using namespace std;
  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());
  // Buffer diagnostics from argument parsing so that we can output them using a
  // well formed diagnostic object.
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  TextDiagnosticPrinter *DiagBuffer =
      new TextDiagnosticPrinter(llvm::errs(), &*DiagOpts);

  DiagnosticsEngine Diags(DiagID, &*DiagOpts, DiagBuffer);

  const char *binary = Argv0;
  const unique_ptr<Driver> driver(
      new Driver(binary, llvm::sys::getDefaultTargetTriple(), Diags));
  driver->CC1Main = &ExecuteCC1Tool;
  std::vector<const char *> Argv;
  Argv.push_back(Argv0);
  // Argv.push_back("-x");
  // Argv.push_back("ir");
  Argv.push_back(filename);
  if (LinkOMP)
    Argv.push_back("-fopenmp");
  if (ResourceDir != "") {
    Argv.push_back("-resource-dir");
    char *chars = (char *)malloc(ResourceDir.length() + 1);
    memcpy(chars, ResourceDir.data(), ResourceDir.length());
    chars[ResourceDir.length()] = 0;
    Argv.push_back(chars);
  }
  if (Verbose) {
    Argv.push_back("-v");
  }
  if (CUDAGPUArch != "") {
    auto a = "--cuda-gpu-arch=" + CUDAGPUArch;
    char *chars = (char *)malloc(a.length() + 1);
    memcpy(chars, a.data(), a.length());
    chars[a.length()] = 0;
    Argv.push_back(chars);
  }
  if (CUDAPath != "") {
    auto a = "--cuda-path=" + CUDAPath;
    char *chars = (char *)malloc(a.length() + 1);
    memcpy(chars, a.data(), a.length());
    chars[a.length()] = 0;
    Argv.push_back(chars);
  }
  if (Opt0) {
    Argv.push_back("-O0");
  }
  if (Opt1) {
    Argv.push_back("-O1");
  }
  if (Opt2) {
    Argv.push_back("-O2");
  }
  if (Opt3) {
    Argv.push_back("-O3");
  }
  if (Output != "") {
    Argv.push_back("-o");
    char *chars = (char *)malloc(Output.length() + 1);
    memcpy(chars, Output.data(), Output.length());
    chars[Output.length()] = 0;
    Argv.push_back(chars);
  }
  for (auto arg : LinkArgs)
    Argv.push_back(arg);

  const unique_ptr<Compilation> compilation(
      driver->BuildCompilation(llvm::ArrayRef<const char *>(Argv)));

  if (ResourceDir != "")
    driver->ResourceDir = ResourceDir;
  SmallVector<std::pair<int, const Command *>, 4> FailingCommands;
  int Res = 0;

  driver->ExecuteCompilation(*compilation, FailingCommands);
  for (const auto &P : FailingCommands) {
    int CommandRes = P.first;
    const Command *FailingCommand = P.second;
    if (!Res)
      Res = CommandRes;

    // If result status is < 0, then the driver command signalled an error.
    // If result status is 70, then the driver command reported a fatal error.
    // On Windows, abort will return an exit code of 3.  In these cases,
    // generate additional diagnostic information if possible.
    bool IsCrash = CommandRes < 0 || CommandRes == 70;
#ifdef _WIN32
    IsCrash |= CommandRes == 3;
#endif
    if (IsCrash) {
      driver->generateCompilationDiagnostics(*compilation, *FailingCommand);
      break;
    }
  }
  Diags.getClient()->finish();

  return Res;
}

/*
 * Helper function to find line number associated with Operation
 */
long getLineNumberOfOp(mlir::Operation &op) {
  using namespace mlir;

  FileLineColLoc lineData = op.getLoc().dyn_cast<FileLineColLoc>();
  if (lineData != NULL) {
    return lineData.getLine();
  } else {
    return -1;
  }
}

/*
 * Recurse through operation tree, printing each operation in pre-order.
 */
void printLineNumbers(mlir::Operation &op) {
  // Block &bodyBlk =
  // module->getRegion().getBlocks().front().front().getRegion(0).getBlocks().front();
  // Block &funcBodyBlk = bodyBlk.front().getRegion(0).getBlocks().front();
  // auto funcBlk = bodyBlk.front().getRegion(0).getBlocks().front();
  using namespace mlir;

  std::cout << "===== OPERATION =====\n";
  op.dump();

  for (Region &regionOp : op.getRegions()) {
    for (Block &blk : regionOp.getBlocks()) {
      for (Operation &op : blk.getOperations()) {
        printLineNumbers(op);
      }
    }
  }
}

/*
 * In order to capture the entirety of the C code in the IP definition, we apply this process:
 * 
 *     1. Travel down the current block's operation list, checking to see if each operation has
 *        a line number that is greater than startLine.
 *     2. If the program encounters a line number that is greater than endLine, backtrack into
 *        the previous operation's region. If there is an IP contained in the file, it must
 *        reside in that subregion, since the IP region must be bookended by operations with
 *        line numbers less than startLine and greater than endLine.
 *     3. Once an operation is found that is within the ip line range, add this operation and
 *        all other operations in this block until we hit an operation that is outside of the
 *        ip range.
 *
 */
void addIPRegion_helper() {

}

/* Retrieves any operations in the regions contained by an Operation */

/*
 * Add every MLIR operation between startLine and endLine to the new ipregion
 * MLIR operation
 * TODO AddIPRegion before any passes
 */
void addIPRegion(mlir::Operation &op, mlir::OpBuilder &builder,
                 const struct IPLocList &ipLocList, bool printOperation = false) {
  long opLineNumber = getLineNumberOfOp(op);
  if (opLineNumber > 0) {
    if (ipLocList.isInIP((unsigned)opLineNumber)) {
      op.setAttr("inIP", builder.getBoolAttr(true));
      // auto ipregion = builder.create<::mlir::polygeist::IpRegionOp>(builder.getUnknownLoc()); // This is a long comment
    }
  }

  if (printOperation) {
    std::cout << "===== ORIGINAL MLIR (before tags) =====\n";
    op.dump();
    std::cout << "\n";
  }

  for (mlir::Region &regionOp : op.getRegions()) {
    for (mlir::Block &blk : regionOp.getBlocks()) {
      for (mlir::Operation &op : blk.getOperations()) {
        addIPRegion(op, builder, ipLocList);
      }
    }
  }
}

#include "Lib/clang-mlir.cc"
int main(int argc, char **argv) {

  if (argc >= 1) {
    if (std::string(argv[1]) == "-cc1") {
      SmallVector<const char *> Argv;
      for (int i = 0; i < argc; i++)
        Argv.push_back(argv[i]);
      return ExecuteCC1Tool(Argv);
    }
  }
  SmallVector<char *> LinkageArgs;
  SmallVector<char *> MLIRArgs;
  {
    bool linkOnly = false;
    for (int i = 0; i < argc; i++) {
      StringRef ref(argv[i]);
      if (ref == "-Wl,--start-group")
        linkOnly = true;
      if (!linkOnly) {
        if (ref == "-fPIC" || ref == "-c" || ref.startswith("-fsanitize")) {
          LinkageArgs.push_back(argv[i]);
        } else if (ref == "-L" || ref == "-l") {
          LinkageArgs.push_back(argv[i]);
          i++;
          LinkageArgs.push_back(argv[i]);
        } else if (ref.startswith("-L") || ref.startswith("-l") ||
                   ref.startswith("-Wl")) {
          LinkageArgs.push_back(argv[i]);
        } else if (ref == "-D" || ref == "-I") {
          MLIRArgs.push_back(argv[i]);
          i++;
          MLIRArgs.push_back(argv[i]);
        } else if (ref.startswith("-D")) {
          MLIRArgs.push_back("-D");
          MLIRArgs.push_back(&argv[i][2]);
        } else if (ref.startswith("-I")) {
          MLIRArgs.push_back("-I");
          MLIRArgs.push_back(&argv[i][2]);
        } else {
          MLIRArgs.push_back(argv[i]);
        }
      } else {
        LinkageArgs.push_back(argv[i]);
      }
      if (ref == "-Wl,--end-group")
        linkOnly = false;
    }
  }
  using namespace mlir;

  int size = MLIRArgs.size();
  char **data = MLIRArgs.data();
  InitLLVM y(size, data);
  cl::ParseCommandLineOptions(size, data);
  assert(inputFileName.size());
  for (auto inp : inputFileName) {
    std::ifstream inputFile(inp);
    if (!inputFile.good()) {
      outs() << "Not able to open file: " << inp << "\n";
      return -1;
    }
  }

  // registerDialect<AffineDialect>();
  // registerDialect<StandardOpsDialect>();
  mlir::DialectRegistry registry;
  mlir::registerOpenMPDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  MLIRContext context(registry);

  context.disableMultithreading();
  context.getOrLoadDialect<AffineDialect>();
  context.getOrLoadDialect<StandardOpsDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
  context.getOrLoadDialect<mlir::NVVM::NVVMDialect>();
  context.getOrLoadDialect<mlir::gpu::GPUDialect>();
  context.getOrLoadDialect<mlir::omp::OpenMPDialect>();
  context.getOrLoadDialect<mlir::math::MathDialect>();
  context.getOrLoadDialect<mlir::memref::MemRefDialect>();
  context.getOrLoadDialect<mlir::linalg::LinalgDialect>();
  context.getOrLoadDialect<mlir::polygeist::PolygeistDialect>();
  // MLIRContext context;

  LLVM::LLVMPointerType::attachInterface<MemRefInsider>(context);
  LLVM::LLVMStructType::attachInterface<MemRefInsider>(context);
  MemRefType::attachInterface<PtrElementModel<MemRefType>>(context);

  if (showDialects) {
    outs() << "Registered Dialects:\n";
    for (Dialect *dialect : context.getLoadedDialects()) {
      outs() << dialect->getNamespace() << "\n";
    }
    return 0;
  }

  mlir::OwningOpRef<mlir::ModuleOp> module(mlir::ModuleOp::create(
      mlir::OpBuilder(&context).getUnknownLoc())); // Take note of this usage of
                                                   // OpBuilder from context

  llvm::Triple triple;
  llvm::DataLayout DL("");
  struct IPLocList
      ipRangeMetadata; // Will store the IP start and end line numbers
  parseMLIR(argv[0], inputFileName, cfunction, includeDirs, defines, module,
            triple, DL, &ipRangeMetadata); /* After we have parsed MLIR, apply
                                              the following passes... */

  mlir::OpBuilder opBuilder(&context);
  mlir::PassManager pm(&context);

  if (ImmediateMLIR) {
    llvm::errs() << "<immediate: mlir>\n";
    module->dump();
    llvm::errs() << "</immediate: mlir>\n";
  }

  bool LinkOMP = false;
  pm.enableVerifier(EarlyVerifier);

  /* optPM is the pass manager that receives the registration of the
     canonicalizer, a filter that removes unnecessary MLIR operations.
     I will want to traverse the generated MLIR tree after these passes
     happen. */
  mlir::OpPassManager &optPM = pm.nest<mlir::FuncOp>();
  if (true) {
    optPM.addPass(mlir::createCSEPass());
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(polygeist::createMem2RegPass());
    optPM.addPass(mlir::createCSEPass());
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(polygeist::createMem2RegPass());
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(polygeist::createRemoveTrivialUsePass());
    optPM.addPass(polygeist::createMem2RegPass());
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(polygeist::createLoopRestructurePass());
    if (!CudaLower)
      optPM.addPass(polygeist::replaceAffineCFGPass());
    optPM.addPass(mlir::createCanonicalizerPass());
    if (ScalarReplacement)
      optPM.addPass(mlir::createAffineScalarReplacementPass());
    /* We want to add the ipregionpass here because there has been no code motion up until this point, so all operations
       (except for constants) should be in c-line-order */
    if (!ipRangeMetadata.empty())
      pm.addPass(polygeist::createExtractIPPass(ipRangeMetadata.list[0].startLine, ipRangeMetadata.list[0].endLine));  // NEW
    optPM.addPass(mlir::createLoopInvariantCodeMotionPass());
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(polygeist::createCanonicalizeForPass());
    optPM.addPass(mlir::createCanonicalizerPass());
    if (RaiseToAffine && !CudaLower) {
      optPM.addPass(polygeist::createCanonicalizeForPass());
      optPM.addPass(mlir::createCanonicalizerPass());
      optPM.addPass(mlir::createLoopInvariantCodeMotionPass());
      optPM.addPass(polygeist::createRaiseSCFToAffinePass());
      optPM.addPass(polygeist::replaceAffineCFGPass());
      if (ScalarReplacement)
        optPM.addPass(mlir::createAffineScalarReplacementPass());
    }
    if (mlir::failed(pm.run(module.get()))) {
      module->dump();
      return 4;
    }
    if (mlir::failed(mlir::verify(module.get()))) {
      module->dump();
      return 5;
    }

#define optPM optPM2
#define pm pm2
    {
      mlir::PassManager pm(&context);
      mlir::OpPassManager &optPM = pm.nest<mlir::FuncOp>();

      if (DetectReduction)
        optPM.addPass(polygeist::detectReductionPass());

      // Disable inlining for -O0
      if (!Opt0) {
        optPM.addPass(mlir::createCanonicalizerPass());
        optPM.addPass(mlir::createCSEPass());
        if (!ExtractIP)  /* At the moment IP extraction requires the program to disable inlining in order to retain the IP call */
          pm.addPass(mlir::createInlinerPass());
      }
      if (mlir::failed(pm.run(module.get()))) {
        module->dump();
        return 4;
      }
    }

    if (CudaLower) {
      mlir::PassManager pm(&context);
      mlir::OpPassManager &optPM = pm.nest<mlir::FuncOp>();
      optPM.addPass(mlir::createCanonicalizerPass());
      pm.addPass(polygeist::createParallelLowerPass());
      mlir::OpPassManager &noptPM = pm.nest<mlir::FuncOp>();
      noptPM.addPass(mlir::createCanonicalizerPass());
      noptPM.addPass(polygeist::createMem2RegPass());
      noptPM.addPass(polygeist::replaceAffineCFGPass());
      noptPM.addPass(mlir::createCanonicalizerPass());
      if (!ExtractIP)  /* At the moment IP extraction requires the program to disable inlining in order to retain the IP call */
        pm.addPass(mlir::createInlinerPass());
      if (mlir::failed(pm.run(module.get()))) {
        module->dump();
        return 4;
      }
    }

    mlir::PassManager pm(&context);
    mlir::OpPassManager &optPM = pm.nest<mlir::FuncOp>();
    if (CudaLower) {
      optPM.addPass(mlir::createCanonicalizerPass());
      optPM.addPass(polygeist::createMem2RegPass());
      optPM.addPass(mlir::createCSEPass());
      optPM.addPass(mlir::createCanonicalizerPass());
      optPM.addPass(polygeist::createCanonicalizeForPass());
      optPM.addPass(mlir::createCanonicalizerPass());

      if (RaiseToAffine) {
        optPM.addPass(polygeist::createCanonicalizeForPass());
        optPM.addPass(mlir::createCanonicalizerPass());
        optPM.addPass(mlir::createLoopInvariantCodeMotionPass());
        optPM.addPass(polygeist::createRaiseSCFToAffinePass());
        optPM.addPass(polygeist::replaceAffineCFGPass());
        if (ScalarReplacement)
          optPM.addPass(mlir::createAffineScalarReplacementPass());
      }
      if (ToCPU == "continuation") {
        optPM.addPass(polygeist::createBarrierRemovalContinuation());
        // pm.nest<mlir::FuncOp>().addPass(mlir::createCanonicalizerPass());
      } else if (ToCPU.size() != 0) {
        optPM.addPass(polygeist::createCPUifyPass(ToCPU));
      }
      optPM.addPass(mlir::createCanonicalizerPass());
    }
    pm.addPass(mlir::createSymbolDCEPass());

    if (EmitLLVM || !EmitAssembly) {
      pm.addPass(mlir::createLowerAffinePass());
      pm.nest<mlir::FuncOp>().addPass(mlir::createConvertMathToLLVMPass());
      /* This appears to be where the PassManager applies its pass
       * transformations to the MLIR module */
      if (mlir::failed(pm.run(module.get()))) {
        module->dump();
        return 4;
      }
      mlir::PassManager pm2(&context);
      if (SCFOpenMP)
        pm2.addPass(createConvertSCFToOpenMPPass());
      if (mlir::failed(pm2.run(module.get()))) {
        module->dump();
        return 4;
      }
      module->walk([&](mlir::omp::ParallelOp) { LinkOMP = true; });
      mlir::PassManager pm3(&context);
      pm3.addPass(mlir::createLowerToCFGPass());
      LowerToLLVMOptions options(&context);
      options.dataLayout = DL;
      pm3.addPass(polygeist::createConvertPolygeistToLLVMPass(options));
      pm3.addPass(createConvertOpenMPToLLVMPass());
      // invalid for gemm.c init array
      // options.useBarePtrCallConv = true;
      pm3.addPass(mlir::createLowerToLLVMPass(options));
      if (mlir::failed(pm3.run(module.get()))) {
        module->dump();
        return 4;
      }
    } else {

      if (mlir::failed(pm.run(module.get()))) {
        module->dump();
        return 4;
      }
    }
    if (mlir::failed(mlir::verify(module.get()))) {
      module->dump();
      return 5;
    }
  }

  // module->dump();
  // Block &bodyBlk = module->getRegion().getBlocks().front();
  // Block &funcBodyBlk = bodyBlk.front().getRegion(0).getBlocks().front();
  // // auto funcBlk = bodyBlk.front().getRegion(0).getBlocks().front();
  // for (Operation &bodyOp : funcBodyBlk) {
  //   FileLineColLoc lineData = bodyOp.getLoc().dyn_cast<FileLineColLoc>();
  //   std::cout << "\n===== OPERATION =====\n";
  //   bodyOp.dump();
  //   if (lineData != NULL) {
  //     std::cout << "\nThis operation corresponds with C line number: " <<
  //     lineData.getLine() << "\n";
  //   } else {
  //     std::cout << "\nCould not dynamic cast this location\n";
  //   }
  //   std::cout << "\n";
  // }

  // for (Operation &bodyOp : funcBodyBlk) {
  //   printLineNumbers(bodyOp);
  // }

  if (EmitLLVM || !EmitAssembly) {
    llvm::LLVMContext llvmContext;
    auto llvmModule = mlir::translateModuleToLLVMIR(module.get(), llvmContext);
    if (!llvmModule) {
      module->dump();
      llvm::errs() << "Failed to emit LLVM IR\n";
      return -1;
    }
    llvmModule->setDataLayout(DL);
    llvmModule->setTargetTriple(triple.getTriple());
    if (!EmitAssembly) {
      auto tmpFile =
          llvm::sys::fs::TempFile::create("/tmp/intermediate%%%%%%%.ll");
      if (!tmpFile) {
        llvm::errs() << "Failed to create temp file\n";
        return -1;
      }
      std::error_code EC;
      {
        llvm::raw_fd_ostream out(tmpFile->FD, /*shouldClose*/ false);
        out << *llvmModule << "\n";
        out.flush();
      }
      int res =
          emitBinary(argv[0], tmpFile->TmpName.c_str(), LinkageArgs, LinkOMP);
      if (tmpFile->discard()) {
        llvm::errs() << "Failed to erase temp file\n";
        return -1;
      }
      return res;
    } else if (Output == "-") {
      llvm::outs() << *llvmModule << "\n";
    } else {
      std::error_code EC;
      llvm::raw_fd_ostream out(Output, EC);
      out << *llvmModule << "\n";
    }

  } else {
    if (Output == "-")
      module->print(outs());
    else {
      std::error_code EC;
      llvm::raw_fd_ostream out(Output, EC);
      module->print(out);
    }
  }
  return 0;
}
