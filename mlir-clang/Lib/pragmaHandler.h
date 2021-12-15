//===- pragmaHandler.ch - Pragmas used to emit MLIR---------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRCLANG_LIB_PRAGMAHANDLER_H
#define MLIR_TOOLS_MLIRCLANG_LIB_PRAGMAHANDLER_H

#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/Sema.h"

#include "llvm/ADT/DenseMap.h"

/// POD holds information processed from the lower_to pragma.
struct LowerToInfo {
  llvm::StringMap<std::string> SymbolTable;
  llvm::SmallVector<llvm::StringRef, 2> InputSymbol;
  llvm::SmallVector<llvm::StringRef, 2> OutputSymbol;
};

/// The location of the scop, as delimited by scop and endscop
/// pragmas by the user.
/// "scop" and "endscop" are the source locations of the scop and
/// endscop pragmas.
/// "start_line" is the line number of the start position.
struct ScopLoc {
  ScopLoc() : end(0) {}

  clang::SourceLocation scop;
  clang::SourceLocation endscop;
  unsigned startLine;
  unsigned start;
  unsigned end;
};

/// Taken from pet.cc
/// List of pairs of #pragma scop and #pragma endscop locations.
struct ScopLocList {
  std::vector<ScopLoc> list;

  // Add a new start (#pragma scop) location to the list.
  // If the last #pragma scop did not have a matching
  // #pragma endscop then overwrite it.
  // "start" points to the location of the scop pragma.

  void addStart(clang::SourceManager &SM, clang::SourceLocation start) {
    ScopLoc loc;

    loc.scop = start;
    int line = SM.getExpansionLineNumber(start);
    start = SM.translateLineCol(SM.getFileID(start), line, 1);
    loc.startLine = line;
    loc.start = SM.getFileOffset(start);
    if (list.size() == 0 || list[list.size() - 1].end != 0)
      list.push_back(loc);
    else
      list[list.size() - 1] = loc;
  }

  // Set the end location (#pragma endscop) of the last pair
  // in the list.
  // If there is no such pair of if the end of that pair
  // is already set, then ignore the spurious #pragma endscop.
  // "end" points to the location of the endscop pragma.

  void addEnd(clang::SourceManager &SM, clang::SourceLocation end) {
    if (list.size() == 0 || list[list.size() - 1].end != 0)
      return;
    list[list.size() - 1].endscop = end;
    int line = SM.getExpansionLineNumber(end);
    end = SM.translateLineCol(SM.getFileID(end), line + 1, 1);
    list[list.size() - 1].end = SM.getFileOffset(end);
  }

  // Check if the current location is in the scop.
  bool isInScop(clang::SourceLocation target) {
    if (!list.size())
      return false;
    for (auto &scopLoc : list)
      if ((target >= scopLoc.scop) && (target <= scopLoc.endscop))
        return true;
    return false;
  }
};

void addPragmaLowerToHandlers(clang::Preprocessor &PP, LowerToInfo &LTInfo);
void addPragmaScopHandlers(clang::Preprocessor &PP, ScopLocList &scopLocList);
void addPragmaEndScopHandlers(clang::Preprocessor &PP, ScopLocList &scopLocList);

/* EMIT IP REGION FEATURE */
struct IPLocList;
void addPragmaIPDefHandlers(clang::Preprocessor &PP, IPLocList &ipLocList);
void addPragmaIPEndHandlers(clang::Preprocessor &PP, IPLocList &ipLocList);

struct IPLoc {
  IPLoc() : end(0) {}

  clang::SourceLocation ipdef;
  clang::SourceLocation ipend;
  unsigned startLine;
  unsigned endLine;
  unsigned start;
  unsigned end;
};

/* IPREGION COMPLETE */
struct IPLocList {
  std::vector<IPLoc> list;

  // Add a new start (#pragma ipdef) location to the list.
  // If the last #pragma ipdef did not have a matching
  // #pragma ipend then overwrite it.
  // "start" points to the location of the ipdef pragma.

  void addStart(clang::SourceManager &SM, clang::SourceLocation start) {
    IPLoc loc;

    loc.ipdef = start;
    int line = SM.getExpansionLineNumber(start);
    start = SM.translateLineCol(SM.getFileID(start), line, 1);
    loc.startLine = line;
    loc.start = SM.getFileOffset(start);
    if (list.size() == 0 || list[list.size() - 1].end != 0)
      list.push_back(loc);
    else
      list[list.size() - 1] = loc;
  }

  // Set the end location (#pragma ipend) of the last pair
  // in the list.
  // If there is no such pair of if the end of that pair
  // is already set, then ignore the spurious #pragma ipend.
  // "end" points to the location of the ipend pragma.

  void addEnd(clang::SourceManager &SM, clang::SourceLocation end) {
    if (list.size() == 0 || list[list.size() - 1].end != 0)
      return;
    list[list.size() - 1].ipend = end;
    int line = SM.getExpansionLineNumber(end);
    end = SM.translateLineCol(SM.getFileID(end), line + 1, 1);
    list[list.size() - 1].end = SM.getFileOffset(end);
    list[list.size() - 1].endLine = line;
  }

  // Check if the current location is in the scop.
  bool isInIP(clang::SourceLocation target) const {
    if (!list.size())
      return false;
    for (auto &IPLoc : list)
      if ((target >= IPLoc.ipdef) && (target <= IPLoc.ipend))
        return true;
    return false;
  }

  bool isInIP(unsigned line) const {
    /* This function currently returns true if the line is in AT LEAST ONE OF multiple IP line ranges */
    for (const IPLoc &ipRegion : list) {
      if (line > ipRegion.startLine && line < ipRegion.endLine) {
        return true;
      }
    }
    return false;
  }
};

#endif
