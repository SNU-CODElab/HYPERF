#include <clang/AST/AST.h>
#include <clang/AST/ASTConsumer.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/FrontendActions.h>
#include <clang/Tooling/Tooling.h>
#include <clang/Tooling/CompilationDatabase.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/raw_ostream.h>

// OpenMP Visitor: AST를 탐색하며 OpenMP 관련 노드를 처리
class OMPVisitor : public clang::RecursiveASTVisitor<OMPVisitor> {
public:
    explicit OMPVisitor(clang::ASTContext &Context) : Context(Context) {}

    bool VisitStmt(clang::Stmt *S) {
        // OpenMP Directive를 탐지
        if (auto *OMPDirective = llvm::dyn_cast<clang::OMPExecutableDirective>(S)) {
            llvm::outs() << "Found OpenMP Directive: " << OMPDirective->getStmtClassName() << "\n";

            // Clause 순회
            for (clang::OMPClause *Clause : OMPDirective->clauses()) {
                if (Clause) {
                    llvm::outs() << "  Clause Kind: " << Clause->getClauseKind() << "\n";
                }
            }
        }
        return true;
    }

private:
    clang::ASTContext &Context;
};

// AST Consumer: Visitor를 호출하여 AST를 처리
class OMPConsumer : public clang::ASTConsumer {
public:
    explicit OMPConsumer(clang::ASTContext &Context) : Visitor(Context) {}

    void HandleTranslationUnit(clang::ASTContext &Context) override {
        // AST의 최상위 노드(TranslationUnitDecl) 순회 시작
        Visitor.TraverseDecl(Context.getTranslationUnitDecl());
    }

private:
    OMPVisitor Visitor;
};

// Frontend Action: ASTConsumer 생성
class OMPFrontendAction : public clang::ASTFrontendAction {
public:
    std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(clang::CompilerInstance &CI, llvm::StringRef file) override {
        return std::make_unique<OMPConsumer>(CI.getASTContext());
    }
};

int main(int argc, const char **argv) {
    // 1. Command Line Option Parsing
    llvm::cl::OptionCategory ToolCategory("OpenMP Parsing Tool");

    // 2. Fixed Compilation Database
    std::string ErrorMessage;
    auto Compilations = clang::tooling::FixedCompilationDatabase::loadFromCommandLine(argc, argv, ErrorMessage);
    if (!Compilations) {
        llvm::errs() << "Error parsing compilation database: " << ErrorMessage << "\n";
        return 1;
    }

    // 3. File List
    std::vector<std::string> SourcePaths = clang::tooling::getSourcePathList(argc, argv);
    if (SourcePaths.empty()) {
        llvm::errs() << "Error: No input files provided.\n";
        return 1;
    }

    // 4. Clang Tool Initialization
    clang::tooling::ClangTool Tool(*Compilations, SourcePaths);

    // 5. Tool Execution
    return Tool.run(clang::tooling::newFrontendActionFactory<OMPFrontendAction>().get());
}