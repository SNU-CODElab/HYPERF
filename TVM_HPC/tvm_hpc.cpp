#include <iostream>
#include <cstdlib>
#include <string>
#include <fstream>
#include <vector>
#include <memory>
#include <filesystem>

// Clang LibTooling 관련 헤더 (경로 및 환경에 따라 다름)
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "llvm/Support/CommandLine.h"
#include "clang/Frontend/CompilerInstance.h"

#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <dlfcn.h>
#include <iostream>
#include <cstring>


using namespace clang;
using namespace clang::tooling;
using namespace clang::ast_matchers;

static int FunctionCount = 0;
// ----------------------------
// 1. 초기 컴파일 & TIR 추출 단계
// ----------------------------
bool runInitialCompileAndExtractTIR(const std::string &inputCppPath, const std::string &compileopt) {
    // 기존 컴파일 명령어 + include path
    std::string command = "clang -fPIC -fopenmp -shared -g -I/root/test/gemm/PolyBench-ACC/OpenMP/utilities -I /root/test/benchmarks/sw4lite/src/double/ -I /usr/lib/x86_64-linux-gnu/openmpi/include "
                          + compileopt+ " " + inputCppPath + " -o " + inputCppPath + "_preprocess_tvm.so";
    std::cout << "Running: " << command << std::endl;
    int ret = std::system(command.c_str());
    if (ret == -1) {
        std::cerr << "Failed to invoke shell" << std::endl;
        return false;
    } else if (WIFEXITED(ret) && WEXITSTATUS(ret) != 0) {
        std::cerr << "Command failed with exit code: " << WEXITSTATUS(ret) << std::endl;
        return false;
    }

    return true;

}

// ----------------------------
// 2. Python 스크립트를 이용한 TVM 빌드 단계
// ----------------------------
bool runTVMBuild(const std::string &libraryPath, const std::string &tirScriptPath, const std::string &outputSoPath, const std::string &ParamsPath) {
    std::string command = "python /root/TVM_HPC/build_with_tvm.py --library_path " + libraryPath +
                          " --tir_script_path " + tirScriptPath + " --output_so_path " + outputSoPath + " --params_path " + ParamsPath;
    std::cout << "Running: " << command << std::endl;
    int ret = system(command.c_str());
    if (ret != 0) {
        std::cerr << "TVM build script failed" << std::endl;
        return false;
    }
    if (!std::filesystem::exists(outputSoPath)) {
        std::cerr << "Final TVM .so not generated" << std::endl;
        return false;
    }
    return true;
}


// ----------------------------
//  (기존) 배열 선언에 alignas(64) 추가하기
// ----------------------------
class AlignasCallback : public MatchFinder::MatchCallback {
public:
    AlignasCallback(Rewriter &R, const std::vector<std::string>& varNames) 
        : TheRewriter(R), varNames(varNames) {}

    virtual void run(const MatchFinder::MatchResult &Result) override {
        return;
        if (const VarDecl *VD = Result.Nodes.getNodeAs<VarDecl>("varDecl")) {
            std::string VarName = VD->getNameAsString();

            // 특정 varNames에만 적용
            if (std::find(varNames.begin(), varNames.end(), VarName) == varNames.end()) {
                return;
            }

            // 이미 alignas가 붙어있다면 무시
            if (VD->hasAttr<AlignedAttr>()) {
                return;
            }
            // for문 내부 변수/함수 인자 등은 무시
            const Stmt *ParentStmt = Result.Context->getParents(*VD)[0].get<Stmt>();
            if (ParentStmt && isa<ForStmt>(ParentStmt)) {
                return;
            }
            if (isa<ParmVarDecl>(VD)) {
                return;
            }
            SourceLocation StartLoc = VD->getBeginLoc();
            if (!StartLoc.isValid() || !TheRewriter.isRewritable(StartLoc)) {
                return;
            }

            // 형식, 선언문 추출
            QualType VarType = VD->getType();
            PrintingPolicy Policy = PrintingPolicy(LangOptions());
            Policy.SuppressUnwrittenScope = true;
            Policy.SuppressTagKeyword = true;

            std::string FullType = VarType.getAsString(Policy);
            std::string NewDecl;
            if (VarType->isArrayType()) {
                // 배열
                std::string BaseType = GetBaseType(VarType, Policy);
                std::string ArraySizes = GetArraySizes(VarType);
                NewDecl = "alignas(64) " + BaseType + " " + VarName + ArraySizes;
                if (VD->hasInit()) {
                    std::string InitValue;
                    llvm::raw_string_ostream OS(InitValue);
                    VD->getInit()->printPretty(OS, nullptr, Policy);
                    OS.flush();
                    NewDecl += " = " + InitValue;
                }
                NewDecl += ";";
            }
            else if (VarType->isBuiltinType()) {
                // 기본형
                NewDecl = "alignas(64) " + FullType + " " + VarName;
                if (VD->hasInit()) {
                    std::string InitValue;
                    llvm::raw_string_ostream OS(InitValue);
                    VD->getInit()->printPretty(OS, nullptr, Policy);
                    OS.flush();
                    NewDecl += " = " + InitValue;
                }
                NewDecl += ";";
            }

            if (!NewDecl.empty()) {
                SourceLocation StartLoc = VD->getBeginLoc();
                SourceLocation EndLoc = VD->getEndLoc();
                EndLoc = Lexer::getLocForEndOfToken(EndLoc, 0, *Result.SourceManager, Result.Context->getLangOpts());
                SourceRange DeclRange(StartLoc, EndLoc);

                llvm::StringRef OriginalText = Lexer::getSourceText(
                    CharSourceRange::getCharRange(DeclRange), 
                    *Result.SourceManager, 
                    Result.Context->getLangOpts());
                llvm::errs() << "Original Text: " << OriginalText << "\n";

                TheRewriter.ReplaceText(DeclRange, NewDecl);
            }
        }
    }

private:
    Rewriter &TheRewriter;
    std::vector<std::string> varNames;

    std::string GetBaseType(QualType QT, const PrintingPolicy &Policy) {
        if (const ConstantArrayType *CAT = dyn_cast<ConstantArrayType>(QT)) {
            return GetBaseType(CAT->getElementType(), Policy);
        }
        return QT.getAsString(Policy);
    }

    std::string GetArraySizes(QualType QT) {
        std::string Result;
        if (const ConstantArrayType *CAT = dyn_cast<ConstantArrayType>(QT)) {
            llvm::APInt Size = CAT->getSize();
            Result = "[" + std::to_string(Size.getZExtValue()) + "]";
            Result += GetArraySizes(CAT->getElementType());
        }
        return Result;
    }
};


// ----------------------------
//  (기존) new float[...] → (float*)std::aligned_alloc(...)
//      -- 단, 선언과 할당이 한 줄에 있는 경우만 매칭
// ----------------------------
class ReplaceNewFloatArrayCallback : public MatchFinder::MatchCallback {
public:
    explicit ReplaceNewFloatArrayCallback(Rewriter &R) : TheRewriter(R) {}

    void run(const MatchFinder::MatchResult &Result) override {
        const VarDecl *VD = Result.Nodes.getNodeAs<VarDecl>("varDecl");
        if (!VD) return;

        const CXXNewExpr *NewE = Result.Nodes.getNodeAs<CXXNewExpr>("newExpr");
        if (!NewE) return;

        // 배열 길이 (예: new float[N])
        const Expr *ArraySizeE = Result.Nodes.getNodeAs<Expr>("arraySizeExpr");
        if (!ArraySizeE) return;

        // varDecl 이 실제로 new float[...] 으로 초기화되는지
        if (!VD->hasInit()) return;

        SourceRange NewExprRange = NewE->getSourceRange();
        if (!NewExprRange.isValid()) return;

        SourceManager &SM = *Result.SourceManager;
        LangOptions LangOpts = Result.Context->getLangOpts();
        StringRef SizeText = Lexer::getSourceText(
            CharSourceRange::getTokenRange(ArraySizeE->getSourceRange()),
            SM, LangOpts
        );
        if (SizeText.empty()) return;

        // 치환 예: (float*)std::aligned_alloc(64, (N)*sizeof(float))
        std::string ReplacementStr = "(float*)std::aligned_alloc(64, (";
        ReplacementStr += SizeText.str();
        ReplacementStr += ") * sizeof(float))";

        TheRewriter.ReplaceText(NewExprRange, ReplacementStr);
    }

private:
    Rewriter &TheRewriter;
};


// ----------------------------
//   (신규) 1) A_data = new float[N];  (정의/할당 분리)
// ----------------------------
class ReplaceNewFloatAssignmentCallback : public MatchFinder::MatchCallback {
public:
    explicit ReplaceNewFloatAssignmentCallback(Rewriter &R) : TheRewriter(R) {}

    void run(const MatchFinder::MatchResult &Result) override {
        if (const BinaryOperator *BO = Result.Nodes.getNodeAs<BinaryOperator>("assignNewFloat")) {
            // RHS: new float[ ... ]
            const CXXNewExpr *NewE = Result.Nodes.getNodeAs<CXXNewExpr>("newExpr");
            if (!NewE) return;

            // 배열 크기
            const Expr *ArraySizeE = Result.Nodes.getNodeAs<Expr>("arraySizeExpr");
            if (!ArraySizeE) return;

            // 바꿀 대상 소스 범위 (BO->getRHS())
            SourceRange RHSRange = BO->getRHS()->getSourceRange();
            if (!RHSRange.isValid()) return;

            SourceManager &SM = *Result.SourceManager;
            LangOptions LangOpts = Result.Context->getLangOpts();

            // 크기부분 텍스트 추출
            StringRef SizeText = Lexer::getSourceText(
                CharSourceRange::getTokenRange(ArraySizeE->getSourceRange()),
                SM, LangOpts
            );
            if (SizeText.empty()) return;

            // 예: A_data = new float[N*N];
            // -> A_data = (float*)std::aligned_alloc(64, (N*N)*sizeof(float));
            std::string ReplacementStr = "(float*)std::aligned_alloc(64, (";
            ReplacementStr += SizeText.str();
            ReplacementStr += ") * sizeof(float))";

            TheRewriter.ReplaceText(RHSRange, ReplacementStr);
        }
    }

private:
    Rewriter &TheRewriter;
};


// ----------------------------
//   (신규) 2) A_data = malloc(...);
//       → A_data = (float*)std::aligned_alloc(64, ...);
// ----------------------------
class ReplaceMallocFloatAssignmentCallback : public MatchFinder::MatchCallback {
public:
    explicit ReplaceMallocFloatAssignmentCallback(Rewriter &R) : TheRewriter(R) {}

    void run(const MatchFinder::MatchResult &Result) override {
        if (const BinaryOperator *BO = Result.Nodes.getNodeAs<BinaryOperator>("assignMallocFloat")) {
            // RHS: malloc(...)
            const CallExpr *CE = Result.Nodes.getNodeAs<CallExpr>("mallocCall");
            if (!CE) return;

            // malloc 인자가 1개 이상인지 확인
            if (CE->getNumArgs() < 1) return;
            const Expr *ArgE = CE->getArg(0);
            if (!ArgE) return;

            // 바꿀 대상 소스 범위
            SourceRange RHSRange = BO->getRHS()->getSourceRange();
            if (!RHSRange.isValid()) return;

            SourceManager &SM = *Result.SourceManager;
            LangOptions LangOpts = Result.Context->getLangOpts();

            // malloc(...) 의 첫 번째 인자 텍스트 (예: N*N*sizeof(float))
            StringRef ArgText = Lexer::getSourceText(
                CharSourceRange::getTokenRange(ArgE->getSourceRange()),
                SM, LangOpts
            );
            if (ArgText.empty()) return;

            // 치환
            // A_data = malloc(N*N*sizeof(float));
            // -> A_data = (float*)std::aligned_alloc(64, N*N*sizeof(float));
            std::string ReplacementStr = "(float*)std::aligned_alloc(64, ";
            ReplacementStr += ArgText.str();
            ReplacementStr += ")";

            TheRewriter.ReplaceText(RHSRange, ReplacementStr);
        }
    }

private:
    Rewriter &TheRewriter;
};


// ----------------------------
//   (추가) 3) delete x / delete[] x → std::free(x)
// ----------------------------
class ReplaceDeleteCallback : public MatchFinder::MatchCallback {
public:
    explicit ReplaceDeleteCallback(Rewriter &R) : TheRewriter(R) {}

    void run(const MatchFinder::MatchResult &Result) override {
        if (const CXXDeleteExpr *DelE = Result.Nodes.getNodeAs<CXXDeleteExpr>("deleteExpr")) {
            // 삭제할 대상 포인터
            const Expr *Arg = DelE->getArgument();
            if (!Arg) return;

            SourceRange DelRange = DelE->getSourceRange();
            if (!DelRange.isValid()) return;

            SourceManager &SM = *Result.SourceManager;
            LangOptions LangOpts = Result.Context->getLangOpts();

            // 예: delete[] myPtr;  ->  std::free(myPtr)
            StringRef ArgText = Lexer::getSourceText(
                CharSourceRange::getTokenRange(Arg->getSourceRange()),
                SM, LangOpts
            );
            if (ArgText.empty()) return;

            // delete X; or delete[] X; -> std::free(X)
            std::string ReplacementStr = "std::free(" + ArgText.str() + ")";
            TheRewriter.ReplaceText(DelRange, ReplacementStr);
        }
    }

private:
    Rewriter &TheRewriter;
};


// ----------------------------
//   #pragma omp tvm → 사용자 정의 코드로 대체
// ----------------------------
class OMPCallback : public MatchFinder::MatchCallback {
public:
    OMPCallback(Rewriter &R, const std::map<std::string, std::string>& ReplacementCodeMap) 
        : TheRewriter(R), ReplacementCodeMap(ReplacementCodeMap) {}

    virtual void run(const MatchFinder::MatchResult &Result) {
        if (const auto *OMPDir = Result.Nodes.getNodeAs<OMPAutotuneForDirective>("ompAutotuneFor")) {
            SourceLocation Start = OMPDir->getBeginLoc();

            // CapturedStmt가 없을 가능성을 처리
            const CapturedStmt *CS = nullptr;
            if (OMPDir->hasAssociatedStmt()) {
                CS = OMPDir->getCapturedStmt(llvm::omp::OMPD_autotune_for);
            }

            SourceLocation End = (CS) ? CS->getEndLoc() : OMPDir->getEndLoc();


            // 전체 Directive 범위를 교체
            SourceRange FullRange(Start, End);
            std::string key = "main"+std::to_string(FunctionCount);
            TheRewriter.ReplaceText(FullRange, ReplacementCodeMap[key]);
            FunctionCount++;

            std::cout << "Replaced Directive from "
                      << FullRange.getBegin().printToString(TheRewriter.getSourceMgr())
                      << " to "
                      << FullRange.getEnd().printToString(TheRewriter.getSourceMgr())
                      << "\n";
        }
    }

private:
    Rewriter &TheRewriter;
    std::map<std::string, std::string> ReplacementCodeMap;
};


// ----------------------------
//   OMPConsumer: AST 매칭 등록
// ----------------------------
class OMPConsumer : public ASTConsumer {
public:
    OMPConsumer(Rewriter &R, 
                const std::map<std::string, std::string>& ReplacementCodeMap,
                const std::vector<std::string>& varNames,
                std::string InputFilePath
                ) 
        : HandlerForTvm(R, ReplacementCodeMap)
        , HandlerForAlignas(R, varNames)
        , HandlerForNewFloatArray(R)
        , HandlerForNewFloatAssignment(R)
        , HandlerForMallocAssignment(R)
        , HandlerForDelete(R)   // << 추가: Delete 콜백
    {
        // 1) 배열 또는 기본형 varDecl에 alignas(64) 추가
        Finder.addMatcher(
            varDecl(hasType(type(anyOf(arrayType(), builtinType())))).bind("varDecl"),
            &HandlerForAlignas
        );

        // 2) #pragma omp tvm 치환
        Finder.addMatcher(
            ompExecutableDirective().bind("ompAutotuneFor"), 
            &HandlerForTvm
        );

        // 3) (기존) float* A = new float[N]; (한 줄)
        Finder.addMatcher(
            varDecl(
                hasInitializer(
                    cxxNewExpr(
                        hasType(pointsTo(asString("float"))),
                        hasArraySize(expr().bind("arraySizeExpr"))
                    ).bind("newExpr")
                )
            ).bind("varDecl"),
            &HandlerForNewFloatArray
        );

        // 4) (신규) A = new float[N]; (정의/할당 분리)
        Finder.addMatcher(
            binaryOperator(
                hasOperatorName("="),
                hasRHS(
                    cxxNewExpr(
                        hasType(pointsTo(asString("float"))),
                        hasArraySize(expr().bind("arraySizeExpr"))
                    ).bind("newExpr")
                )
            ).bind("assignNewFloat"),
            &HandlerForNewFloatAssignment
        );

        // 5) (신규) A = malloc(...);
        Finder.addMatcher(
            binaryOperator(
                hasOperatorName("="),
                hasRHS(
                    callExpr(
                        callee(functionDecl(hasName("malloc")))
                    ).bind("mallocCall")
                )
            ).bind("assignMallocFloat"),
            &HandlerForMallocAssignment
        );

        // 6) (추가) delete/delete[] -> std::free(...)
        Finder.addMatcher(
            cxxDeleteExpr().bind("deleteExpr"),
            &HandlerForDelete
        );
    }

    void HandleTranslationUnit(ASTContext &Context) override {
        Finder.matchAST(Context);
    }

private:
    MatchFinder Finder;

    OMPCallback                     HandlerForTvm;
    AlignasCallback                 HandlerForAlignas;
    ReplaceNewFloatArrayCallback    HandlerForNewFloatArray;
    ReplaceNewFloatAssignmentCallback HandlerForNewFloatAssignment;
    ReplaceMallocFloatAssignmentCallback HandlerForMallocAssignment;
    ReplaceDeleteCallback           HandlerForDelete;  // << 추가
};


// ----------------------------
//   OMPFrontendAction
// ----------------------------
class OMPFrontendAction : public ASTFrontendAction {
public:
    OMPFrontendAction(const std::map<std::string, std::string> ReplacementCodeMap,
                      const std::map<std::string, std::string> LoadCodeMap,
                      const std::vector<std::string>& varNames,
                      std::string InputFilePath
                      )
        : ReplacementCodeMap(ReplacementCodeMap),LoadCodeMap(LoadCodeMap), varNames(varNames),InputFilePath(InputFilePath) {}

    void EndSourceFileAction() override {
        SourceManager &SM = TheRewriter.getSourceMgr();
        FileID FID = SM.getMainFileID();
        SourceLocation FileStartLoc = SM.getLocForStartOfFile(FID);

        // 결과 코드 상단에 필요한 헤더 삽입
        std::string Includes = R"INCLUDES(#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <dlfcn.h>
#include <iostream>
#include <chrono>
)INCLUDES";
        Includes += "auto start = std::chrono::system_clock::now();";
        for (const auto& [key, value] : LoadCodeMap) {
            Includes += value;
        }
        Includes += "auto end = std::chrono::system_clock::now();";
        Includes += "std::chrono::duration<double> import_seconds = end-start;\n";
        TheRewriter.InsertText(FileStartLoc, Includes, true, true);

        // 최종 변환 결과 출력
        if (const RewriteBuffer *RewriteBuf = TheRewriter.getRewriteBufferFor(SM.getMainFileID())) {
            std::ofstream outFile(InputFilePath+"_replaced.cpp");
            outFile << std::string(RewriteBuf->begin(), RewriteBuf->end());
            outFile.close();
            llvm::outs() << "Transformed source saved to"+ InputFilePath+"_replaced.cpp\n";

            // std::string transformedCode(RewriteBuf->begin(), RewriteBuf->end());

            // // TVM_HOME 등등...
            // const char* tvmHome = std::getenv("TVM_HOME");
            // if (!tvmHome) {
            //     std::cerr << "TVM_HOME environment variable is not set." << std::endl;
            // }

            // // 실제로 컴파일까지 시도
            // std::string compileCmd = "clang -x c++ -O3 -march=native ";
            // compileCmd += "-I/root/test/gemm/polybench-c-3.2/utilities ";
            // compileCmd += "-I" + std::string(tvmHome) + "/include ";
            // compileCmd += "-I" + std::string(tvmHome) + "/3rdparty/dmlc-core/include ";
            // compileCmd += "-I" + std::string(tvmHome) + "/3rdparty/dlpack/include ";
            // compileCmd += "-I" + std::string(tvmHome) + "/3rdparty/rang/include ";
            // compileCmd += "-I" + std::string(tvmHome) + "/src/target/spirv ";
            // compileCmd += "-L" + std::string(tvmHome) + "/build -ltvm_runtime -ldl -o transformed_output";

            // std::cout << "Compile command: " << compileCmd << std::endl;
            // FILE *clangPipe = popen(compileCmd.c_str(), "w");
            // if (!clangPipe) {
            //     llvm::errs() << "Failed to start clang++ process\n";
            //     return;
            // }

            // // 변환된 코드를 파이프로 전달
            // if (fwrite(transformedCode.data(), 1, transformedCode.size(), clangPipe) != transformedCode.size()) {
            //     llvm::errs() << "Failed to write all code to clang++ stdin\n";
            // }

            // int status = pclose(clangPipe);
            // if (status == 0) {
            //     llvm::outs() << "Compilation successful. Run ./transformed_output\n";
            // } else {
            //     llvm::errs() << "Compilation failed.\n";
            // }
        }
    }

    std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI, StringRef inFile) override {
        TheRewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
        return std::make_unique<OMPConsumer>(TheRewriter, ReplacementCodeMap, varNames, InputFilePath);
    }

private:
    Rewriter TheRewriter;
    std::map<std::string, std::string> ReplacementCodeMap;
    std::map<std::string, std::string> LoadCodeMap; 
    std::vector<std::string> varNames;
    std::string InputFilePath;
};


// ----------------------------
//   OMPFrontendActionFactory
// ----------------------------
class OMPFrontendActionFactory : public clang::tooling::FrontendActionFactory {
    std::map<std::string, std::string> ReplacementCodeMap;
    std::map<std::string, std::string> LoadCodeMap;
    std::vector<std::string> varNames;
    std::string InputFilePath;
public:
     OMPFrontendActionFactory(const std::map<std::string, std::string>& code, const std::map<std::string, std::string>& code_load, const std::vector<std::string>& names, std::string inputfilepath) 
        : ReplacementCodeMap(code),LoadCodeMap(code_load), varNames(names),InputFilePath(inputfilepath)  {}

    std::unique_ptr<clang::FrontendAction> create() override {
        return std::make_unique<OMPFrontendAction>(ReplacementCodeMap,LoadCodeMap ,varNames, InputFilePath);
    }
};



// ----------------------------
//   TVM 호출코드 생성 함수
// ----------------------------

std::string MakeLoadCode(const std::string& finalSoPath, 
                            const std::string& paramsPath, 
                            std::vector<std::string>& varNames,
                            const std::string& funcNameforCheck) 
                            {
    std::string loadCode = "// TVM 모듈 호출 코드\n";
     if(funcNameforCheck == "main0"){
        loadCode += "tvm::runtime::Module mod = tvm::runtime::Module::LoadFromFile(\"" + finalSoPath + "\");\n";
        loadCode += "DLDevice dev{kDLCPU, 0};\n\n";
    }
    loadCode += "tvm::runtime::PackedFunc f"+funcNameforCheck+" = mod.GetFunction(\""+funcNameforCheck+"\");\n";
    return loadCode;
}

std::string MakeReplaceCode(const std::string& finalSoPath, 
                            const std::string& paramsPath, 
                            std::vector<std::string>& varNames,
                            const std::string& funcNameforCheck) 
{
    std::string replacementCode = "// TVM 모듈 호출 코드\n";
    // replacementCode += "auto start_nd = std::chrono::high_resolution_clock::now();\n";
   
  

    std::ifstream file(paramsPath);
    std::string line;
    
    std::vector<std::string> shapes;
    std::vector<std::string> varNames_nd;
    std::map<std::string, std::string> memcpy_vars;
    std::map<std::string, std::string> memcpy_vars_org;

    while (getline(file, line)) {
        std::istringstream iss(line);
        std::string varName, shapeStr, dtype, funcName;

        size_t funcNameStart = line.find("Function Name: ") + 15;
        size_t funcNameEnd = line.find(", Variable Name:");
        funcName = line.substr(funcNameStart, funcNameEnd - funcNameStart);

        if (funcNameforCheck != funcName) {
            std::cout << "Skipping function: " << funcName << std::endl;
            continue; // main0과 다르면 무시하고 다음 라인으로
        }

        
        size_t nameStart = line.find("Variable Name: ") + 15;
        size_t nameEnd   = line.find(", Shape:");
        std::string varName_org = line.substr(nameStart, nameEnd - nameStart);
        varNames.push_back(varName_org);
        
        std::string varName_replaced = varName_org;
        std::replace(varName_replaced.begin(), varName_replaced.end(), '.', '_');
        std::cout << "varName_replaced: " << varName_replaced << std::endl; 
        varName = varName_replaced + "_" + funcNameforCheck;

        size_t shapeStart = line.find("{") + 1;
        size_t shapeEnd   = line.find("}");
        shapeStr = line.substr(shapeStart, shapeEnd - shapeStart);
        std::cout << "shapeStr: " << shapeStr << std::endl;

        size_t typeStart = line.find("Data Type: ") + 11;
        size_t typeEnd   = line.find(", Buffer type: ");
        dtype = line.substr(typeStart, typeEnd - typeStart);
        size_t pos = dtype.find_first_of("0123456789");
        std::string baseType = dtype.substr(0, pos);
        std::string bitWidth = dtype.substr(pos); 
        
        size_t bufferTypeStart = line.find("Buffer type: ") + 13;
        std::string bufferOrgType = line.substr(bufferTypeStart); 

        // varNames.push_back(varName);
        shapes.push_back(shapeStr);
        varNames_nd.push_back(varName + "_nd");

        // shape 파싱
        std::vector<int64_t> shapeVec;
        {
            std::istringstream shapeStream(shapeStr);
            std::string dim;
            while (std::getline(shapeStream, dim, ',')) {
                shapeVec.push_back(std::stoll(dim));
            }
        }

        // DLTensor 세팅
        if (shapeVec.size() == 1 && shapeVec[0] == 1) {
            // 단일 스칼라 float, int 등
            std::string type_reg;
            if      (dtype == "kDLInt8")      type_reg = "int8_t";
            else if (dtype == "kDLInt16")     type_reg = "int16_t";
            else if (dtype == "kDLInt32")     type_reg = "int32_t";
            else if (dtype == "kDLInt64")     type_reg = "int64_t";
            else if (dtype == "kDLUInt8")     type_reg = "uint8_t";
            else if (dtype == "kDLUInt16")    type_reg = "uint16_t";
            else if (dtype == "kDLUInt32")    type_reg = "uint32_t";
            else if (dtype == "kDLUInt64")    type_reg = "uint64_t";
            else if (dtype == "kDLFloat16")   type_reg = "float16_t";
            else if (dtype == "kDLFloat32")   type_reg = "float";
            else if (dtype == "kDLFloat64")   type_reg = "double";
            else if (dtype == "kDLBfloat")    type_reg = "bfloat16_t";
            else if (dtype == "kDLOpaqueHandle") type_reg = "void*";
            // kDLUInt1
            else if (dtype == "kDLUInt1")     type_reg = "uint8_t"; // 1비트는 uint8_t로 처리
            else if (dtype == "kDLFloat8")    type_reg = "float8_t"; // float8은 별도의 타입으로 처리
            else if (dtype == "kDLFloat8E4M3") type_reg = "float8_e4m3_t"; // E4M3 포맷
            else if (dtype == "kDLFloat8E5M2") type_reg = "float8_e5m2_t"; // E5M2 포맷

            else {
                std::cerr << "Unsupported data type: " << dtype << std::endl;
                return "";
            }
            // replacementCode += varName + "_dltensor.data = reinterpret_cast<void*>(const_cast<" + type_reg + "*>(&" + varName + "));\n";

            replacementCode += "tvm::runtime::NDArray " + varName + "_nd = tvm::runtime::NDArray::Empty({1}, DLDataType {" + baseType + "," + bitWidth + ",1}, dev);\n";
            replacementCode += "*reinterpret_cast<" + type_reg + "*>(" + varName + "_nd.ToDLPack()->dl_tensor.data) = " + varName_org + ";\n\n";
            memcpy_vars[varName] = type_reg;
            memcpy_vars_org[varName] = varName_org;
            continue;
        }
        else if (bufferOrgType.find("*") != std::string::npos) {
            // 배열(포인터) 변수
            replacementCode += "DLTensor " + varName + "_dltensor;\n";
            replacementCode += varName + "_dltensor.data = reinterpret_cast<void*>(" + varName_org + ");\n";
        }
        else {
            // 그 외(직접적인 배열? 등)
            replacementCode += "DLTensor " + varName + "_dltensor;\n";
            replacementCode += varName + "_dltensor.data = " + varName_org + ";\n";
        }
        replacementCode += varName + "_dltensor.device = {kDLCPU, 0};\n";
        replacementCode += varName + "_dltensor.ndim = " + std::to_string(shapeVec.size()) + ";\n";
        replacementCode += varName + "_dltensor.dtype = DLDataType {" + baseType + "," + bitWidth + ",1};\n";
        replacementCode += "int64_t " + varName + "_shape[] = {";
        for (size_t i = 0; i < shapeVec.size(); ++i) {
            replacementCode += std::to_string(shapeVec[i]);
            if (i < shapeVec.size() - 1) replacementCode += ", ";
        }
        replacementCode += "};\n";
        replacementCode += varName + "_dltensor.shape = " + varName + "_shape;\n";
        replacementCode += varName + "_dltensor.strides = nullptr;\n";
        replacementCode += varName + "_dltensor.byte_offset = 0;\n";
        replacementCode += "tvm::runtime::NDArray " + varName + "_nd = tvm::runtime::NDArray::FromExternalDLTensor(" + varName + "_dltensor);\n\n";
    }

    // replacementCode += "auto end_nd = std::chrono::high_resolution_clock::now();\n";
    // replacementCode += "std::chrono::duration<double> elapsed_nd = end_nd - start_nd;\n";

    // replacementCode += "// Warm-up phase\n";
    // replacementCode += "int warmup_iters = 100;\n";
    // replacementCode += "for (int w = 0; w < warmup_iters; w++) {\n";
    // replacementCode += "    f(";
    // for (int i = 0; i < (int)varNames_nd.size(); i++) {
        // replacementCode += varNames_nd[i];
        // if (i < (int)varNames_nd.size() - 1) {
            // replacementCode += ", ";
        // }
    // }
    // replacementCode += ");\n";
    // replacementCode += "}\n\n";

    replacementCode += "// Timing measurement phase\n";
    // replacementCode += "auto start = std::chrono::high_resolution_clock::now();\n";
    // replacementCode += "int measure_iters = 1000;\n";
    // replacementCode += "for (int m = 0; m < measure_iters; m++) {\n";
    replacementCode += "f"+funcNameforCheck+"(";
    for (int i = 0; i < (int)varNames_nd.size(); i++) {
        replacementCode += varNames_nd[i];
        if (i < (int)varNames_nd.size() - 1) {
            replacementCode += ", ";
        }
    }
    replacementCode += ");\n";
    // replacementCode += "}\n\n";
    // replacementCode += "auto end = std::chrono::high_resolution_clock::now();\n";
    // replacementCode += "std::chrono::duration<double> elapsed = (end - start) / measure_iters;\n";
    

    // memcpy 처리
    // replacementCode += "auto start_post = std::chrono::high_resolution_clock::now();\n";
    for (const auto& [varName, type_reg] : memcpy_vars) {
        replacementCode += memcpy_vars_org[varName] + "= *reinterpret_cast<" + type_reg + "*>(" + varName +"_nd.ToDLPack()->dl_tensor.data);\n";
    }
    // replacementCode += "auto end_post = std::chrono::high_resolution_clock::now();\n";
    // replacementCode += "std::chrono::duration<double> elapsed_post = end_post - start_post;\n";
  
    // replacementCode += "std::cout << std::fixed << std::setprecision(12);\n";
    // replacementCode += "std::cout << \"전처리 시간  : \" << elapsed_nd.count() + elapsed_post.count() << \" 초\\n\";\n";
    // replacementCode += "std::cout << \"평균 실행 시간: \" << elapsed.count() << \" 초\\n\";\n\n";

    std::cout << replacementCode << std::endl;
    return replacementCode;
}



// ----------------------------
//  main
// ----------------------------
int main(int argc, const char **argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input.cpp>" << std::endl;
        return 1;
    }

    std::string inputCpp = argv[1];
    std::string compile_opt = argv[2];
    std::filesystem::path inputPath(inputCpp);
    inputPath = std::filesystem::absolute(inputPath);
    inputCpp = inputPath.string();

    // 1) 초기 컴파일 & TIR 추출
    if (!runInitialCompileAndExtractTIR(inputCpp, compile_opt)) {
        return 1;
    }
    std::cout << "Initial compile and TIR extraction completed successfully." << std::endl;

    // 2) TVM 빌드
    std::string libraryPath  = inputCpp + "_preprocess_tvm.so";
    std::string tirScriptPath= inputCpp + "_tvm_prim_func.txt";
    std::string finalSoPath  = inputCpp + "_tvm_prim_func.so";
    std::string paramsPath   = inputCpp + "_tvm_prim_func_params.txt";
    if (!runTVMBuild(libraryPath, tirScriptPath, finalSoPath, paramsPath)) {
        return 1;
    }
    std::cout << "TVM build completed successfully." << std::endl;
 
    // 3) LibTooling을 이용한 소스 코드 AST 변환
    llvm::cl::OptionCategory MyToolCategory("tvm-transform options");
    // auto OptionsParser = clang::tooling::CommonOptionsParser::create(argc, argv, MyToolCategory);
    const char *args[] = {argv[0], argv[1]};
    int new_argc = 2;
    auto OptionsParser = clang::tooling::CommonOptionsParser::create(new_argc, args, MyToolCategory);
    if (!OptionsParser) {
        llvm::errs() << "Error creating OptionsParser\n";
        return 1;
    }
    
    std::vector<std::string> FuncNames;
    std::ifstream file(paramsPath);
    std::string line;
    while (getline(file, line)) {
        std::istringstream iss(line);
        std::string varName, shapeStr, dtype, funcName;

        size_t funcNameStart = line.find("Function Name: ") + 15;
        size_t funcNameEnd = line.find(", Variable Name:");
        funcName = line.substr(funcNameStart, funcNameEnd - funcNameStart);
        // 중복된 함수명은 무시
        if (std::find(FuncNames.begin(), FuncNames.end(), funcName) != FuncNames.end()) {
            continue;
        }else{
            FuncNames.push_back(funcName);
        }
    }



    std::map<std::string, std::string> ReplacementCodeMap;
    std::map<std::string, std::string> LoadCodeMap;
    std::vector<std::string> AllVarNames;

    for (auto funcName : FuncNames) {
        std::vector<std::string> varNames;
        std::string replacementCode = MakeReplaceCode(finalSoPath, paramsPath, varNames, funcName);
        std::string loadCode = MakeLoadCode(finalSoPath, paramsPath, varNames, funcName);
        //중복제거해서 AllVarNames에 추가
        for (auto varName : varNames) {
            if (std::find(AllVarNames.begin(), AllVarNames.end(), varName) == AllVarNames.end()) {
                AllVarNames.push_back(varName);
            }
        } 
        std::cout<< "replacementCode: "<<replacementCode<<std::endl;
        ReplacementCodeMap[funcName] = replacementCode;
        LoadCodeMap[funcName] = loadCode;
    }


    clang::tooling::ClangTool Tool(OptionsParser->getCompilations(), OptionsParser->getSourcePathList());

    // 추가 컴파일 옵션 설정
    
    std::vector<std::string> ExtraArgs = {
        "-fopenmp",
        "-I/usr/lib/llvm-10/include/openmp",
        "-I/root/test/gemm/polybench-c-3.2/utilities",
        // 필요 시 더 추가
    };

    Tool.appendArgumentsAdjuster(
        clang::tooling::getInsertArgumentAdjuster(ExtraArgs, 
        clang::tooling::ArgumentInsertPosition::BEGIN)
    );

    // 실제 액션 실행
    OMPFrontendActionFactory actionFactory(ReplacementCodeMap, LoadCodeMap , AllVarNames, inputCpp);
    int Result = Tool.run(&actionFactory);
    if (Result != 0) {
        llvm::errs() << "Tool run failed\n";
        return Result;
    }

    std::cout << "Pipeline completed successfully." << std::endl;
    return 0;
}