#include "llvm/ExecutionEngine/MCJIT.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FileSystem.h"

using namespace llvm;

int main() {
	InitializeNativeTarget();
	InitializeNativeTargetAsmPrinter();
	InitializeNativeTargetAsmParser();

	LLVMContext Context;
	std::string ErrorMessage;
    auto Buffer = MemoryBuffer::getFile("./sum.bc");
	if (!Buffer) {
		errs() << "sum.bc not found\n";
		return -1;
	}

	auto M = parseBitcodeFile(Buffer.get()->getMemBufferRef(), Context);
	if (!M) {
		errs() << "Failed to load BitCode" << "\n";
		return -1;
	}
	auto EE = EngineBuilder(std::unique_ptr<Module>(M.get()))
		.create();

	int (*Sum)(int, int) = NULL;
	Sum = (int (*)(int, int)) EE->getFunctionAddress(std::string("sum"));

	int res = Sum(4,5);
	outs() << "Sum result: " << res << "\n";
	res = Sum(res, 6);
	outs() << "Sum result: " << res << "\n";

	llvm_shutdown();
	return 0;
}
