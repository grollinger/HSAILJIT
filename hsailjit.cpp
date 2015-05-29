#include "llvm/ExecutionEngine/MCJIT.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/ADT/Triple.h"
#include <llvm/ExecutionEngine/JITEventListener.h>
#include "llvm/Object/ObjectFile.h"
#include <memory>
#include <iostream>


using namespace llvm;

class BRIGEventListener : public JITEventListener
{
public:
	const object::ObjectFile* LastObject;

	BRIGEventListener()
		: LastObject(nullptr) {
		}

	virtual void NotifyObjectEmitted(const object::ObjectFile &Obj, const RuntimeDyld::LoadedObjectInfo &L) override {
		LastObject = &Obj;
	}

	virtual void NotifyFreeingObject(const object::ObjectFile &Obj) override {}
};

int main() {
	InitializeAllTargets();
	InitializeAllTargetMCs();
	InitializeAllAsmParsers();
	InitializeAllAsmPrinters();

	LLVMContext Context;
	std::string ErrorMessage;
    auto Buffer = MemoryBuffer::getFile("./sum-manual.bc");
	if (!Buffer) {
		errs() << "sum.bc not found\n";
		return -1;
	}

	auto M = parseBitcodeFile(Buffer.get()->getMemBufferRef(), Context);
	if (!M) {
		errs() << "Failed to load BitCode" << "\n";
		return -1;
	}
	EngineBuilder EB(std::unique_ptr<Module>(M.get()));

	StringRef empty;
	SmallVector< std::string , 4> emptySequence;
	std::string ERR;

	auto T = Triple("hsail64", "", "");

	auto TM = EB
		.setEngineKind(EngineKind::JIT)
		.setErrorStr(&ERR)
		.selectTarget(
			T,
			empty,
			empty,
			emptySequence
		);

	auto EE = EB
		.create(TM);

	if (!EE) {
		errs() << "Failed to create ExecutionEngine" << "\n";
		errs() << ERR << "\n";
		return -2;
	}


	BRIGEventListener Listener;

	EE->RegisterJITEventListener(&Listener);

	EE->generateCodeForModule(M.get());

	//Sum =  EE->getFunctionAddress(std::string("__OpenCL_sum_kernel"));

	auto Obj = Listener.LastObject;

	auto FMT = Obj->getFileFormatName();
	std::cerr << "File Form: " << FMT.str();


	llvm_shutdown();
	return 0;
}
