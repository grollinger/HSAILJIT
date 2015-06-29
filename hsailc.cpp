#include "llvm/ADT/Triple.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/CodeGen/LinkAllAsmWriterComponents.h"
#include "llvm/CodeGen/LinkAllCodegenComponents.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/Pass.h"
#include "llvm/PassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/PluginLoader.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include "llvm/Transforms/IPO.h"
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include <memory>
#include "llvm/Support/raw_ostream.h"

// Link
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Linker/Linker.h"

#include "hsa.h"
#include "hsa_ext_finalize.h"

using namespace llvm;

namespace std {
	template< class T, class... Args >
	unique_ptr<T> make_unique( Args&&... args )
	{
		return unique_ptr<T>(new T(std::forward<Args>(args)...));
	}
}

// Returns the TargetMachine instance or zero if no triple is provided.
static TargetMachine* GetTargetMachine(Triple TheTriple) {
  std::string Error;
  auto MArch = "hsail64";

  const Target *TheTarget = TargetRegistry::lookupTarget(MArch, TheTriple,
                                                         Error);
  // Some modules don't specify a triple, and this is okay.
  if (!TheTarget) {
	  errs() << "Target Not Found \n";
    return nullptr;
  }

  // Package up features to be passed to target/subtarget
  std::string FeaturesStr;
  std::string MCPU;


  return TheTarget->createTargetMachine(TheTriple.getTriple(),
                                        MCPU, FeaturesStr,
                                        TargetOptions(),
                                        Reloc::Default, CodeModel::Default,
                                        CodeGenOpt::Aggressive);
}
static inline void addPass(PassManagerBase &PM, Pass *P) {
  // Add the pass to the pass manager...
  PM.add(P);

  // If we are verifying all of the intermediate steps, add the verifier...
  if (false) {
    PM.add(createVerifierPass());
    //PM.add(createDebugInfoVerifierPass());
  }
}

/// This routine adds optimization passes based on selected optimization level,
/// OptLevel.
///
/// OptLevel - Optimization Level
static void AddOptimizationPasses(PassManagerBase &MPM,FunctionPassManager &FPM,
                                  unsigned OptLevel, unsigned SizeLevel) {
  FPM.add(createVerifierPass());          // Verify that input is correct
  //MPM.add(createDebugInfoVerifierPass()); // Verify that debug info is correct

  PassManagerBuilder Builder;
  Builder.OptLevel = OptLevel;
  Builder.SizeLevel = SizeLevel;

//  if (DisableInline) {
//    // No inlining pass
//  } else if (OptLevel > 1) {
    Builder.Inliner = createFunctionInliningPass(OptLevel, SizeLevel);
//  } else {
//    Builder.Inliner = createAlwaysInlinerPass();
//  }
  //Builder.DisableUnitAtATime = !UnitAtATime;
  //Builder.DisableUnrollLoops = (DisableLoopUnrolling.getNumOccurrences() > 0) ?
  //                             DisableLoopUnrolling : OptLevel == 0;

  // When #pragma vectorize is on for SLP, do the same as above
  Builder.SLPVectorize = true;
      //DisableSLPVectorization ? false : OptLevel > 1 && SizeLevel < 2;

  Builder.populateFunctionPassManager(FPM);
  Builder.populateModulePassManager(MPM);
}

static void AddStandardLinkPasses(PassManagerBase &PM) {
  PassManagerBuilder Builder;
  Builder.VerifyInput = true;

	Builder.Inliner = createFunctionInliningPass();
  Builder.populateLTOPassManager(PM);
}

static void diagnosticHandler(const DiagnosticInfo &DI) {
  unsigned Severity = DI.getSeverity();
  switch (Severity) {
  case DS_Error:
    errs() << "ERROR: ";
    break;
  case DS_Warning:
    errs() << "WARNING: ";
    break;
  case DS_Remark:
  case DS_Note:
    llvm_unreachable("Only expecting warnings and errors");
  }

  DiagnosticPrinterRawOStream DP(errs());
  DI.print(DP);
  errs() << '\n';
}

int load_module_from_file(const char* file_name, hsa_ext_module_t* module) {
    int rc = -1;

    FILE *fp = fopen(file_name, "rb");

    rc = fseek(fp, 0, SEEK_END);

    size_t file_size = (size_t) (ftell(fp) * sizeof(char));

    rc = fseek(fp, 0, SEEK_SET);

    char* buf = (char*) malloc(file_size);

    memset(buf,0,file_size);

    size_t read_size = fread(buf,sizeof(char),file_size,fp);

    if(read_size != file_size) {
        free(buf);
    } else {
        rc = 0;
        *module = (hsa_ext_module_t) buf;
    }

    fclose(fp);

    return rc;
}

void run_kernel(hsa_ext_module_t);

static std::string KERNEL_NAME("&__OpenCL_copy_kernel");
static std::string BUILTINS_PATH("./builtins-hsail.bc");
static std::string FILE_PATH("./vector_copy.bc");

int main(int argc, char** argv) {
	llvm_shutdown_obj X;

	// Initialization
//	InitializeAllTargets();
//	InitializeAllTargetMCs();
//	InitializeAllAsmParsers();
//	InitializeAllAsmPrinters();
//
//	PassRegistry* Registry = PassRegistry::getPassRegistry();
//	initializeCore(*Registry);
//	initializeCodeGen(*Registry);
//	initializeLoopStrengthReducePass(*Registry);
//	initializeLowerIntrinsicsPass(*Registry);
//	initializeUnreachableBlockElimPass(*Registry);

	std::string HSATargetTriple("hsail64-unknown-unknown");

	LLVMContext& Context = getGlobalContext();
	SMDiagnostic Err;
	std::unique_ptr<Module> Builtins;
	std::unique_ptr<Module> M;
	Triple TheTriple;
	std::error_code EC;
    std::unique_ptr<tool_output_file> Out;

    // ${CLC_PATH}/clc2 -cl-std=CL2.0 -o ${FILE}-spir.bc ${FILE}.cl
	// -> input file

	// Load input file (SPIR Bitcode)
	M = parseIRFile(FILE_PATH, Err, Context);
	if (!M) {
		Err.print(argv[0], errs());
		return 1;
	}

    // ${LLVM_PATH}/llvm-link -o ${FILE}-lnk.bc ${FILE}-spir.bc ${BUILTINS_PATH}

	// Load builtins
	Builtins = parseIRFile(BUILTINS_PATH, Err, Context);
	if (!Builtins) {
		errs() << "Failed loading builtins\n";
		Err.print(argv[0], errs());
		return 1;
	}


	// Replace SPIR Target with HSAIL
	//M->setTargetTriple(Triple::normalize(HSATargetTriple));
	//TheTriple = Triple(M->getTargetTriple());
	//Builtins->setTargetTriple(TheTriple.getTriple());

	//////// OUTPUT builtins
    Out.reset(new tool_output_file("vector_copy-builtins.ll", EC, sys::fs::F_None));
    if (EC) {
      errs() << EC.message() << '\n';
      return 1;
    }
	Builtins->print(Out->os(), nullptr);
	Out->keep();

	//////// OUTPUT PRE-LINK
    Out.reset(new tool_output_file("vector_copy-spir.ll", EC, sys::fs::F_None));
    if (EC) {
      errs() << EC.message() << '\n';
      return 1;
    }
	M->print(Out->os(), nullptr);
	Out->keep();

	// Link in the builtins
	auto Composite = std::make_unique<Module>("llvm-link", Context);
	Composite->setDataLayout(M->getDataLayout());
	Linker L(Composite.get(), diagnosticHandler);

    if (L.linkInModule(M.release()))
      return 1;

    if (L.linkInModule(Builtins.release()))
      return 1;


	if (verifyModule(*Composite)) {
		errs() << argv[0] << ": linked module is broken!\n";
		return 1;
	}

	M = std::move(Composite);

	//////// OUTPUT POST-LINK
    Out.reset(new tool_output_file("vector_copy-link.ll", EC, sys::fs::F_None));
    if (EC) {
      errs() << EC.message() << '\n';
      return 1;
    }
	M->print(Out->os(), nullptr);
	Out->keep();
    // ${LLVM_PATH}/opt -O3 -o ${FILE}-opt.bc ${FILE}-lnk.bc


  InitializeAllTargets();
  InitializeAllTargetMCs();
  InitializeAllAsmPrinters();

  // Initialize passes
  PassRegistry &Registry = *PassRegistry::getPassRegistry();
  initializeCore(Registry);
  initializeScalarOpts(Registry);
  //initializeObjCARCOpts(Registry);
  initializeVectorization(Registry);
  initializeIPO(Registry);
  initializeAnalysis(Registry);
  initializeIPA(Registry);
  initializeTransformUtils(Registry);
  initializeInstCombine(Registry);
  //initializeInstrumentation(Registry);
  initializeTarget(Registry);
  // For codegen passes, only passes that do IR to IR transformation are
  // supported.
  initializeCodeGenPreparePass(Registry);
  initializeAtomicExpandPass(Registry);
  initializeRewriteSymbolsPass(Registry);

  if (!M) {
    Err.print(argv[0], errs());
    return 1;
  }

  // Create a PassManager to hold and optimize the collection of passes we are
  // about to build.
  //
  PassManager Passes;

  // Add an appropriate TargetLibraryInfo pass for the module's triple.
  auto TLI = new TargetLibraryInfoWrapperPass(Triple(M->getTargetTriple()));
  Passes.add(TLI);

  Triple ModuleTriple(M->getTargetTriple());
  TargetMachine *Machine = nullptr;
  if (ModuleTriple.getArch())
    Machine = GetTargetMachine(Triple(ModuleTriple));
  std::unique_ptr<TargetMachine> TM(Machine);

  // Add internal analysis passes from the target machine.
  Passes.add(createTargetTransformInfoWrapperPass(TM ? TM->getTargetIRAnalysis()
                                                     : TargetIRAnalysis()));

  std::unique_ptr<FunctionPassManager> FPasses;
  FPasses.reset(new FunctionPassManager(M.get()));
  FPasses->add(createTargetTransformInfoWrapperPass(
	TM ? TM->getTargetIRAnalysis() : TargetIRAnalysis()));

  AddStandardLinkPasses(Passes);


  AddOptimizationPasses(Passes, *FPasses, 3, 0);

	FPasses->doInitialization();
	for (Function &F : *M)
	  FPasses->run(F);
	FPasses->doFinalization();

  // Check that the module is well formed on completion of optimization
    Passes.add(createVerifierPass());
    //Passes.add(createDebugInfoVerifierPass());

	// output code to stderr
      Passes.add(createPrintModulePass(errs()));
  // Now that we have all of the passes ready, run them.
  Passes.run(*M);


	//////// OUTPUT POST-OPT
    Out.reset(new tool_output_file("vector_copy-opt.ll", EC, sys::fs::F_None));
    if (EC) {
      errs() << EC.message() << '\n';
      return 1;
    }
	M->print(Out->os(), nullptr);
	Out->keep();

	// ${LLVM_PATH}/llc -O2 -march=hsail64 -filetype=obj -o ${FILE}.brig ${FILE}-opt.bc

	PassManager* PM = new PassManager();

    // Eliminate all unused functions
    PM->add(createGlobalOptimizerPass());
    PM->add(createStripDeadPrototypesPass());

	// Inline all functions with always_inline attribute
    PM->add(createAlwaysInlinerPass());

    auto FPM = std::unique_ptr<FunctionPassManager>( new FunctionPassManager(nullptr /*M*/));

    // Enqueue standard optimizations
    PassManagerBuilder PMB;
    PMB.OptLevel = CodeGenOpt::Aggressive;
    PMB.populateFunctionPassManager(*FPM);


	// OUTPUT ---------------------------------------------
	SmallVector<char, 4096> ObjBufferSV;
	raw_svector_ostream OS(ObjBufferSV);

	// Output format
	auto FT = (0) ? TargetMachine::CGFT_AssemblyFile : TargetMachine::CGFT_ObjectFile;

	// Ask the target to add backend passes as necessary.
	if (TM->addPassesToEmitFile(*PM, OS, FT, false,
				nullptr, nullptr)) {
		errs() << argv[0] << ": target does not support generation of this"
			<< " file type!\n";
		return 1;
	}

	PM->run(*M);

	OS.flush();


	//////// OUTPUT
    Out.reset(new tool_output_file(
				(FT == TargetMachine::CGFT_AssemblyFile)
				 ? "vector_copy.hsail"
				 : "vector_copy.brig",
				 EC, sys::fs::F_None));
    if (EC) {
      errs() << EC.message() << '\n';
      return 1;
    }
	Out->os().write(ObjBufferSV.data(), ObjBufferSV.size());
	Out->os().flush();
	Out->keep();

	hsa_ext_module_t brig;

	if(FT == TargetMachine::CGFT_AssemblyFile)
	{
		// HSAIL output
		errs() << OS.str();
		// Use brig file
		load_module_from_file("vector_copy.brig", &brig);
	}
	else
	{
		// Continue with in-memory BRIG
		brig = (hsa_ext_module_t) ObjBufferSV.data();
	}



	run_kernel(brig);

	return 0;
}


#define check(msg, status) \
	if (status != HSA_STATUS_SUCCESS) { \
		const char* _stat;\
		hsa_status_string(status, &_stat);\
		printf("%s failed.\n%s", #msg, _stat); \
		\
		exit(1); \
	} else { \
		printf("%s succeeded.\n", #msg); \
	}

/*
 * Determines if the given agent is of type HSA_DEVICE_TYPE_GPU
 * and sets the value of data to the agent handle if it is.
 */
static hsa_status_t get_gpu_agent(hsa_agent_t agent, void *data) {
	hsa_status_t status;
	hsa_device_type_t device_type;
	status = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);
	if (HSA_STATUS_SUCCESS == status && HSA_DEVICE_TYPE_GPU == device_type) {
		hsa_agent_t* ret = (hsa_agent_t*)data;
		*ret = agent;
		return HSA_STATUS_INFO_BREAK;
	}
	return HSA_STATUS_SUCCESS;
}

/*
 * Determines if a memory region can be used for kernarg
 * allocations.
 */
static hsa_status_t get_kernarg_memory_region(hsa_region_t region, void* data) {
	hsa_region_segment_t segment;
	hsa_region_get_info(region, HSA_REGION_INFO_SEGMENT, &segment);
	if (HSA_REGION_SEGMENT_GLOBAL != segment) {
		return HSA_STATUS_SUCCESS;
	}

	hsa_region_global_flag_t flags;
	hsa_region_get_info(region, HSA_REGION_INFO_GLOBAL_FLAGS, &flags);
	if (flags & HSA_REGION_GLOBAL_FLAG_KERNARG) {
		hsa_region_t* ret = (hsa_region_t*) data;
		*ret = region;
		return HSA_STATUS_INFO_BREAK;
	}

	return HSA_STATUS_SUCCESS;
}

void run_kernel(hsa_ext_module_t module) {
	hsa_status_t err;

	err = hsa_init();
	check(Initializing the hsa runtime, err);

	/*
	 * Iterate over the agents and pick the gpu agent using
	 * the get_gpu_agent callback.
	 */
	hsa_agent_t agent;
	err = hsa_iterate_agents(get_gpu_agent, &agent);
	if(err == HSA_STATUS_INFO_BREAK) { err = HSA_STATUS_SUCCESS; }
	check(Getting a gpu agent, err);

	/*
	 * Query the name of the agent.
	 */
	char name[64] = { 0 };
	err = hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, name);
	check(Querying the agent name, err);
	printf("The agent name is %s.\n", name);

	/*
	 * Query the maximum size of the queue.
	 */
	uint32_t queue_size = 0;
	err = hsa_agent_get_info(agent, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_size);
	check(Querying the agent maximum queue size, err);
	printf("The maximum queue size is %u.\n", (unsigned int) queue_size);

	/*
	 * Create a queue using the maximum size.
	 */
	hsa_queue_t* queue;
	err = hsa_queue_create(agent, queue_size, HSA_QUEUE_TYPE_SINGLE, NULL, NULL, UINT32_MAX, UINT32_MAX, &queue);
	check(Creating the queue, err);

	/*
	 * Create hsa program.
	 */
	hsa_ext_program_t program;
	memset(&program,0,sizeof(hsa_ext_program_t));
	err = hsa_ext_program_create(HSA_MACHINE_MODEL_LARGE, HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, NULL, &program);
	check(Create the program, err);

	/*
	 * Add the BRIG module to hsa program.
	 */
	err = hsa_ext_program_add_module(program, module);
	check(Adding the brig module to the program, err);

	/*
	 * Determine the agents ISA.
	 */
	hsa_isa_t isa;
	err = hsa_agent_get_info(agent, HSA_AGENT_INFO_ISA, &isa);
	check(Query the agents isa, err);

	/*
	 * Finalize the program and extract the code object.
	 */
	hsa_ext_control_directives_t control_directives;
	memset(&control_directives, 0, sizeof(hsa_ext_control_directives_t));
	hsa_code_object_t code_object;
	err = hsa_ext_program_finalize(program, isa, 0, control_directives, "", HSA_CODE_OBJECT_TYPE_PROGRAM, &code_object);
	check(Finalizing the program, err);

	/*
	 * Destroy the program, it is no longer needed.
	 */
	err=hsa_ext_program_destroy(program);
	check(Destroying the program, err);

	/*
	 * Create the empty executable.
	 */
	hsa_executable_t executable;
	err = hsa_executable_create(HSA_PROFILE_FULL, HSA_EXECUTABLE_STATE_UNFROZEN, "", &executable);
	check(Create the executable, err);

	/*
	 * Load the code object.
	 */
	err = hsa_executable_load_code_object(executable, agent, code_object, "");
	check(Loading the code object, err);

	/*
	 * Freeze the executable; it can now be queried for symbols.
	 */
	err = hsa_executable_freeze(executable, "");
	check(Freeze the executable, err);

	/*
	 * Extract the symbol from the executable.
	 */
	hsa_executable_symbol_t symbol;
	err = hsa_executable_get_symbol(executable, NULL, KERNEL_NAME.c_str(), agent, 0, &symbol);
	check(Extract the symbol from the executable, err);

	/*
	 * Extract dispatch information from the symbol
	 */
	uint64_t kernel_object;
	uint32_t kernarg_segment_size;
	uint32_t group_segment_size;
	uint32_t private_segment_size;
	err = hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernel_object);
	check(Extracting the symbol from the executable, err);
	err = hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, &kernarg_segment_size);
	check(Extracting the kernarg segment size from the executable, err);
	err = hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, &group_segment_size);
	check(Extracting the group segment size from the executable, err);
	err = hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE, &private_segment_size);
	check(Extracting the private segment from the executable, err);

	/*
	 * Create a signal to wait for the dispatch to finish.
	 */
	hsa_signal_t signal;
	err=hsa_signal_create(1, 0, NULL, &signal);
	check(Creating a HSA signal, err);

	/*
	 * Allocate and initialize the kernel arguments and data.
	 */
	char* in=(char*)malloc(1024*1024*4);
	memset(in, 1, 1024*1024*4);
	err=hsa_memory_register(in, 1024*1024*4);
	check(Registering argument memory for input parameter, err);

	char* out=(char*)malloc(1024*1024*4);
	memset(out, 0, 1024*1024*4);
	err=hsa_memory_register(out, 1024*1024*4);
	check(Registering argument memory for output parameter, err);

	struct __attribute__ ((aligned(16))) args_t {
		void* in;
		void* out;
	} args;

	args.in=in;
	args.out=out;

	/*
	 * Find a memory region that supports kernel arguments.
	 */
	hsa_region_t kernarg_region;
	kernarg_region.handle=(uint64_t)-1;
	hsa_agent_iterate_regions(agent, get_kernarg_memory_region, &kernarg_region);
	err = (kernarg_region.handle == (uint64_t)-1) ? HSA_STATUS_ERROR : HSA_STATUS_SUCCESS;
	check(Finding a kernarg memory region, err);
	void* kernarg_address = NULL;

	/*
	 * Allocate the kernel argument buffer from the correct region.
	 */
	err = hsa_memory_allocate(kernarg_region, kernarg_segment_size, &kernarg_address);
	check(Allocating kernel argument memory buffer, err);
	memcpy(kernarg_address, &args, sizeof(args));

	/*
	 * Obtain the current queue write index.
	 */
	uint64_t index = hsa_queue_load_write_index_relaxed(queue);

	/*
	 * Write the aql packet at the calculated queue index address.
	 */
	const uint32_t queueMask = queue->size - 1;
	hsa_kernel_dispatch_packet_t* dispatch_packet = &(((hsa_kernel_dispatch_packet_t*)(queue->base_address))[index&queueMask]);

	dispatch_packet->setup  |= 1 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
	dispatch_packet->workgroup_size_x = (uint16_t)256;
	dispatch_packet->workgroup_size_y = (uint16_t)1;
	dispatch_packet->workgroup_size_z = (uint16_t)1;
	dispatch_packet->grid_size_x = (uint32_t) (1024*1024);
	dispatch_packet->grid_size_y = 1;
	dispatch_packet->grid_size_z = 1;
	dispatch_packet->completion_signal = signal;
	dispatch_packet->kernel_object = kernel_object;
	dispatch_packet->kernarg_address = (void*) kernarg_address;
	dispatch_packet->private_segment_size = private_segment_size;
	dispatch_packet->group_segment_size = group_segment_size;

    uint16_t header = 0;
    header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
    header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;
    header |= HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;

	__atomic_store_n((uint16_t*)(&dispatch_packet->header), header, __ATOMIC_RELEASE);

	/*
	 * Increment the write index and ring the doorbell to dispatch the kernel.
	 */
	hsa_queue_store_write_index_relaxed(queue, index+1);
	hsa_signal_store_relaxed(queue->doorbell_signal, index);
	check(Dispatching the kernel, err);

	/*
	 * Wait on the dispatch completion signal until the kernel is finished.
	 */
	hsa_signal_value_t value = hsa_signal_wait_acquire(signal, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

	/*
	 * Validate the data in the output buffer.
	 */
	int valid=1;
	int fail_index=0;
	for(int i=0; i<1024*1024; i++) {
		if(out[i]!=in[i]) {
			fail_index=i;
			valid=0;
			break;
		}
	}

	if(valid) {
		printf("Passed validation.\n");
	} else {
		printf("VALIDATION FAILED!\nBad index: %d\n", fail_index);
	}

	/*
	 * Cleanup all allocated resources.
	 */
    err = hsa_memory_free(kernarg_address);
    check(Freeing kernel argument memory buffer, err);

    err=hsa_signal_destroy(signal);
    check(Destroying the signal, err);

    err=hsa_executable_destroy(executable);
    check(Destroying the executable, err);

    err=hsa_code_object_destroy(code_object);
    check(Destroying the code object, err);

    err=hsa_queue_destroy(queue);
    check(Destroying the queue, err);

    err=hsa_shut_down();
    check(Shutting down the runtime, err);

    free(in);
    free(out);
}
