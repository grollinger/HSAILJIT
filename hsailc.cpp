#include "llvm/ADT/Triple.h"
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
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/PluginLoader.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Target/TargetLibraryInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include <memory>

#include "hsa.h"
#include "hsa_ext_finalize.h"

using namespace llvm;

void run_kernel(hsa_ext_module_t);

static std::string KERNEL_NAME("&__OpenCL_copy_kernel");
static std::string FILE_PATH("./vector_copy.bc");

int main(int argc, char** argv) {
	llvm_shutdown_obj X;

	InitializeAllTargets();
	InitializeAllTargetMCs();
	InitializeAllAsmParsers();
	InitializeAllAsmPrinters();

	std::string HSATargetTriple("hsail64-unknown-unknown");

	LLVMContext& Context = getGlobalContext();
    SMDiagnostic Err;
    std::unique_ptr<Module> M;
    Triple TheTriple;

	PassRegistry* Registry = PassRegistry::getPassRegistry();
	initializeCore(*Registry);
    initializeCodeGen(*Registry);
    initializeLoopStrengthReducePass(*Registry);
    initializeLowerIntrinsicsPass(*Registry);
    initializeUnreachableBlockElimPass(*Registry);

    M = parseIRFile(FILE_PATH, Err, Context);
    if (!M) {
      Err.print(argv[0], errs());
      return 1;
    }

	M->setTargetTriple(Triple::normalize(HSATargetTriple));
	TheTriple = Triple(M->getTargetTriple());


	// Get the target specific parser.
	std::string Error;
	const Target *TheTarget = TargetRegistry::lookupTarget(MArch, TheTriple,
			Error);
	if (!TheTarget) {
		errs() << argv[0] << ": " << Error;
		return 1;
	}

	CodeGenOpt::Level OLvl = CodeGenOpt::Default; // O2
	TargetOptions Options; // Defaults

	std::unique_ptr<TargetMachine> Target(
			TheTarget->createTargetMachine(TheTriple.getTriple(), MCPU, "",
				Options, RelocModel, CMModel, OLvl));
	assert(Target && "Could not allocate target machine!");

	PassManager PM;

	// Add an appropriate TargetLibraryInfo pass for the module's triple.
	TargetLibraryInfo *TLI = new TargetLibraryInfo(TheTriple);
	PM.add(TLI);

	// Add the target data from the target machine, if it exists, or the module.
	if (const DataLayout *DL = Target->getSubtargetImpl()->getDataLayout())
		M->setDataLayout(DL);
	PM.add(new DataLayoutPass());

	SmallVector<char, 4096> ObjBufferSV;
	raw_svector_ostream OS(ObjBufferSV);
    formatted_raw_ostream FOS(OS);

    // Ask the target to add backend passes as necessary.
    if (Target->addPassesToEmitFile(PM, FOS, TargetMachine::CGFT_ObjectFile, false,
                                    nullptr, nullptr)) {
      errs() << argv[0] << ": target does not support generation of this"
             << " file type!\n";
      return 1;
    }

    PM.run(*M);

    FOS.flush();
	OS.flush();

	hsa_ext_module_t brig = (hsa_ext_module_t) ObjBufferSV.begin();

	run_kernel(brig);

	return 0;
}


#define check(msg, status) \
if (status != HSA_STATUS_SUCCESS) { \
    printf("%s failed.\n", #msg); \
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
    err = hsa_executable_get_symbol(executable, "", KERNEL_NAME.c_str(), agent, 0, &symbol);
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

    dispatch_packet->header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
    dispatch_packet->header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;
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
    __atomic_store_n((uint8_t*)(&dispatch_packet->header), (uint8_t)HSA_PACKET_TYPE_KERNEL_DISPATCH, __ATOMIC_RELEASE);

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
	printf("Signal returned: %i", value);

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
