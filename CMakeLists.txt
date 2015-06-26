set(HSART_HINTS /opt/hsa/lib ENV HSA_RUNTIME_PATH)
find_library(
	HSART_LIB
	PATHS ${HSART_HINTS}
	NAMES hsa-runtime64
	)
find_library(
	HSART_EXT_LIB
	PATHS ${HSART_HINTS}
	NAMES hsa-runtime-ext64
	)

find_path(
	HSART_INCLUDE_DIR
	NAMES hsa.h hsa_ext_finalize.h
	)

if (NOT HSART_LIB OR NOT HSART_EXT_LIB)
	message(WARNING "HSA Runtime not found")
elseif(NOT HSART_INCLUDE_DIR)
	message(WARNING "HSA Includes not found")
else()
	message(STATUS "Using HSA Runtime at ${HSART_LIB}")
	message(STATUS "Using HSA Runtime Extensions at ${HSART_EXT_LIB}")
	message(STATUS "Using HSA Includes at ${HSART_INCLUDE_DIR}")

	set(LLVM_LINK_COMPONENTS
	  HSAIL
	  X86
	  IRReader
	  Linker
	  )

	add_llvm_tool(hsailc
	  hsailc.cpp
	  )

	target_include_directories(hsailc PUBLIC ${HSART_INCLUDE_DIR})
	target_link_libraries(hsailc ${HSART_LIB})
	target_link_libraries(hsailc ${HSART_EXT_LIB})
endif()
