
find_library(
	HSART_LIBRARIES
	NAMES hsa-runtime64 hsa-runtime-ext64
	)

find_path(
	HSART_INCLUDE_DIR
	NAMES hsa.h hsa_ext_finalize.h
	)

if (NOT HSART_LIBRARIES)
	message(SEND_ERROR "HSA Runtime not found")
elseif(NOT HSART_INCLUDE_DIR)
	message(SEND_ERROR "HSA Includes not found")
else()
	message(STATUS "Using HSA Runtime at ${HSART_LIBRARIES}")
	message(STATUS "Using HSA Includes at ${HSART_INCLUDE_DIR}")

	set(LLVM_LINK_COMPONENTS
	  HSAIL
	  X86
	  MCJIT
	  IRReader
	  )

	add_llvm_tool(hsailc
	  hsailc.cpp
	  )

	target_include_directories(hsailc PUBLIC ${HSART_INCLUDE_DIR})
	target_link_libraries(hsailc ${HSART_LIBRARIES})
endif()
