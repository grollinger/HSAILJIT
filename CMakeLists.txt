set(LLVM_LINK_COMPONENTS
  HSAIL
  X86
  MCJIT
  IRReader
  )

add_llvm_tool(hsailjit
  hsailjit.cpp
  )
