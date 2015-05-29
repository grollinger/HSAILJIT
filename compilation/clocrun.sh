#!/bin/bash
# try building using llvm 3.6.1 toolchain
CLC_PATH=/opt/amd/bin
LLVM_PATH=~/bin

FILE=${1-sum}
BUILDDIR=build_cloc

cd ${BUILDDIR}

rm ${FILE}*.*

cp ../src/${FILE}.cl .

cloc -ll -hsail -k -t . ${FILE}.cl

exit

${CLC_PATH}/clc2 -cl-std=CL2.0 -o ${FILE}-spir.bc ../src/${FILE}.cl

${LLVM_PATH}/llvm-dis -o ${FILE}-spir.ll ${FILE}-spir.bc

${LLVM_PATH}/llvm-link ~/hsa/cppamp/lib/builtins-hsail.opt.bc -o ${FILE}-lnk.bc ${FILE}-spir.bc

${LLVM_PATH}/llvm-dis -o ${FILE}-lnk.ll ${FILE}-lnk.bc

${LLVM_PATH}/opt -dce -O3 -o ${FILE}-opt.bc ${FILE}-lnk.bc

${LLVM_PATH}/llvm-dis -o ${FILE}-opt.ll ${FILE}-opt.bc

${LLVM_PATH}/llc -O2 -march=hsail64 -filetype=obj -o ${FILE}.brig ${FILE}-opt.bc

${LLVM_PATH}/HSAILasm -disassemble -o ${FILE}.hsail ${FILE}.brig
