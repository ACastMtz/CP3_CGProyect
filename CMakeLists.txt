cmake_minimum_required(VERSION 2.8)
find_package(CUDA)

# Pass options to NVCC
set(CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 -arch sm_30
    )

# The source files (no 'main')
set(src
      common.h
      global.h
      geometry.cu
      geometry.h
      linalg.cu
      linalg.h
      cg.cu
      cg.h
   )

# The executable (has 'main')
cuda_add_executable(run-cg run-cg.cu ${src})
