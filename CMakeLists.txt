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
      transform.cu
      transform.h
      cg.cu
      cg.h
      cg_gpu.cu
      cg_gpu.h
      cg_gpu_sp.cu
      cg_gpu_sp.h 
      cg_gpu_mp.cu
      cg_gpu_mp.h
      mp_refinement.cu
      mp_refinement.h
      mp_refinement_cpu.cu
      mp_refinement_cpu.h
      cg_pm.cu
      cg_pm.h
      cg_gpu_pm.cu
      cg_gpu_pm.h
      pow_method.cu
      pow_method.h
      pow_method_gpu.cu
      pow_method_gpu.h
      dot_prod_test.cu
      dot_prod_test.h
      cg_test.cu
      cg_test.h
      pm_test.cu
      pm_test.h
	)

# The executable (has 'main')
cuda_add_executable(run-cg run-cg.cu ${src})
