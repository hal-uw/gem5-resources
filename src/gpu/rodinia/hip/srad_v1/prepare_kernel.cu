#include "hip/hip_runtime.h"
#include "../../common/kernel_timer.h"
// statistical kernel
__device__ void prepare_body(	long d_Ne,
											fp *d_I,										// pointer to output image (DEVICE GLOBAL MEMORY)
											fp *d_sums,									// pointer to input image (DEVICE GLOBAL MEMORY)
											fp *d_sums2){

	// indexes
	int bx = blockIdx.x;													// get current horizontal block index
	int tx = threadIdx.x;													// get current horizontal thread index
	int ei = (bx*NUMBER_THREADS)+tx;										// unique thread id, more threads than actual elements !!!

	// copy input to output & log uncompress
	if(ei<d_Ne){															// do only for the number of elements, omit extra threads

		d_sums[ei] = d_I[ei];
		d_sums2[ei] = d_I[ei]*d_I[ei];

	}

}

// prepare timed with s_memrealtime (common/kernel_timer.h)
__global__ void prepare(unsigned long long *kt, 	long d_Ne,
											fp *d_I,										// pointer to output image (DEVICE GLOBAL MEMORY)
											fp *d_sums,									// pointer to input image (DEVICE GLOBAL MEMORY)
											fp *d_sums2)
{
    KT_BEGIN();
    prepare_body(d_Ne, d_I, d_sums, d_sums2);
    KT_END(kt);
}
