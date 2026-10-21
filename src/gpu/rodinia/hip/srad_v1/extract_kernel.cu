#include "hip/hip_runtime.h"
#include "../../common/kernel_timer.h"
// statistical kernel
__device__ void extract_body(	long d_Ne,
											fp *d_I){										// pointer to input image (DEVICE GLOBAL MEMORY)

	// indexes
	int bx = blockIdx.x;													// get current horizontal block index
	int tx = threadIdx.x;													// get current horizontal thread index
	int ei = (bx*NUMBER_THREADS)+tx;						// unique thread id, more threads than actual elements !!!

	// copy input to output & log uncompress
	if(ei<d_Ne){															// do only for the number of elements, omit extra threads

		d_I[ei] = exp(d_I[ei]/255);												// exponentiate input IMAGE and copy to output image

	}

}

// extract timed with s_memrealtime (common/kernel_timer.h)
__global__ void extract(unsigned long long *kt, 	long d_Ne,
											fp *d_I)
{
    KT_BEGIN();
    extract_body(d_Ne, d_I);
    KT_END(kt);
}
