/*
 * Per-launch kernel runtime from s_memrealtime (same scheme as gem5-resources'
 * pannotia/pennant/heterosync/lulesh copies).
 *
 * Every wave reads s_memrealtime at the start and end of the kernel, and
 * lane 0 folds them into its launch's slot with atomicMax, so a slot ends up
 * holding the first wave start (stored inverted, so that a zeroed slot is
 * empty) and the last wave end of that launch. To avoid serializing thousands
 * of atomics on one address, each launch has KT_SUBSLOTS slots on separate
 * cache lines, picked by wave number, and report() combines them.
 * s_memrealtime is a constant-rate clock shared by all CUs (100 MHz on
 * MI300X), unlike s_memtime, whose per-CU counters cannot be compared across
 * waves. The span excludes the dispatch setup and end-of-kernel cache flushes
 * that GRBM_GUI_ACTIVE and profiler timestamps include.
 *
 * Each kernel's body is a __device__ function called from a timed __global__
 * wrapper that takes the slot as an extra first argument:
 *   __global__ void k(unsigned long long *kt, ...) { KT_BEGIN(); k_body(...); KT_END(kt); }
 *   k<<<grid, block>>>(kt.next("k"), ...);
 * The timer allocates its slots at the first launch (on the current device)
 * and prints its report at exit; call kt.report() earlier if the program
 * resets the device before exiting. Set KT_PER_LAUNCH=1 to also list every
 * launch's time in launch order.
 */
#ifndef KERNEL_TIMER_H
#define KERNEL_TIMER_H

#include "hip/hip_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <map>
#include <string>
#include <vector>

#ifndef KT_MAX_LAUNCHES
#define KT_MAX_LAUNCHES 8192         // launches timed; later ones are not
#endif
#ifndef KT_SUBSLOTS
#define KT_SUBSLOTS 32               // slots per launch, one per cache line
#endif
#define KT_SLOT_WORDS 16             // 128 B between slots
#define KT_LAUNCH_WORDS (KT_SUBSLOTS * KT_SLOT_WORDS)

#define KT_BEGIN() \
    unsigned long long kt_start = __builtin_amdgcn_s_memrealtime()

// wait for this wave's memory accesses so the end time covers them
#define KT_END(kt) do { \
    __asm__ volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory"); \
    unsigned long long kt_end = __builtin_amdgcn_s_memrealtime(); \
    if ((kt) && __lane_id() == 0) { \
        unsigned kt_wave = ((blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + \
            blockIdx.x) * ((blockDim.x * blockDim.y * blockDim.z + 63) / 64) + \
            ((threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x) / 64; \
        unsigned long long *kt_slot = (kt) + (kt_wave % KT_SUBSLOTS) * KT_SLOT_WORDS; \
        atomicMax(&kt_slot[0], ~kt_start); \
        atomicMax(&kt_slot[1], kt_end); \
    } \
} while (0)

struct KernelTimer {
    unsigned long long *slots_d = nullptr;  // per launch: KT_SUBSLOTS x {~start, end}
    std::vector<std::string> names;         // kernel name of each launch

    void init();

    // slot for the next launch of kernel 'name' (nullptr once full)
    unsigned long long *next(const char *name) {
        if (!slots_d)
            init();
        if (names.size() >= KT_MAX_LAUNCHES)
            return nullptr;
        names.push_back(name);
        return slots_d + (names.size() - 1) * KT_LAUNCH_WORDS;
    }

    // print per-kernel totals and free the slots; later calls do nothing
    void report() {
        size_t n = names.size();
        if (!slots_d)
            return;
        std::vector<unsigned long long> h(n * KT_LAUNCH_WORDS);
        (void)hipDeviceSynchronize();
        if (n && hipMemcpy(h.data(), slots_d, h.size() * sizeof(h[0]),
                           hipMemcpyDeviceToHost) != hipSuccess) {
            fprintf(stderr, "kernel timer: copy back failed\n");
            return;
        }
        int rate_khz = 0;  // s_memrealtime frequency
        (void)hipDeviceGetAttribute(&rate_khz, hipDeviceAttributeWallClockRate, 0);
        if (rate_khz <= 0) {
            rate_khz = 100000;
            printf("kernel timer: wall clock rate unknown, assuming 100 MHz\n");
        }
        struct Stat { int launches = 0; double total = 0, min = 1e300, max = 0; };
        std::map<std::string, Stat> stats;
        double total = 0;
        // KT_PER_LAUNCH=1 also lists every launch, in launch order
        const char *per_launch = getenv("KT_PER_LAUNCH");
        bool list = per_launch && per_launch[0] && per_launch[0] != '0';
        if (list)
            printf("\nkernel timer per launch (launch #, per-kernel #, kernel, us):\n");
        std::map<std::string, int> nth;
        for (size_t i = 0; i < n; i++) {
            unsigned long long inv_start = 0, end = 0;
            for (int j = 0; j < KT_SUBSLOTS; j++) {
                const unsigned long long *slot =
                    &h[i * KT_LAUNCH_WORDS + j * KT_SLOT_WORDS];
                if (slot[0] > inv_start) inv_start = slot[0];
                if (slot[1] > end) end = slot[1];
            }
            if (inv_start == 0) continue;  // launch had no waves
            double us = (end - ~inv_start) * 1000.0 / rate_khz;
            if (list)
                printf("  %6zu %6d  %-28s %10.3f\n", i + 1, ++nth[names[i]], names[i].c_str(), us);
            Stat &s = stats[names[i]];
            s.launches++;
            s.total += us;
            if (us < s.min) s.min = us;
            if (us > s.max) s.max = us;
            total += us;
        }
        // own line even if the program left its last line unterminated
        printf("\nkernel timer (s_memrealtime, first wave start to last wave end):\n");
        for (auto &kv : stats)
            printf("  %-28s launches %6d  total %12.3f us  avg %10.3f us  "
                   "min %10.3f us  max %10.3f us\n", kv.first.c_str(),
                   kv.second.launches, kv.second.total,
                   kv.second.total / kv.second.launches, kv.second.min,
                   kv.second.max);
        printf("  %-28s launches %6zu  total %12.3f us\n", "all kernels", n, total);
        if (n == KT_MAX_LAUNCHES)
            printf("kernel timer: launch limit %d reached; later launches not timed\n",
                   KT_MAX_LAUNCHES);
        fflush(stdout);
        (void)hipFree(slots_d);
        slots_d = nullptr;
        names.clear();
    }
};

// one timer per program, shared by all its source files (a function-local
// static rather than a C++17 inline variable, for older toolchains)
inline KernelTimer &kt_instance() { static KernelTimer t; return t; }
static KernelTimer &kt = kt_instance();

inline void KernelTimer::init() {
    size_t bytes = (size_t)KT_MAX_LAUNCHES * KT_LAUNCH_WORDS * sizeof(*slots_d);
    if (hipMalloc(&slots_d, bytes) != hipSuccess ||
        hipMemset(slots_d, 0, bytes) != hipSuccess ||
        hipDeviceSynchronize() != hipSuccess) {
        fprintf(stderr, "kernel timer: allocation failed\n");
        exit(1);
    }
    atexit([] { kt.report(); });
}

#endif
