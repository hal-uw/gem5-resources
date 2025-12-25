#ifndef __EVENT_TIMER_H__
#define __EVENT_TIMER_H__
#include "hip/hip_runtime.h"
#ifdef GEM5_FUSION
#include <util/m5/src/m5_mmap.h>
#include <gem5/m5ops.h>
#endif

struct TimingEvent{
    std::string name;
    hipEvent_t start;
    hipEvent_t end;
    float elapsedTime;
};


void start_timer(TimingEvent* event){
    hipError_t hipErr;
    hipErr = hipEventCreate(&(event->start));
    hipErr = hipEventCreate(&(event->end));
    

    hipEventRecord(event->start, 0);
}

void end_timer(TimingEvent* event){
    hipEventRecord((event->end), 0);
    hipEventSynchronize((event->end));
    hipEventElapsedTime(&(event->elapsedTime), (event->start), (event->end));
}

void free_timer(TimingEvent* event){
    hipError_t hipErr;
    hipErr = hipEventDestroy(event->start);
    hipErr = hipEventDestroy(event->end);
}

#endif // __EVENT_TIMER_H__