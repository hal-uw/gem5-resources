#include "hip/hip_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//#include <sys/time.h>
//#include <omp.h>
#include "../graph_parser/util.h"
#include "kernel.h"
#include <unistd.h>
#include <sys/mman.h>
#include <fstream>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string>

#ifndef GEM5_FUSION
#include <filesystem>
#else
#include <gem5/m5ops.h>
#endif

struct TimingEvent{
    std::string name;
    hipEvent_t start;
    hipEvent_t end;
    float elapsedTime;
    std::string DataFilePath;
    bool fileCreated;
};

void clear_timer(TimingEvent* event){
    hipError_t hipErr;
    hipErr = hipEventDestroy(event->start);
    hipErr = hipEventDestroy(event->end);
}

void create_data_file(TimingEvent* event) {
    if (event->fileCreated){
        printf("Timing File already created for this event.\n");
        return;
    }
    else {
        printf("Timing Data File is being created.\n");
    }
    std::string save_dir = "TimingDataOut";
    if (mkdir(save_dir.c_str(), 0777) == -1){
        printf("Directory Already exists, skipping.\n");
    }
    else{
        printf("Directory created.\n");
    }
    
    std::string contents = "Kernel,Real GPU(ms)\n";
    std::string filePath =  save_dir +"/" + event->name + ".csv";

    std::ofstream fout(filePath, std::ios::out | std::ios::binary);

    if (!fout.is_open()) {
        printf("Failed to create file");
        return;
    }



    fout.write(contents.c_str(), contents.size());
    fout.close();
    event->fileCreated = true;
    event->DataFilePath = filePath;
}   

void _start_timer(TimingEvent* event){
    create_data_file(event);
    hipError_t hipErr;
    hipErr = hipEventCreate(&(event->start));
    hipErr = hipEventCreate(&(event->end));
    

    hipEventRecord(event->start, 0);
}

void start_timer(TimingEvent* event, bool dumpStats){
    #ifndef GEM5_FUSION
        _start_timer(event);
    #else
        _start_timer(event);
        if (dumpStats) {
            m5_dump_reset_stats(0, 0);
        }
    #endif
}

void _end_timer(TimingEvent* event, const std::string& kernelName){
    hipEventRecord((event->end), 0);
    hipEventSynchronize((event->end));
    hipEventElapsedTime(&(event->elapsedTime), (event->start), (event->end));
    std::ofstream fout(event->DataFilePath, std::ios::app);
    fout << kernelName << "," << event->elapsedTime << "\n";
}

void stop_timer(TimingEvent* event, const std::string& kernelName){
    #ifndef GEM5_FUSION
        _end_timer(event, kernelName);
        clear_timer(event);
    #else
	_end_timer(event, kernelName);
        clear_timer(event);
        m5_dump_reset_stats(0, 0);
    #endif
}

std::string strip_before_last_slash(const std::string& s) {
    size_t pos = s.find_last_of('/');
    if (pos == std::string::npos)
        return s;          // no slash → return whole string
    return s.substr(pos + 1);
}

