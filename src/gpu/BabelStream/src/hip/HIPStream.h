
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#pragma once

#include <iostream>
#include <stdexcept>
#include <sstream>

#include "Stream.h"

#define IMPLEMENTATION_STRING "HIP"
#define DOT_READ_DWORDS_PER_LANE 4


template <class T>
class HIPStream : public Stream<T>
{
  // Make sure that either:
  //    DOT_READ_DWORDS_PER_LANE is less than sizeof(T), in which case we default to 1 element
  //    or
  //    DOT_READ_DWORDS_PER_LANE is divisible by sizeof(T)
  static_assert((DOT_READ_DWORDS_PER_LANE * sizeof(unsigned int) < sizeof(T)) ||
                (DOT_READ_DWORDS_PER_LANE * sizeof(unsigned int) % sizeof(T) == 0),
                "DOT_READ_DWORDS_PER_LANE not divisible by sizeof(element_type)");

  // Take into account the datatype size
  // That is, for 4 DOT_READ_DWORDS_PER_LANE, this is 2 FP64 elements
  // and 4 FP32 elements
  static constexpr unsigned int dot_elements_per_lane{
    (DOT_READ_DWORDS_PER_LANE * sizeof(unsigned int)) < sizeof(T) ? 1 : (
     DOT_READ_DWORDS_PER_LANE * sizeof(unsigned int) / sizeof(T))};

  protected:
    // Size of arrays
    intptr_t array_size;
    intptr_t dot_num_blocks;

    // Partial sums for dot kernel: device array and its host copy
    T *d_sums;
    T *sums;

    // Device side pointers to arrays
    T *d_a;
    T *d_b;
    T *d_c;

    // s_memrealtime kernel timer: per-wave {start, end}
    // written by every kernel, and their reduction for the last kernel
    unsigned long long *d_kt;
    unsigned long long *d_kt_span;
    size_t kt_blocks;    // blocks in the last kernel
    int kt_rate_khz;     // s_memrealtime frequency

    // If UVM is disabled, host arrays for verification purposes
    std::vector<T> h_a, h_b, h_c;

  public:
    HIPStream(BenchId bs, const intptr_t array_size, const int device_id,
	      T initA, T initB, T initC);
    ~HIPStream();

    void copy() override;
    void add() override;
    void mul() override;
    void triad() override;
    void nstream() override;
    T dot() override;

    void get_arrays(T const*& a, T const*& b, T const*& c) override;    
    double last_kernel_seconds() override;
    void init_arrays(T initA, T initB, T initC);
};
