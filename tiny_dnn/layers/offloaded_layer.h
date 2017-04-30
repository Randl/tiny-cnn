/*
    Copyright (c) 2016, Xilinx, Inc.
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

// offloaded_layer -- simply calls a hook function every time its forward pass
// is called

#include <vector>
#include "tiny_dnn/layers/layer.h"
#include "tiny_dnn/util/product.h"
#include "tiny_dnn/util/util.h"

namespace tiny_dnn {

typedef struct {
  serial_size_t in_width;      // input feature map width
  serial_size_t in_height;     // input feature map height
  serial_size_t window_size;   // convolution kernel window size
  serial_size_t in_channels;   // # input feature maps
  serial_size_t out_channels;  // # output feature maps
} OffloadConvParams;

#ifdef SOLITAIRE
// function type for offload handling. args are (input, output, offloadID, conv
// params if any or 0, target set of weigths)
typedef void (*OffloadHandler)(
  const vec_t&, vec_t&, size_t, OffloadConvParams*, size_t);

class offloaded_layer : public layer<activation::identity> {
 public:
  typedef layer<activation::identity> Base;

  offloaded_layer(serial_size_t in_dim,
                  serial_size_t out_dim,
                  OffloadHandler handler,
                  size_t offloadID,
                  OffloadConvParams* convParams = 0,
                  size_t targetSet              = 0)
    : Base(in_dim, out_dim, 0, 0),
      offloadHandler_(handler),
      offloadID_(offloadID),
      offloadConvParams_(convParams),
      targetSet_(targetSet)
#else
// function type for offload handling. args are (input, output, offloadID, conv
// params if any or 0)
typedef void (*OffloadHandler)(const vec_t&,
                               vec_t&,
                               size_t,
                               OffloadConvParams*);

class offloaded_layer : public layer<activation::identity> {
 public:
  typedef layer<activation::identity> Base;

  offloaded_layer(serial_size_t in_dim,
                  serial_size_t out_dim,
                  OffloadHandler handler,
                  size_t offloadID,
                  OffloadConvParams* convParams = 0)
    : Base(in_dim, out_dim, 0, 0),
      offloadHandler_(handler),
      offloadID_(offloadID),
      offloadConvParams_(convParams)
#endif
  {
    // disable parallelization for offloaded layers
    Base::set_parallelize(false);
  }

  std::string layer_type() const override { return "offloaded"; }

  size_t param_size() const override { return 0; }

  size_t connection_size() const override { return 0; }

  size_t fan_in_size() const override { return in_size_; }

  size_t fan_out_size() const override { return out_size_; }

  // forward prop does nothing except calling the
  const vec_t& forward_propagation(const vec_t& in, size_t index) override {
    vec_t& out = output_[index];
#ifdef SOLITAIRE
    offloadHandler_(in, out, offloadID_, offloadConvParams_, targetSet_);
#else
    offloadHandler_(in, out, offloadID_, offloadConvParams_);
#endif
    return next_ ? next_->forward_propagation(out, index) : out;
  }

  // offloaded layer is feedforward only, does not support training
  const vec_t& back_propagation(const vec_t& curr_delta,
                                size_t index) override {
    throw "Not implemented";
    return curr_delta;
  }

  const vec_t& back_propagation_2nd(const vec_t& current_delta2) override {
    throw "Not implemented";
    return current_delta2;
  }

 protected:
  OffloadHandler offloadHandler_;
  OffloadConvParams* offloadConvParams_;
  size_t offloadID_;
#ifdef SOLITAIRE
  size_t targetSet_;
#endif
};

}  // namespace tiny_dnn
