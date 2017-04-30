/*
    Copyright (c) 2016, Xilinx, Inc.
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once
#include <fstream>
#include <string>
#include "tiny_dnn/layers/layer.h"

// right now functional during inference only
// pre-trained batchnorm params must be manually set
// TODO:
// - implement proper batch normalization during training

// during training, for each neuron output, batch normalization learns
// four statistics from the data that passes through it. we keep all
// four of these as part of the layer's weights, in the following order:
// * all of the shifts (beta)
// * all of the scales (gamma)
// * all of the means (mu)
// * all of the inverted stddevs (inv_std)

namespace tiny_dnn {

class batchnorm_layer : public layer {
 public:
  typedef layer Base;

  /**
   *
   * @param channels number of channels. each channel has a batchnorm
   * parameter set.
   * @param dim number of pixels/elements in each channel.
   * @param paramFile
   */
  batchnorm_layer(serial_size_t channels,
                  serial_size_t dim     = 1,
                  std::string paramFile = "")
    : Base(dim * channels, dim * channels, 4 * channels, 0),
      dim_(dim),
      channels_(channels) {
    if (paramFile != "") {
      loadFromBinaryFile(paramFile);
    }
  }

  void loadFromBinaryFile(std::string fileName) {
    // TODO this assumes the binary file always uses 4 bytes per batchnorm
    // parameter

    std::ifstream wf(fileName, std::ios::binary | std::ios::in);
    for (size_t line = 0; line < Base::W_.size(); line++) {
      float e = 0;
      wf.read((char*)&e, sizeof(float));
      W_[line] = e;
    }
    wf.close();
  }

  size_t connection_size() const { return in_size_; }

  serial_size_t fan_in_size() const override { return dim_; }

  serial_size_t fan_out_size() const override { return dim_; }

  void forward_propagation(const vec_t& in, size_t index) override {
    // TODO(Randl)
    vec_t& a   = a_[index];
    vec_t& out = output_[index];

    for_i(parallelize_, channels_, [&](int ch) {
      for (size_t j = 0; j < dim_; j++) {
        size_t pos = ch * dim_ + j;
        a[pos]     = gamma(ch) * (in[pos] - mean(ch)) * invstd(ch) + beta(ch);
      }
    });

    for_i(parallelize_, out_size_, [&](int i) { out[i] = h_.f(a, i); });
    CNN_LOG_VECTOR(out, "[bn]forward");

    return next_ ? next_->forward_propagation(out, index) : out;
  }

  void back_propagation(const vec_t& curr_delta, size_t index) override {
    throw "Not yet implemented";
    return curr_delta;
  }

  void back_propagation_2nd(const vec_t& current_delta2) override {
    throw "Not yet implemented";
    return current_delta2;
  }

  std::string layer_type() const override { return "batchnorm"; }

 protected:
  size_t dim_;
  size_t channels_;
  inline float_t beta(size_t ind) { return W_[(channels_ * 0) + ind]; }

  inline float_t gamma(size_t ind) { return W_[(channels_ * 1) + ind]; }

  inline float_t mean(size_t ind) { return W_[(channels_ * 2) + ind]; }

  inline float_t invstd(size_t ind) { return W_[(channels_ * 3) + ind]; }
};

}  // namespace tiny_dnn
