#pragma once
#include "tiny_cnn/layers/layer.h"
#include "tiny_cnn/activations/activation_function.h"
#include "tiny_cnn/util/util.h"
#include <vector>

namespace tiny_cnn {
class bnn_threshold_layer : public layer<activation::identity> {
public:
    typedef layer<activation::identity> Base;

    // channels: number of channels. each channel has a separate threshold
    // dim: number of pixels/elements in each channel.
    bnn_threshold_layer(cnn_size_t channels, cnn_size_t dim = 1)
        : Base(dim*channels, dim*channels, 0, 0), dim_(dim), channels_(channels),
          thresholds_(channels, 0), invertOutput_(channels, false)
    {
      set_parallelize(false);
    }

    std::vector<int> & thresholds() {
      return thresholds_;
    }

    std::vector<bool> & invertOutput() {
      return invertOutput_;
    }

    size_t connection_size() const override {
        return in_size_;
    }

    size_t fan_in_size() const override {
        return dim_;
    }

    size_t fan_out_size() const override {
        return dim_;
    }

    const vec_t& forward_propagation(const vec_t& in, size_t index) override {
        vec_t &out = output_[index];

        for(unsigned int ch = 0; ch < channels_; ch++) {
          for(unsigned int j = 0; j < dim_; j++) {
              unsigned int pos = ch*dim_ + j;
              out[pos] = (in[pos] >= thresholds_[ch] ? +1 : -1);

              if(invertOutput_[ch])
                  out[pos] = -out[pos];
          }
        }

        return next_ ? next_->forward_propagation(out, index) : out;
    }

    const vec_t& back_propagation(const vec_t& curr_delta, size_t index) override {
        throw "Not yet implemented";
        return curr_delta;
    }

    const vec_t& back_propagation_2nd(const vec_t& current_delta2) override {
        throw "Not yet implemented";
        return current_delta2;
    }

    std::string layer_type() const override { return "bnn_threshold_layer"; }

protected:
    unsigned int dim_;
    unsigned int channels_;


    std::vector<int> thresholds_;
    std::vector<bool> invertOutput_;
};

} // namespace tiny_cnn
