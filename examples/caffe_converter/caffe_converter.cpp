/*
Copyright (c) 2016, Taiga Nomi
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
* Neither the name of the <organization> nor the
names of its contributors may be used to endorse or promote products
derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#include <iostream>
#include <memory>
#include <ctime>
#define NO_STRICT
#define CNN_USE_CAFFE_CONVERTER
#include "tiny_dnn/tiny_dnn.h"

using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace std;

image<float> compute_mean(const string& mean_file, int width, int height)
{
    caffe::BlobProto blob;
    detail::read_proto_from_binary(mean_file, &blob);

    auto data = blob.mutable_data()->mutable_data();

    image<float> original(data, blob.width(), blob.height(), image_type::bgr);

    return mean_image(original);
}

void preprocess(const image<float>& img,
                 const image<float>& mean,
                 int width,
                 int height,
                 vec_t* dst)
{
    image<float> resized = resize_image(img, width, height);

    image<> resized_uint8(resized);

    if (!mean.empty()) {
        image<float> normalized = subtract_scalar(resized, mean);
        *dst = normalized.to_vec();
    }
    else {
        *dst = resized.to_vec();
    }
}

vector<string> get_label_list(const string& label_file)
{
    string line;
    ifstream ifs(label_file.c_str());

    if (ifs.fail() || ifs.bad())
        throw runtime_error("failed to open:" + label_file);

    vector<string> lines;
    while (getline(ifs, line))
        lines.push_back(line);

    return lines;
}

void load_validation_data(const string& validation_file,
                          vector<pair<string, int>>* validation) {
  string line;
  ifstream ifs(validation_file.c_str());

  if (ifs.fail() || ifs.bad()) {
      throw runtime_error("failed to open:" + validation_file);
  }

  vector<string> lines;
  while (getline(ifs, line)) {
      lines.push_back(line);
  }
}

void test(const string& model_file,
          const string& trained_file,
          const string& mean_file,
          const string& label_file,
          const string& img_file)
{
    auto labels = get_label_list(label_file);
    auto net = create_net_from_caffe_prototxt(model_file);
    reload_weight_from_caffe_protobinary(trained_file, net.get());

    // int channels = (*net)[0]->in_data_shape()[0].depth_;
    int width = (*net)[0]->in_data_shape()[0].width_;
    int height = (*net)[0]->in_data_shape()[0].height_;

    std::vector<std::pair<std::string, int>> validation(1);
    load_validation_data(img_file, &validation);

    auto mean = compute_mean(mean_file, width, height);

    for (size_t i = 0; i < validation.size(); ++i) {

      image<float> img(img_file, image_type::bgr);

      vec_t vec;

      preprocess(img, mean, width, height, &vec);

      clock_t begin = clock();
      
      auto result = net->predict(vec);

      clock_t end = clock();
      double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
      cout <<"Elapsed time(s): " << elapsed_secs << endl;

      vector<tiny_dnn::float_t> sorted(result.begin(), result.end());

      int top_n = 5;
      partial_sort(sorted.begin(), sorted.begin()+top_n, sorted.end(), greater<tiny_dnn::float_t>());

      for (int i = 0; i < top_n; i++) {
          size_t idx = distance(result.begin(), find(result.begin(), result.end(), sorted[i]));
          cout << labels[idx] << "," << sorted[i] << endl;
      }
    }

}

int main(int argc, char** argv) {
    int arg_channel = 1;
    string model_file = argv[arg_channel++];
    string trained_file = argv[arg_channel++];
    string mean_file = argv[arg_channel++];
    string label_file = argv[arg_channel++];
    string img_file = argv[arg_channel++];

    try {
        test(model_file, trained_file, mean_file, label_file, img_file);
    } catch (const nn_error& e) {
        cout << e.what() << endl;
    }
}
