/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include <tvm/runtime/c_runtime_api.h>

#include <assert.h>
#include <stdio.h>
#include <sys/time.h>
#include <float.h>
#include <math.h>

#include "bundle.h"
#include "build/keyword_graph.json.c"
#include "build/keyword_params.bin.c"

#define in_dim0     1
#define in_dim1     49
#define in_dim2     10

#define out_dim0    1
#define out_dim1    12

int main(int argc, char **argv) {
  // assert(argc == 2 && "Usage: demo_static <cat.bin>");

  char * json_data = (char *)(build_keyword_graph_json);
  char * params_data = (char *)(build_keyword_params_bin);
  uint64_t params_size = build_keyword_params_bin_len;

  struct timeval t0, t1, t2, t3, t4, t5;
  gettimeofday(&t0, 0);

  auto *handle = tvm_runtime_create(json_data, params_data, params_size);
  gettimeofday(&t1, 0);

  float input_storage[in_dim0 * in_dim1 * in_dim2];
  FILE * fp = fopen(argv[1], "rb");
  fread(input_storage, in_dim0 * in_dim1 * in_dim2, 4, fp);
  fclose(fp);

  DLTensor input;
  input.data = input_storage;
  DLContext ctx = {kDLCPU, 0};
  input.ctx = ctx;
  input.ndim = 3;
  DLDataType dtype = {kDLFloat, 32, 1};
  input.dtype = dtype;
  int64_t shape [3] = {in_dim0, in_dim1, in_dim2};
  input.shape = shape;
  input.strides = NULL;
  input.byte_offset = 0;

  tvm_runtime_set_input(handle, "Mfcc", &input);
  gettimeofday(&t2, 0);

  tvm_runtime_run(handle);
  gettimeofday(&t3, 0);

  float output_storage[out_dim0 * out_dim1];
  DLTensor output;
  output.data = output_storage;
  DLContext out_ctx = {kDLCPU, 0};
  output.ctx = out_ctx;
  output.ndim = 2;
  DLDataType out_dtype = {kDLFloat, 32, 1};
  output.dtype = out_dtype;
  int64_t out_shape [2] = {out_dim0, out_dim1};
  output.shape = out_shape;
  output.strides = NULL;
  output.byte_offset = 0;

  tvm_runtime_get_output(handle, 0, &output);
  gettimeofday(&t4, 0);


  float exp_out[out_dim0 * out_dim1];
  fp = fopen(argv[2], "rb");
  fread(exp_out, out_dim0 * out_dim1, 4, fp);
  fclose(fp);

  int result = 1;
  for (int i = 0; i < (out_dim0 * out_dim1); ++i) {
    fprintf(stdout, "out: %f.4,\t expt: %f.4\n", output_storage[i], exp_out[i]);
    if (fabs(output_storage[i] - exp_out[i]) >= 1e-3f) {
      result = 0;
      break;
    }
  }

  float max_iter = -FLT_MAX;
  int32_t max_index = -1;
  for (int i = 0; i < out_dim1; ++i) {
    if (output_storage[i] > max_iter) {
      max_iter = output_storage[i];
      max_index = i;
    }
  }

  fprintf(stdout, "The maximum position in output vector is: %d, with max-value %f.\n",
         max_index, max_iter);
  fprintf(stdout, "Resutl: %d\n", result);
  tvm_runtime_destroy(handle);
  gettimeofday(&t5, 0);

  // printf("The maximum position in output vector is: %d, with max-value %f.\n",
  //        max_index, max_iter);
  printf("timing: %.2f ms (create), %.2f ms (set_input), %.2f ms (run), "
         "%.2f ms (get_output), %.2f ms (destroy)\n",
         (t1.tv_sec-t0.tv_sec)*1000000 + (t1.tv_usec-t0.tv_usec)/1000.f,
         (t2.tv_sec-t1.tv_sec)*1000000 + (t2.tv_usec-t1.tv_usec)/1000.f,
         (t3.tv_sec-t2.tv_sec)*1000000 + (t3.tv_usec-t2.tv_usec)/1000.f,
         (t4.tv_sec-t3.tv_sec)*1000000 + (t4.tv_usec-t3.tv_usec)/1000.f,
         (t5.tv_sec-t4.tv_sec)*1000000 + (t5.tv_usec-t4.tv_usec)/1000.f);
  
  return 0;
}
