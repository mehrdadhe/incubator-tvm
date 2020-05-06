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
#include "build/cifar_graph.json.c"
#include "build/cifar_params.bin.c"

#define in_dim0     1
#define in_dim1     3
#define in_dim2     32
#define in_dim3     32

#define out_dim0    1
#define out_dim1    10

int main(int argc, char **argv) {
  // assert(argc == 2 && "Usage: demo_static <cat.bin>");

  char * json_data = (char *)(build_cifar_graph_json);
  char * params_data = (char *)(build_cifar_params_bin);
  uint64_t params_size = build_cifar_params_bin_len;

  struct timeval t0, t1, t2, t3, t4, t5;
  gettimeofday(&t0, 0);

  auto *handle = tvm_runtime_create(json_data, params_data, params_size);
  gettimeofday(&t1, 0);

  float input_storage[in_dim0 * in_dim1 * in_dim2 * in_dim3];
  FILE * fp = fopen(argv[1], "rb");
  fread(input_storage, in_dim0 * in_dim1 * in_dim2 * in_dim3, 4, fp);
  fclose(fp);

  DLTensor input;
  input.data = input_storage;
  DLContext ctx = {kDLCPU, 0};
  input.ctx = ctx;
  input.ndim = 4;
  DLDataType dtype = {kDLInt, 8, 1};
  input.dtype = dtype;
  int64_t shape [4] = {in_dim0, in_dim1, in_dim2, in_dim3};
  input.shape = shape;
  input.strides = NULL;
  input.byte_offset = 0;

  tvm_runtime_set_input(handle, "data", &input);
  gettimeofday(&t2, 0);

  tvm_runtime_run(handle);
  gettimeofday(&t3, 0);

  float output_storage[out_dim0 * out_dim1];
  DLTensor output;
  output.data = output_storage;
  DLContext out_ctx = {kDLCPU, 0};
  output.ctx = out_ctx;
  output.ndim = 2;
  DLDataType out_dtype = {kDLInt, 8, 1};
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

  float max_iter = -FLT_MAX;
  int32_t max_index = -1;
  int result = 1;
  for (int i = 0; i < (out_dim0 * out_dim1); ++i) {
    if (fabs(output_storage[i] - exp_out[i]) >= 1e-3f) {
      result = 0;
      break;
    }
  }

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
