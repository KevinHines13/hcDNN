//MIT License
//
//Copyright (c) 2017 Shi Dong
//
//Permission is hereby granted, free of charge, to any person obtaining a copy
//of this software and associated documentation files (the "Software"), to deal
//in the Software without restriction, including without limitation the rights
//to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//copies of the Software, and to permit persons to whom the Software is
//furnished to do so, subject to the following conditions:
//
//The above copyright notice and this permission notice shall be included in all
//copies or substantial portions of the Software.
//
//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//SOFTWARE.

#include "hcdnn.h"

hcdnnStatus_t
hcdnnCreateConvolutionDescriptor(hcdnnConvolutionDesc_t *convolution_desc) {
  *convolution_desc = new hcdnnConvolutionStruct_t();
  return HCDNN_STATUS_SUCCESS;
}

hcdnnStatus_t
hcdnnSetConvolutionDescriptor(hcdnnConvolutionDesc_t convolution_desc,
                              int pad_h,
                              int pad_w,
                              int u,
                              int v,
                              int upscale_x,
                              int upscale_y) {
  convolution_desc->pad_h_ = pad_h;
  convolution_desc->pad_w_ = pad_w;
  convolution_desc->u_ = u;
  convolution_desc->v_ = v;
  convolution_desc->upscale_x_ = upscale_x;
  convolution_desc->upscale_y_ = upscale_y;
  return HCDNN_STATUS_SUCCESS;
}

hcdnnStatus_t
hcdnnDestroyConvolutionDescriptor(hcdnnConvolutionDesc_t convolution_desc) {
  delete convolution_desc;
  return HCDNN_STATUS_SUCCESS;
}

hcdnnStatus_t
hcdnnCreateConvKernelDescriptor(hcdnnConvKernelDesc_t *kernel_desc) {
  *kernel_desc = new hcdnnConvKernelStruct_t;
  return HCDNN_STATUS_SUCCESS;
}

hcdnnStatus_t
hcdnnSetConvKernelDescriptor(hcdnnConvKernelDesc_t kernel_desc,
                             int k,
                             int c,
                             int h,
                             int w,
                             hcdnnDataType_t data_type) {
  kernel_desc->k_ = k;
  kernel_desc->c_ = c;
  kernel_desc->h_ = h;
  kernel_desc->w_ = w;
  kernel_desc->data_type_ = data_type;
  return HCDNN_STATUS_SUCCESS;
}

hcdnnStatus_t
hcdnnDestroyConvKernelDescriptor(hcdnnConvKernelDesc_t kernel_desc) {
  delete kernel_desc;
  return HCDNN_STATUS_SUCCESS;
}

hcdnnStatus_t
hcdnnConvolutionForward(const void *alpha,
                        const hcdnn4DTensorDesc_t x_desc,
                        const void *x,
                        const hcdnnConvKernelDesc_t w_desc,
                        const void *w,
                        const hcdnnConvolutionDesc_t conv_desc,
                        const void *beta,
                        const hcdnn4DTensorDesc_t y_desc,
                        void *y) {
  // Implementation here
  return HCDNN_STATUS_SUCCESS;
}
