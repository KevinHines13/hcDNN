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
hcdnnCreate4DTensor(hcdnn4DTensorDesc_t *tensor_desc) {
  *tensor_desc = new hcdnn4DTensorStruct_t();
  return HCDNN_STATUS_SUCCESS;
}

hcdnnStatus_t
hcdnnSet4DTensor(hcdnn4DTensorDesc_t tensor_desc,
                 int n,
                 int c,
                 int h,
                 int w,
                 hcdnnDataType_t data_type) {
  tensor_desc->n_ = n;
  tensor_desc->c_ = c;
  tensor_desc->h_ = h;
  tensor_desc->w_ = w;
  tensor_desc->data_type_ = data_type;
  return HCDNN_STATUS_SUCCESS;
}

hcdnnStatus_t
hcdnnDestroy4DTensor(hcdnn4DTensorDesc_t tensor_desc) {
  delete tensor_desc;
  return HCDNN_STATUS_SUCCESS;
}

