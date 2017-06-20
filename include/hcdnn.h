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

#ifndef INCLUDE_HCDNN_H_
#define INCLUDE_HCDNN_H_

typedef enum {
  HCDNN_STATUS_SUCCESS = 0,
  HCDNN_STATUS_ERROR
} hcdnnStatus_t; 

typedef enum {
  HCDNN_FLOAT = 0,
  HCDNN_DOUBLE
} hcdnnDataType_t;

typedef struct {
  int n_;
  int c_;
  int h_;
  int w_;
  hcdnnDataType_t data_type_;
} hcdnn4DTensorStruct_t;

typedef hcdnn4DTensorStruct_t* hcdnn4DTensorDesc_t ;

typedef struct {
  int pad_h_;
  int pad_w_;
  int u_;
  int v_;
  int upscale_x_;
  int upscale_y_;
} hcdnnConvolutionStruct_t;

typedef hcdnnConvolutionStruct_t* hcdnnConvolutionDesc_t;

typedef struct {
  int k_;
  int c_;
  int h_;
  int w_;
  hcdnnDataType_t data_type_;
} hcdnnConvKernelStruct_t;

typedef hcdnnConvKernelStruct_t* hcdnnConvKernelDesc_t;

hcdnnStatus_t
hcdnnCreate4DTensor(hcdnn4DTensorDesc_t *tensor_desc);

hcdnnStatus_t
hcdnnSet4DTensor(hcdnn4DTensorDesc_t tensor_desc,
                 int n,
                 int c,
                 int h,
                 int w,
                 hcdnnDataType_t data_type);

hcdnnStatus_t
hcdnnDestroy4DTensor(hcdnn4DTensorDesc_t tensor_desc);

hcdnnStatus_t
hcdnnCreateConvolutionDescriptor(hcdnnConvolutionDesc_t *convolution_desc);

hcdnnStatus_t
hcdnnSetConvolutionDescriptor(hcdnnConvolutionDesc_t convolution_desc,
                              int pad_h,
                              int pad_w,
                              int u,
                              int v,
                              int upscale_x,
                              int upscale_y);
hcdnnStatus_t
hcdnnDestroyConvolutionDescriptor(hcdnnConvolutionDesc_t convolution_desc);

hcdnnStatus_t
hcdnnCreateConvKernelDescriptor(hcdnnConvKernelDesc_t *kernel_desc);

hcdnnStatus_t
hcdnnSetConvKernelDescriptor(hcdnnConvKernelDesc_t kernel_desc,
                             int k,
                             int c,
                             int h,
                             int w,
                             hcdnnDataType_t data_type);

hcdnnStatus_t
hcdnnDestroyConvKernelDescriptor(hcdnnConvKernelDesc_t kernel_desc);

hcdnnStatus_t
hcdnnConvolutionForward(const void *alpha,
                        const hcdnn4DTensorDesc_t x_desc,
                        const void *x,
                        const hcdnnConvKernelDesc_t w_desc,
                        const void *w,
                        const hcdnnConvolutionDesc_t conv_desc,
                        const void *beta,
                        const hcdnn4DTensorDesc_t y_desc,
                        void *y);
//lrn
typedef struct {
	int K_;
	int n_;
	float alpha_;
	float beta_;
}hcdnnLRNStruct_t;

typedef hcdnnLRNStruct_t* hcdnnLRNDesc_t;

hcdnnStatus_t
hcdnnCreateLRNDescriptor(hcdnnLRNDesc_t *pooling_desc); //pooling desc
							
hcdnnStatus_t
hcdnnSetLRNDescriptor(hcdnnLRNDesc_t  lrn_desc,
	unsigned		lrnN,
	double			lrnAlpha,
	double			lrnBeta,
	double			lrnK);

hcdnnStatus_t

hcdnnGetLRNDescriptor(	hcdnnLRNDesc_t	lrn_desc,
	unsigned		*lrnN,
	double			*lrnAlpha,
	double			*lrnBeta,
	double			*lrnK);

hcdnnStatus_t
hcdnnDestroyLRNDescriptor(hcdnnLRNDesc_t	lrn_desc);

hcdnnStatus_t
hcdnnLRNForward(
	hcdnnLRNDesc_t				lrn_desc,
	const void					*alpha,
	const hcdnn4DTensorDesc_t	x_desc,
	const void					*x,
	const void					*beta,
	const hcdnn4DTensorDesc_t	y_desc,
	void						*y);

hcdnnStatus_t
hcdnnLRNBackward(
	hcdnnLRNDesc_t				lrn_desc,
	const void					*alpha,
	const hcdnn4DTensorDesc_t	y_desc,
	const void					*y,
	const hcdnn4DTensorDesc_t	dy_desc,
	const void					*dy,
	const hcdnn4DTensorDesc_t	x_desc,
	const void					*x,
	const void					*beta,
	const hcdnn4DTensorDesc_t	dx_desc,
	void						*dx);

#endif // INCLUDE_HCDNN_H_
