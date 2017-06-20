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
#include <hcc/hc.hpp>

hcdnnStatus_t
hcdnnCreateLRNDescriptor(hcdnnLRNDesc_t *pooling_desc) {
	*pooling_desc = new hcdnnLRNStruct_t();
	return HCDNN_STATUS_SUCCESS;
}

hcdnnStatus_t
hcdnnSetLRNDescriptor(hcdnnLRNDesc_t  lrn_desc,
	unsigned		lrnN,
	double			lrnAlpha,
	double			lrnBeta,
	double			lrnK) {
	lrn_desc->n_ = lrnN;
	lrn_desc->alpha_ = lrnAlpha;
	lrn_desc->beta_ = lrnBeta;
	lrn_desc->K_ = lrnK;
	return HCDNN_STATUS_SUCCESS;
}

hcdnnStatus_t
hcdnnDestroyLRNDescriptor(hcdnnLRNDesc_t lrn_desc) {
	delete lrn_desc;
	return HCDNN_STATUS_SUCCESS;
}

hcdnnStatus_t
hcdnnLRNForward(
	hcdnnLRNDesc_t				lrn_desc,
	const void					*alpha,
	const hcdnn4DTensorDesc_t	x_desc,
	const void					*x,
	const void					*beta,
	const hcdnn4DTensorDesc_t	y_desc,
	void						*y) {

	//implementation
	
	//organize input tensor  in array_view and pass to a parallel_for_each...


		
		

	return HCDNN_STATUS_SUCCESS;

