/*
* Software License Agreement (BSD License)
*
*  Copyright (c) 2011, Willow Garage, Inc.
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of Willow Garage, Inc. nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*
*  Author: Anatoly Baskeheev, Itseez Ltd, (myname.mysurname@mycompany.com)
*/

/* For syntax highlight */
#ifndef __CUDACC__
#define __CUDACC__
#endif
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

#include "internal.hpp"

#include "pcl/gpu/utils/safe_call.hpp"
#include "pcl/gpu/utils/device/warp.hpp"
#include "pcl/gpu/utils/device/funcattrib.hpp"

#include "pcl/gpu/features/device/eigen.hpp"

using namespace pcl::gpu;

namespace pcl
{
	namespace device
	{
		struct IRFEstimator
		{
			enum
			{
				CTA_SIZE = 256,
				WAPRS = CTA_SIZE / Warp::WARP_SIZE,

				MIN_NEIGHBOORS = 1
			};

			struct plus
			{
				__forceinline__ __device__ float operator()(const float &lhs, const volatile float& rhs) const { return lhs + rhs; }
			};

			PtrStep<int> indices;
			const int *sizes;
			const PointType *points;

			PtrSz<IRFType> irfs;

			__device__ __forceinline__ void operator()() const
			{
				__shared__ float cov_buffer[6][CTA_SIZE + 1];
				__shared__ float w_buffer[CTA_SIZE + 1];

				int warp_idx = Warp::id();
				int idx = blockIdx.x * WAPRS + warp_idx;	// index of central point / current point of which normal will be computed

				if (idx >= irfs.size)
					return;

				int size = sizes[idx];	// number of central point's neighbors
				int lane = Warp::laneId();

				if (size < MIN_NEIGHBOORS)
				{
					const float NaN = numeric_limits<float>::quiet_NaN();
					IRFType irf;
					irf.axis_x = irf.axis_y = irf.axis_z = make_float3(NaN, NaN, NaN);
					if (lane == 0) {
						irfs.data[idx] = irf;
					}
				}

				// get centroid (aka current point)
				float3 c = fetch(idx);

				//nvcc bug workaround. if comment this => c.z == 0 at line: float3 d = fetch(*t) - c;
				//__threadfence_block();

				//compute covariance matrix        
				int tid = threadIdx.x;

				for (int i = 0; i < 6; ++i)
					cov_buffer[i][tid] = 0.f;

				// get neighbors of current point
				const int *ibeg = indices.ptr(idx);
				const int *iend = ibeg + size;

				for (const int *t = ibeg + lane; t < iend; t += Warp::STRIDE)
				{
					float w = 1.f / sizes[*t];
					float3 p = fetch(*t);
					float3 d = p - c;

					cov_buffer[0][tid] += w * d.x * d.x; //cov (0, 0) 
					cov_buffer[1][tid] += w * d.x * d.y; //cov (0, 1) 
					cov_buffer[2][tid] += w * d.x * d.z; //cov (0, 2) 
					cov_buffer[3][tid] += w * d.y * d.y; //cov (1, 1) 
					cov_buffer[4][tid] += w * d.y * d.z; //cov (1, 2) 
					cov_buffer[5][tid] += w * d.z * d.z; //cov (2, 2)

					w_buffer[tid] += w;	//sum w
				}

				Warp::reduce(&cov_buffer[0][tid - lane], plus());
				Warp::reduce(&cov_buffer[1][tid - lane], plus());
				Warp::reduce(&cov_buffer[2][tid - lane], plus());
				Warp::reduce(&cov_buffer[3][tid - lane], plus());
				Warp::reduce(&cov_buffer[4][tid - lane], plus());
				Warp::reduce(&cov_buffer[5][tid - lane], plus());

				Warp::reduce(&w_buffer[tid - lane], plus());

				volatile float *cov = &cov_buffer[0][tid - lane];
				if (lane < 6)
					cov[lane] = cov_buffer[lane][tid - lane] / w_buffer[tid - lane];

				//solvePlaneParameters
				if (lane == 0)
				{
					// Extract the eigenvalues and eigenvectors
					typedef Eigen33::Mat33 Mat33;
					Eigen33 eigen33(&cov[lane]);

					Mat33&     tmp = (Mat33&)cov_buffer[1][tid - lane];
					Mat33& vec_tmp = (Mat33&)cov_buffer[2][tid - lane];
					Mat33& evecs = (Mat33&)cov_buffer[3][tid - lane];
					float3 evals;

					eigen33.compute(tmp, vec_tmp, evecs, evals);
					//evecs[0] - eigenvector with the lowerst eigenvalue

					IRFType output;
			
					// Axis-X
					// The normalization is not necessary, since the eigenvectors from Eigen33 are already normalized
					output.axis_x.x = evecs[2].x;
					output.axis_x.y = evecs[2].y;
					output.axis_x.z = evecs[2].z;

					// Axis-Y
					// The normalization is not necessary, since the eigenvectors from Eigen33 are already normalized
					output.axis_x.x = evecs[1].x;
					output.axis_x.y = evecs[1].y;
					output.axis_x.z = evecs[1].z;

					// Axis-Z
					output.axis_z = output.axis_x ^ output.axis_y;

					irfs.data[idx] = output;
				}
			}

			__device__ __forceinline__ float3 fetch(int idx) const
			{
				/*PointType p = points[idx];
				return make_float3(p.x, p.y, p.z);*/
				return *(float3*)&points[idx];
			}

		};

		__global__ void EstimateIRFKernel(const IRFEstimator est) { est(); }
	}
}

void pcl::device::computeIRF(const PointCloud& cloud, const NeighborIndices& nn_indices, Normals& normals)
{
	IRFEstimator est;
	est.indices = nn_indices;	// convert NieghborIndices to PtrStep<int>
	est.sizes = nn_indices.sizes;
	est.points = cloud;
	est.normals = normals;

	//printFuncAttrib(EstimateNormaslKernel);

	int block = IRFEstimator::CTA_SIZE;
	int grid = divUp((int)normals.size(), IRFEstimator::WAPRS);
	EstimateNormaslKernel << <grid, block >> >(est);

	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
}