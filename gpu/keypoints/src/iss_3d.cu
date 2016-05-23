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

#include "internal.hpp"

#include "pcl/gpu/utils/device/funcattrib.hpp"
#include "pcl/gpu/utils/device/warp.hpp"
#include "pcl/gpu/utils/safe_call.hpp"

#include "pcl/gpu/utils/device/eigen.hpp"

using namespace pcl::gpu;

namespace pcl
{
    namespace device
    {
        struct MaxEigenValueCalculator
        {
            enum
            {
                CTA_SIZE = 256,
                WAPRS = CTA_SIZE / Warp::WARP_SIZE,

                MIN_NEIGHBOORS = 1
            };

            struct plus
            {
                __forceinline__ __device__ float operator()(const float& lhs, const volatile float& rhs) const
                {
                    return lhs + rhs;
                }
            };

            PtrStep<int> indices;
            const int min_neighboors;
            const int* sizes;
            const PointType* points;

            const float threshold21, threshold32;

            PtrSz<float> max_eigen_value;  // Step 1, return max eigen value of each point

            __device__ __forceinline__ void operator()() const
            {
                __shared__ float cov_buffer[6][CTA_SIZE + 1];
                __shared__ float w_buffer[CTA_SIZE + 1];

                int warp_idx = Warp::id();
                int idx = blockIdx.x * WAPRS +
                          warp_idx;  // index of central point / current point of which normal will be computed

                if (idx >= max_eigen_value.size) return;

                int size = sizes[idx];  // number of central point's neighbors
                int lane = Warp::laneId();

                if (size < MIN_NEIGHBOORS || size < min_neighboors)
                {
                    if (lane == 0) max_eigen_value[idx] = -1.0;
                    return;
                }

                // get centroid(current point)
                float3 c = fetch(idx);

                // nvcc bug workaround. if comment this => c.z == 0 at line: float3 d = fetch(*t) - c;
                __threadfence_block();

                // compute covariance matrix
                int tid = threadIdx.x;

                for (int i = 0; i < 6; ++i) cov_buffer[i][tid] = 0.f;
                w_buffer[tid] = 0.f;

                // get neighbors of current point
                const int* ibeg = indices.ptr(idx);
                const int* iend = ibeg + size;

                for (const int* t = ibeg + lane; t < iend; t += Warp::STRIDE)
                {
                    float w = 1.f / sizes[*t];
                    float3 p = fetch(*t);
                    float3 d = p - c;

                    cov_buffer[0][tid] += w * d.x * d.x;  // cov (0, 0)
                    cov_buffer[1][tid] += w * d.x * d.y;  // cov (0, 1)
                    cov_buffer[2][tid] += w * d.x * d.z;  // cov (0, 2)
                    cov_buffer[3][tid] += w * d.y * d.y;  // cov (1, 1)
                    cov_buffer[4][tid] += w * d.y * d.z;  // cov (1, 2)
                    cov_buffer[5][tid] += w * d.z * d.z;  // cov (2, 2)

                    w_buffer[tid] += w;  // sum w
                }

                Warp::reduce(&cov_buffer[0][tid - lane], plus());
                Warp::reduce(&cov_buffer[1][tid - lane], plus());
                Warp::reduce(&cov_buffer[2][tid - lane], plus());
                Warp::reduce(&cov_buffer[3][tid - lane], plus());
                Warp::reduce(&cov_buffer[4][tid - lane], plus());
                Warp::reduce(&cov_buffer[5][tid - lane], plus());

                Warp::reduce(&w_buffer[tid - lane], plus());

                volatile float* cov = &cov_buffer[0][tid - lane];
                if (lane < 6) cov[lane] = cov_buffer[lane][tid - lane] / w_buffer[tid - lane];

                // solvePlaneParameters
                if (lane == 0)
                {
                    // Extract the eigenvalues and eigenvectors
                    typedef Eigen33::Mat33 Mat33;
                    Eigen33 eigen33(&cov[lane]);

                    Mat33& tmp = (Mat33&)cov_buffer[1][tid - lane];
                    Mat33& vec_tmp = (Mat33&)cov_buffer[2][tid - lane];
                    Mat33& evecs = (Mat33&)cov_buffer[3][tid - lane];
                    float3 evals;

                    eigen33.compute(tmp, vec_tmp, evecs, evals);
                    // evecs[0] - eigenvector with the lowerst eigenvalue

                    gamma21 = evals[1] / evals[0];
                    gamma32 = evals[2] / evals[1];
                    if (gamma21 < threshold21 || gamma32 < threshold32)
                        max_eval_buffer[idx] = evals[2];
                    else
                        max_eval_buffer[idx] = -1.;
                }
            }

            __device__ __forceinline__ float3 fetch(int idx) const
            {
                /*PointType p = points[idx];
                return make_float3(p.x, p.y, p.z);*/
                return *(float3*)&points[idx];
            }
        };

        struct NonMaxSuppressor
        {
            enum
            {
                CTA_SIZE = 256,
                WAPRS = CTA_SIZE / Warp::WARP_SIZE,

                MIN_NEIGHBOORS = 1
            };

            struct logic_and
            {
                __forceinline__ __device__ bool operator()(const bool& lhs, const volatile bool& rhs) const
                {
                    return lhs && rhs;
                }
            };

            PtrSz<float> max_eigen_value;
            PtrStep<int> indices;
            const int min_neighboors;
            const int* sizes;

            PtrSz<bool> is_keypoint;

            __device__ __forceinline__ void operator()() const
            {
                __shared__ bool is_max[CTA_SIZE + 1];

                int warp_idx = Warp::id();
                int idx = blockIdx.x * WAPRS + warp_idx;

                if (idx >= max_eigen_value.size) return;

                int size = sizes[idx];
                int lane = Warp::laneId();

                if (max_eigen_value[idx] < 0.0)
                {
                    if (lane == 0) is_keypoint[idx] = false;
                    return;
                }

                int tid = threadIdx.x;

                is_max[tid] = true;

                const int* ibeg = indices.ptr(idx);
                const int* iend = ibeg + size;

                for (const int* t = ibeg + lane; t < iend; t += Warp::STRIDE)
                {
                    if (max_eigen_value[idx] < max_eigen_value[*t]) {
                        is_max[tid] = false;
                        break;
                    }
                }

                Warp::reduce(&is_max[tid - lane], logic_and());

                if (lane == 0) 
                    is_keypoint[idx] = ((max_eigen_value[idx] != -1.) && is_max[tid - lane]);
            }
        };

        __global__ void MaxEigenValueCalculatorKernel(const MaxEigenValueCalculator mevc) { mevc(); }
        __global__ void NonMaxSuppressorKernel(const NonMaxSuppressor nms) { nms(); }
    }
}

void pcl::device::detectISSKeypoint3D(
    const PointCloud& cloud, const int min_neighboors,
    const NeighborIndices& nn_indices,   // NeighborIndices for calculate max eigen value of scatter matrix
    const NeighborIndices& nn_indices2,  // NeighborIndices for non max suppress/detect keypoints
    IsKeypoint is_keypoint)
{
    // Step 1. calculate max eigen value of each point
    DeviceArray<float> max_eigen_value;
    max_eigen_value.create(cloud.size());  // TODO: check the usage of `create`

    MaxEigenValueCalculator mevc;
    mevc.min_neighboors = min_neighboors;
    mevc.indices = nn_indices;  // convert NieghborIndices to PtrStep<int>
    mevc.sizes = nn_indices.sizes;
    mevc.points = cloud;
    mevc.max_eigen_value = max_eigen_value;

    int block = MaxEigenValueCalculator::CTA_SIZE;    // number of threads in each block
    int grid = divUp((int)max_eigen_value.size(), MaxEigenValueCalculator::WAPRS);    // number of blocks in each grid
    MaxEigenValueCalculatorKernel<<<grid, block>>>(mevc);

    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());

    // Step 2. non-maximum suppression for each point
    NonMaxSuppressor nms;
    nms.min_neighboors = min_neighboors;
    nms.indices = nn_indices2;
    nms.sizes = nn_indices2.sizes;
    nms.max_eigen_value = max_eigen_value;
    nms.is_keypoint = is_keypoint;

    int block = NonMaxSuppressor::CTA_SIZE;
    int grid = divUp((int)is_keypoint.size(), NonMaxSuppressor::WAPRS);
    NonMaxSuppressorKernel<<<grid, block>>>(nms);

    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());
}