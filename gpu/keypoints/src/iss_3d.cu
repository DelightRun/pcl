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

#include <thrust/count.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>

#include <stdio.h>

using namespace pcl::gpu;
using namespace thrust;

namespace pcl
{
    namespace device
    {
        struct ThirdEigenValueCalculator
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
            int min_neighboors;
            const int* sizes;
            const PointType* points;

            float threshold21, threshold32;

            PtrSz<float> third_eigen_value;  // Step 1, return max eigen value of each point

            __device__ __forceinline__ void operator()() const
            {
                __shared__ float cov_buffer[6][CTA_SIZE + 1];

                int warp_idx = Warp::id();
                int idx = blockIdx.x * WAPRS +
                          warp_idx;  // index of central point / current point of which normal will be computed

                if (idx >= third_eigen_value.size) return;

                int size = sizes[idx];  // number of central point's neighbors
                int lane = Warp::laneId();

                if (size < MIN_NEIGHBOORS || size < min_neighboors)
                {
                    if (lane == 0)
                        third_eigen_value.data[idx] = -1.0;
                    return;
                }

                // get centroid(current point)
                float3 c = fetch(idx);

                // nvcc bug workaround. if comment this => c.z == 0 at line: float3 d = fetch(*t) - c;
                __threadfence_block();

                // compute covariance matrix
                int tid = threadIdx.x;

                for (int i = 0; i < 6; ++i) cov_buffer[i][tid] = 0.f;

                // get neighbors of current point
                const int* ibeg = indices.ptr(idx);
                const int* iend = ibeg + size;

                for (const int* t = ibeg + lane; t < iend; t += Warp::STRIDE)
                {
                    float3 p = fetch(*t);
                    float3 d = p - c;

                    cov_buffer[0][tid] += d.x * d.x;  // cov (0, 0)
                    cov_buffer[1][tid] += d.x * d.y;  // cov (0, 1)
                    cov_buffer[2][tid] += d.x * d.z;  // cov (0, 2)
                    cov_buffer[3][tid] += d.y * d.y;  // cov (1, 1)
                    cov_buffer[4][tid] += d.y * d.z;  // cov (1, 2)
                    cov_buffer[5][tid] += d.z * d.z;  // cov (2, 2)
                }

                Warp::reduce(&cov_buffer[0][tid - lane], plus());
                Warp::reduce(&cov_buffer[1][tid - lane], plus());
                Warp::reduce(&cov_buffer[2][tid - lane], plus());
                Warp::reduce(&cov_buffer[3][tid - lane], plus());
                Warp::reduce(&cov_buffer[4][tid - lane], plus());
                Warp::reduce(&cov_buffer[5][tid - lane], plus());

                volatile float* cov = &cov_buffer[0][tid - lane];
                if (lane < 6) cov[lane] = cov_buffer[lane][tid - lane];

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

                    const float& e1c = evals.z;
                    const float& e2c = evals.y;
                    const float& e3c = evals.x;

                    if (e3c < 0) {
                        third_eigen_value.data[idx] = -1.;
                        return;
                    }

                    float gamma21 = e2c / e1c;
                    float gamma32 = e3c / e2c;

                    if (gamma21 < threshold21 || gamma32 < threshold32)
                        third_eigen_value.data[idx] = e3c;
                    else
                        third_eigen_value.data[idx] = -1.;
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

            PtrSz<float> third_eigen_value;
            PtrStep<int> indices;
            int min_neighboors;
            const int* sizes;

            PtrSz<bool> is_keypoint;

            __device__ __forceinline__ void operator()() const
            {
                __shared__ bool is_max[CTA_SIZE + 1];

                int warp_idx = Warp::id();
                int idx = blockIdx.x * WAPRS + warp_idx;

                if (idx >= third_eigen_value.size) return;

                int size = sizes[idx];
                int lane = Warp::laneId();

                if (third_eigen_value.data[idx] < 0.0)
                {
                    if (lane == 0)
                        is_keypoint.data[idx] = false;
                    return;
                }

                int tid = threadIdx.x;

                is_max[tid] = true;

                const int* ibeg = indices.ptr(idx);
                const int* iend = ibeg + size;

                for (const int* t = ibeg + lane; t < iend; t += Warp::STRIDE)
                {
                    if (third_eigen_value.data[idx] < third_eigen_value.data[*t]) {
                        is_max[tid] = false;
                        break;
                    }
                }

                Warp::reduce(&is_max[tid - lane], logic_and());

                if (lane == 0) {
                    is_keypoint.data[idx] = is_max[tid - lane];
                }
            }
        };

        __global__ void ThirdEigenValueCalculatorKernel(const ThirdEigenValueCalculator mevc) { mevc(); }
        __global__ void NonMaxSuppressorKernel(const NonMaxSuppressor nms) { nms(); }
    }
}

void pcl::device::detectISSKeypoint3D(
    const PointCloud& cloud, const int min_neighboors,
    const float threshold21, const float threshold32,
    const NeighborIndices& nn_indices,   // NeighborIndices for calculate max eigen value of scatter matrix
    const NeighborIndices& nn_indices2,  // NeighborIndices for non max suppress/detect keypoints
    PointCloud& keypoints, Indices& keypoints_indices)
{
    // Step 1. calculate max eigen value of each point
    DeviceArray<float> third_eigen_value;
    third_eigen_value.create(nn_indices.neighboors_size());

    ThirdEigenValueCalculator mevc;
    mevc.min_neighboors = min_neighboors;
    mevc.threshold21 = threshold21;
    mevc.threshold32 = threshold32;
    mevc.indices = nn_indices;  // convert NeighborIndices to PtrStep<int>
    mevc.sizes = nn_indices.sizes;
    mevc.points = cloud;
    mevc.third_eigen_value = third_eigen_value;

    int block = ThirdEigenValueCalculator::CTA_SIZE;    // number of threads in each block
    int grid = divUp((int)third_eigen_value.size(), ThirdEigenValueCalculator::WAPRS);    // number of blocks in each grid
    ThirdEigenValueCalculatorKernel<<<grid, block>>>(mevc);

    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());

    // Step 2. non-maximum suppression for each point
    DeviceArray<bool> is_keypoint;
    is_keypoint.create(nn_indices2.neighboors_size());

    NonMaxSuppressor nms;
    nms.min_neighboors = min_neighboors;
    nms.indices = nn_indices2;
    nms.sizes = nn_indices2.sizes;
    nms.third_eigen_value = third_eigen_value;
    nms.is_keypoint = is_keypoint;

    block = NonMaxSuppressor::CTA_SIZE;
    grid = divUp((int)is_keypoint.size(), NonMaxSuppressor::WAPRS);
    NonMaxSuppressorKernel<<<grid, block>>>(nms);

    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());

    // Finally. copy results
    thrust::device_ptr<bool> is_keypoint_ptr(is_keypoint.ptr());
    thrust::device_ptr<const PointType> cloud_ptr((const PointType*)cloud.ptr());

    int count = thrust::count(is_keypoint_ptr, is_keypoint_ptr + is_keypoint.size(), true);
    keypoints.create(count);
    keypoints_indices.create(count);

    thrust::device_ptr<PointType> keypoints_ptr(keypoints.ptr());
    thrust::device_ptr<int> keypoints_indices_ptr(keypoints_indices.ptr());

    thrust::copy_if(cloud_ptr, cloud_ptr + cloud.size(), is_keypoint_ptr, keypoints_ptr, thrust::identity<bool>());
    thrust::copy_if(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(cloud.size()), is_keypoint_ptr, keypoints_indices_ptr, thrust::identity<bool>());
}
