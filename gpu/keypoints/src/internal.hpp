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

#ifndef PCL_GPU_FEATURES_INTERNAL_HPP_
#define PCL_GPU_FEATURES_INTERNAL_HPP_

#include <pcl/gpu/containers/device_array.h>
#include <pcl/gpu/octree/device_format.hpp>

#include <cuda_runtime.h>

#undef PI
#ifndef PI
#define PI 3.1415926535897931f
#endif

namespace pcl
{
    namespace device
    {
        using pcl::gpu::DeviceArray;
        using pcl::gpu::DeviceArray2D;
        using pcl::gpu::NeighborIndices;

        typedef float4 PointType;
        typedef float4 NormalType;
        typedef float4 PointXYZRGB;

        typedef DeviceArray<PointType> PointCloud;
        typedef DeviceArray<int> Indices;

        typedef DeviceArray<PointType> PointXYZRGBCloud;

        // ISS keypoints estimation
        void detectISSKeypoint3D(const PointCloud& cloud,
            const int min_neighboors,
            const float threshold21, const float threshold32,
            const NeighborIndices& nn_indices,   // NeighborIndices for calculate max eigen value of scatter matrix
            const NeighborIndices& nn_indices2,  // NeighborIndices for non max suppress/detect keypoints
            PointCloud& keypoints, Indices& keypoints_indices);
    }
}

#endif /* PCL_GPU_FEATURES_INTERNAL_HPP_ */
