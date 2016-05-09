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

#ifndef _PCL_GPU_KEYPOINTS_HPP_
#define _PCL_GPU_KEYPOINTS_HPP_

#include <pcl/gpu/containers/device_array.h>
#include <pcl/pcl_macros.h>
#include <pcl/point_types.h>
#include <pcl/gpu/octree/device_format.hpp>
#include <pcl/gpu/octree/octree.hpp>

namespace pcl
{
    namespace gpu
    {
        ////////////////////////////////////////////////////////////////////////////////////////////
        /** \brief @b Keypoints represents the base keypoints class.  */

        struct PCL_EXPORTS Keypoints
        {
          public:
            typedef PointXYZ PointType;
            typedef PointXYZ NormalType;

            typedef DeviceArray<PointType> PointCloud;
            typedef DeviceArray<NormalType> Normals;

            typedef DeviceArray<int> Indices;
            typedef DeviceArray<bool> BorderPoints;

            Keypoints();

            void setInputCloud(const PointCloud &cloud);
            void setSearchSurface(const PointCloud &surface);
            void setIndices(const Indices &indices);
            void setRadiusSearch(float radius, int max_results);

          protected:
            PointCloud cloud_;
            PointCloud surface_;
            Indices indices_;
            float radius_;
            int max_results_;

            Octree octree_;
        };

        struct PCL_EXPORTS ISSKeypoint3D : Keypoints
        {
          public:
            ISSKeypoint3D(double salient_radius = 0.0001);

            void setSalientRadius(double salient_radius);
            void setNonMaxRadius(double non_max_radius);
            void setThreshold21(double gamma_21);
            void setThreshold32(double gamma_32);
            void setMinNeighbors(double min_neighbors);
            void setBorderPoints(const EdgePoints &edge_points);

            void detectPoints(PointCloud &output);  // TODO: 寻找合适的输出形式
          private:
            NeighborIndices nn_indices_, nn_indices2_;

            BorderPoints border_points_;

            float salient_radius_;
            float non_max_radius_;
            float border_radius_;

            float gamma_21_;
            float gamma_32_;

            int min_neighbors_;
        };
    };
};

#endif /* _PCL_GPU_KEYPOINTS_HPP */
