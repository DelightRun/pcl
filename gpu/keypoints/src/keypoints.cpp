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

#include <pcl/gpu/containers/initialization.h>
#include <pcl/gpu/keypoints/keypoints.hpp>
#include <pcl/gpu/utils/device/static_check.hpp>
#include "internal.hpp"

#include <pcl/console/print.h>
#include <pcl/exceptions.h>

#include <iostream>

using namespace pcl::device;

/////////////////////////////////////////////////////////////////////////
//// Keypoints

pcl::gpu::Keypoints::Keypoints() { radius_ = 0.f, max_results_ = 0; }
void pcl::gpu::Keypoints::setInputCloud(const PointCloud &cloud) { cloud_ = cloud; }
void pcl::gpu::Keypoints::setSearchSurface(const PointCloud &surface) { surface_ = surface; }
void pcl::gpu::Keypoints::setIndices(const Indices &indices) { indices_ = indices; }
void pcl::gpu::Keypoints::setRadiusSearch(float radius, int max_results)
{
    radius_ = radius;
    max_results_ = max_results;
}
void pcl::gpu::Keypoints::compute(PointCloud &output) { detectKeypoints(output); }

/////////////////////////////////////////////////////////////////////////
//// ISSKeypoint3D

pcl::gpu::ISSKeypoint3D::ISSKeypoint3D(double salient_radius)
    : salient_radius_(salient_radius),
      non_max_radius_(0.0),
      gamma_21_(0.975),
      gamma_32_(0.975),
      min_neighbors_(5)
{
    radius_ = salient_radius_;
}

void pcl::gpu::ISSKeypoint3D::setSalientRadius(double salient_radius) { salient_radius_ = salient_radius; }
void pcl::gpu::ISSKeypoint3D::setNonMaxRadius(double non_max_radius) { non_max_radius_ = non_max_radius; }
void pcl::gpu::ISSKeypoint3D::setThreshold21(double gamma_21) { gamma_21_ = gamma_21; }
void pcl::gpu::ISSKeypoint3D::setThreshold32(double gamma_32) { gamma_32_ = gamma_32; }
void pcl::gpu::ISSKeypoint3D::setMinNeighbors(double min_neighbors) { min_neighbors_ = min_neighbors; }

void pcl::gpu::ISSKeypoint3D::detectKeypoints(PointCloud& output)
{
    PointCloud& surface = surface_.empty() ? cloud_ : surface_;

    octree_.setCloud(surface);
    octree_.build();

    if (indices_.empty() || (!indices_.empty() && indices_.size() == cloud_.size()))
    {
        octree_.radiusSearch(cloud_, salient_radius_, max_results_, nn_indices_);
        octree_.radiusSearch(cloud_, non_max_radius_, max_results_, nn_indices2_);
    }
    else
    {
        octree_.radiusSearch(cloud_, indices_, salient_radius_, max_results_, nn_indices_);
        octree_.radiusSearch(cloud_, indices_, non_max_radius_, max_results_, nn_indices2_);
    }

    const device::PointCloud& c = (const device::PointCloud&)cloud_;
    device::PointCloud& o = (device::PointCloud&)output;

    device::detectISSKeypoint3D(c, min_neighbors_, gamma_21_, gamma_32_, nn_indices_, nn_indices2_, o);
}
