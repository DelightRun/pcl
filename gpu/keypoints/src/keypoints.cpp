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

#include <pcl/exceptions.h>
#include <pcl/console/print.h>

using namespace pcl::device;

/////////////////////////////////////////////////////////////////////////
//// Keypoints

pcl::gpu::Keypoints::Keypoints() { radius_ = 0.f, max_results_ = 0; }
void pcl::gpu::Keypoints::setInputCloud(const PointCloud& cloud) { cloud_ = cloud; }
void pcl::gpu::Keypoints::setSearchSurface(const PointCloud& surface) { surface_ = surface; }
void pcl::gpu::Keypoints::setIndices(const Indices& indices) { indices_ = indices; }
void pcl::gpu::Keypoints::setRadiusSearch(float radius, int max_results) { radius_ = radius; max_results_ = max_results; }

/////////////////////////////////////////////////////////////////////////
//// ISSKeypoint3D

pcl::gpu::ISSKeypoint3D::ISSKeypoint3D(double salient_radius = 0.0001)
	: salient_radius_(salient_radius)
	, non_max_radius_(0.0)
	, normal_radius_(0.0)
	, border_radius_(0.0)
	, gamma_21_(0.975)
	, gamma_32_(0.975)
	, min_neighbors_(5)
	, angle_threshold_(static_cast<float> (M_PI) / 2.0f)
{
	radius_ = salient_radius_;
}

void pcl::gpu::ISSKeypoint3D::setSalientRadius(double salient_radius) { salient_radius_ = salient_radius; }
void pcl::gpu::ISSKeypoint3D::setNonMaxRadius(double non_max_radius) { non_max_radius_ = non_max_radius; }
void pcl::gpu::ISSKeypoint3D::setNormalRadius(double normal_radius) { normal_radius_ = normal_radius; }
void pcl::gpu::ISSKeypoint3D::setBorderRadius(double border_radius) { border_radius_ = border_radius; }
void pcl::gpu::ISSKeypoint3D::setThreshold21(double gamma_21) { gamma_21_ = gamma_21; }
void pcl::gpu::ISSKeypoint3D::setThreshold32(double gamma_32) { gamma_32_ = gamma_32; }
// TODO