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
*   * Iss_Detectorither the name of Willow Garage, Inc. nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITISS_DETECTORSS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWISS_DETECTORR OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSIISS_DETECTORSS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING ISS_DETECTORGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*
*  Author: Anatoly Baskeheev, Itseez Ltd, (myname.mysurname@mycompany.com)
*/

#if (defined(__GNUC__) && !defined(__CUDACC__) && (GTEST_GCC_VER_ >= 40000))
  #define GTEST_USE_OWN_TR1_TUPLE 0
#endif

#if defined(_MSC_VER) && (_MSC_VER >= 1500)
  #define GTEST_USE_OWN_TR1_TUPLE 0
#endif

#include <gtest/gtest.h>
#include <iostream>
#include "data_source.hpp"

#include <pcl/gpu/keypoints/keypoints.hpp>
#include <pcl/gpu/containers/initialization.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/search/search.h>

using namespace std;
using namespace pcl;
using namespace pcl::gpu;

const char* filepath;

// TEST(PCL_FeaturesGPU, DISABLED_keypoints_highlevel_1)
TEST(PCL_KeypointsGPU, isskeypoint3d_highlevel)
{
    DataSource source(filepath);
    cout << "Cloud size: " << source.cloud->points.size() << endl;
    cout << "Salient Radius: " << source.salient_radius << endl;
    cout << "Non Max Radius: " << source.non_max_radius << endl;
    cout << "Max_elems: " << source.max_elements << endl;

    // source.runCloudViewer();

    pcl::ISSKeypoint3D<PointXYZ, PointXYZ> iss_detector;
    iss_detector.setNumberOfThreads(1);
    iss_detector.setInputCloud(source.cloud);
    iss_detector.setSearchMethod (pcl::search::KdTree<PointXYZ>::Ptr (new pcl::search::KdTree<PointXYZ>));
    iss_detector.setSalientRadius(source.salient_radius);
    iss_detector.setNonMaxRadius(source.non_max_radius);

    PointCloud<PointXYZ>::Ptr keypoints(new PointCloud<PointXYZ>());
    iss_detector.compute(*keypoints);

    pcl::gpu::ISSKeypoint3D::PointCloud cloud_device;
    cloud_device.upload(source.cloud->points);

    pcl::gpu::ISSKeypoint3D iss_detector_device;
    iss_detector_device.setInputCloud(cloud_device);
    iss_detector_device.setRadiusSearch(source.salient_radius, source.max_elements);
    iss_detector_device.setSalientRadius(source.salient_radius);
    iss_detector_device.setNonMaxRadius(source.non_max_radius);

    pcl::gpu::ISSKeypoint3D::PointCloud keypoints_device;
    iss_detector_device.compute(keypoints_device);

    vector<PointXYZ> downloaded;
    keypoints_device.download(downloaded);

    for (size_t i = 0; i < downloaded.size(); ++i)
    {
        PointXYZ cpu = keypoints->points[i];

        PointXYZ gpu = downloaded[i];

        float abs_error = 0.01f;
        ASSERT_NEAR(cpu.x, gpu.x, abs_error);
        ASSERT_NEAR(cpu.y, gpu.y, abs_error);
        ASSERT_NEAR(cpu.z, gpu.z, abs_error);
    }
}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        cerr << "No test file given. Please download `office_chair_model.pcd` and pass its path to the test." << endl;
        return (-1);
    }

    filepath = argv[1];

    pcl::gpu::setDevice(0);
    pcl::gpu::printShortCudaDeviceInfo(0);
    testing::InitGoogleTest(&argc, argv);
    return (RUN_ALL_TESTS());
}
