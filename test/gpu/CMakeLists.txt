PCL_ADD_TEST(gpu_keypoints_iss_3d test_gpu_iss_3d
              FILES test_gpu_iss_3d.cpp
              LINK_WITH pcl_gtest pcl_gpu_containers pcl_gpu_octree pcl_gpu_keypoints pcl_io pcl_features pcl_visualization pcl_common
              ARGUMENTS "${PCL_SOURCE_DIR}/test/office_chair_model.pcd")
