if(BUILD_GPU_TESTS)
    find_package(Gtest REQUIRED)
    include_directories(SYSTEM ${GTEST_INCLUDE_DIRS} ${GTEST_SRC_DIR})
    
    set(the_test_target test_gpu_keypoints)

    enable_testing()
    include_directories(${PCL_INCLUDE_DIRS})
    
    FILE(GLOB test_src *.cpp *.hpp)
    PCL_ADD_TEST(gpu_keypoints_test ${the_test_target} FILES ${test_src} LINK_WITH pcl_gtest pcl_gpu_containers pcl_gpu_octree pcl_gpu_keypoints pcl_io pcl_features pcl_visualization pcl_common ARGUMENTS "${PCL_SOURCE_DIR}/test/office_chair_model.pcd")
endif(BUILD_GPU_TESTS)
