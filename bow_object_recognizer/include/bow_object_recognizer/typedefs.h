#ifndef TYPEDEFS_H
#define TYPEDEFS_H

// Type definitions for Point Cloud
using PointType = pcl::PointXYZRGB;
using PointTypeNoColor = pcl::PointXYZI;
using PointCloud = pcl::PointCloud<PointType>;
using PointCloudNoColor = pcl::PointCloud<PointTypeNoColor>;
using PointCloudPtr = PointCloud::Ptr;
using PointCloudNoColorPtr = PointCloudNoColor::Ptr;
using PointCloudConstPtr = PointCloud::ConstPtr;

// Type definitions for Normals
using NormalType = pcl::Normal;
using SurfaceNormals = pcl::PointCloud<NormalType>;
using SurfaceNormalsPtr = SurfaceNormals::Ptr;

// Misc
using DistType = flann::ChiSquareDistance<float>;
using BoWDescriptorPoint = std::vector<float>;
using BoWDescriptor = std::vector<float>;
using bow_model_sample = std::pair<string, std::vector<float>>;

// KdTree
using KdTree = pcl::search::KdTree<PointType>;
using KdTreePtr = pcl::search::KdTree<PointType>::Ptr;

#endif // TYPEDEFS_H