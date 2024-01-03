#ifndef FEATURE_ESTIMATOR_H
#define FEATURE_ESTIMATOR_H

#include <string>
#include <utility>
#include <vector>
#include <iostream>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/console/print.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/pfhrgb.h>
#include <pcl/features/shot.h>
#include <pcl/kdtree/kdtree_flann.h>
#include "typedefs.h"

template <typename PointType>
class FeatureEstimator
{
protected:
    float support_radius_;
    pcl::PointIndices::Ptr nan_indices_;

public:
    using Ptr = boost::shared_ptr<FeatureEstimator<PointType>>;

    FeatureEstimator() {}

    virtual void calculateFeatures(
        PointCloudPtr &in,
        PointCloudPtr &keypoints,
        SurfaceNormalsPtr &normals,
        std::vector<BoWDescriptorPoint> &features) = 0;

    virtual void saveFeatures(std::vector<BoWDescriptorPoint> &features, std::string path) = 0;

    virtual void loadFeatures(std::string path, std::vector<BoWDescriptorPoint> &features) = 0;

    ~FeatureEstimator() {}

    inline void setSupportRadius(float r)
    {
        support_radius_ = r;
    }

    inline pcl::PointIndices::Ptr getNaNIndices()
    {
        return nan_indices_;
    }

    std::string name;

    int dimensionality;
};

template <typename PointType>
class FeatureEstimatorSHOT : public FeatureEstimator<PointType>
{
    using FeatureEstimator<PointType>::dimensionality;
    using FeatureEstimator<PointType>::support_radius_;
    using FeatureEstimator<PointType>::nan_indices_;

protected:
    using DescriptorType = pcl::SHOT352;
    using DescriptorCloudPtr = pcl::PointCloud<DescriptorType>::Ptr;
    using DescriptorEstimator = FeatureEstimatorSHOT<PointType>;

public:
    using Ptr = boost::shared_ptr<DescriptorEstimator>;

    FeatureEstimatorSHOT()
    {
        dimensionality = 352;
        nan_indices_ = pcl::PointIndices::Ptr(new pcl::PointIndices);
    }

    double computeCloudResolution(const PointCloudConstPtr &cloud)
    {
        double resolution = 0.0;
        int number_of_points = 0;
        int nres;
        std::vector<int> indices(2);
        std::vector<float> squared_distances(2);
        KdTree tree;
        tree.setInputCloud(cloud);

        for (size_t i = 0; i < cloud->size(); ++i)
        {
            if (!pcl_isfinite((*cloud)[i].x))
                continue;

            // Considering the second neighbor since the first is the point itself.
            nres = tree.nearestKSearch(i, 2, indices, squared_distances);
            if (nres == 2)
            {
                resolution += sqrt(squared_distances[1]);
                ++number_of_points;
            }
        }
        if (number_of_points != 0)
            resolution /= number_of_points;

        return resolution;
    }

    void calculateFeatures(
        PointCloudPtr &in,
        PointCloudPtr &keypoints,
        SurfaceNormalsPtr &normals,
        std::vector<BoWDescriptorPoint> &features) noexcept
    {
        std::cout << "Calculate features SHOT ...\n";

        DescriptorCloudPtr shots(new pcl::PointCloud<DescriptorType>());

        double resolution = computeCloudResolution(in);

        pcl::SHOTEstimationOMP<PointType, NormalType, DescriptorType> shot_estimate;
        shot_estimate.setInputCloud(keypoints);
        shot_estimate.setRadiusSearch(support_radius_);

        shot_estimate.setInputNormals(normals);
        shot_estimate.setSearchSurface(in);
        shot_estimate.setNumberOfThreads(8);
        shot_estimate.compute(*shots);

        PCL_INFO("SHOT descriptors has %d points\n\n", static_cast<int>(shots->points.size()));

        features.reserve(shots->size());

        // Preprocess features: remove NaNs
        for (size_t j = 0; j < shots->points.size(); j++)
        {
            BoWDescriptorPoint descriptor_vector(dimensionality);

            if (!pcl_isfinite(shots->at(j).descriptor[0]))
            {
                nan_indices_->indices.push_back(j);
                continue;
            }

            for (int idx = 0; idx < dimensionality; idx++)
            {
                descriptor_vector[idx] = shots->points[j].descriptor[idx];
            }

            features.push_back(std::move(descriptor_vector));
        }

        PCL_INFO("SHOT descriptors has %d points after NaN removal\n\n", static_cast<int>(features.size()));

        std::cout << "[calculateFeatures] NaNs in scene keypoints: " << nan_indices_->indices.size() << "\n";
    }

    void saveFeatures(std::vector<BoWDescriptorPoint> &features, std::string path)
    {
        DescriptorCloudPtr features_cloud(new pcl::PointCloud<DescriptorType>);
        features_cloud->resize(features.size());

        size_t j = 0;
        std::for_each(features.begin(), features.end(), [&j, &features_cloud, &dimensionality](BoWDescriptorPoint &feature)
                      {
            for(int idx = 0; idx < dimensionality; idx++)
            {
                features_cloud->points[j].descriptor[idx] = feature[idx];
            }
            j++; });

        pcl::io::savePCDFileASCII(path.c_str(), *features_cloud);

        std::cout << "Features were saved to file: " << path << "\n";
    }

    void loadFeatures(std::string path, std::vector<BoWDescriptorPoint> &features) noexcept
    {
        DescriptorCloudPtr features_cloud(new pcl::PointCloud<DescriptorType>);

        // Load descriptors from the file
        pcl::io::loadPCDFile(path.c_str(), *features_cloud);

        PCL_INFO("Descriptor %s has size: %d\n", path.c_str(), features_cloud->size());

        features.reserve(features_cloud->size());

        for (size_t j = 0; j < features_cloud->points.size(); j++)
        {
            BoWDescriptorPoint descriptor_vector(dimensionality);

            if (!pcl::isFinite<DescriptorType>(features_cloud->points[j]))
                PCL_WARN("Point %d is NaN\n", static_cast<int>(j));

            for (int idx = 0; idx < dimensionality; idx++)
            {
                descriptor_vector[idx] = features_cloud->points[j].descriptor[idx];
            }

            features.emplace(std::move(descriptor_vector));
        }
    }

    ~FeatureEstimatorSHOT() {}
};

template <typename PointType>
class FeatureEstimatorFPFH : public FeatureEstimator<PointType>
{
    using FeatureEstimator<PointType>::dimensionality;
    using FeatureEstimator<PointType>::support_radius_;
    using FeatureEstimator<PointType>::nan_indices_;

protected:
    using DescriptorType pcl::FPFHSignature33;
    using DescriptorCloud = pcl::PointCloud<DescriptorType>::Ptr;
    using DescriptorCloudPtr = pcl::PointCloud<DescriptorType>::Ptr;
    using DescriptorEstimator = FeatureEstimatorFPFH<PointType>;

public:
    using Ptr = boost::shared_ptr<DescriptorEstimator>;

    FeatureEstimatorFPFH()
    {
        dimensionality = 33;
        nan_indices_ = pcl::PointIndices::Ptr(new pcl::PointIndices);
    }

    void calculateFeatures(
        PointCloudPtr &in,
        PointCloudPtr &keypoints,
        SurfaceNormalsPtr &normals,
        std::vector<BoWDescriptorPoint> &features) noexcept
    {
        std::cout << "Calculate features FPFH ...\n";

        for (size_t i = 0; i < in->points.size(); i++)
        {
            if (!pcl::isFinite<PointType>(in->points[i]))
                PCL_WARN("Point %d is NaN\n", static_cast<int>(i));
        }

        DescriptorCloudPtr fpfhs(new pcl::PointCloud<DescriptorType>());

        pcl::FPFHEstimationOMP<PointType, NormalType, DescriptorType> fpfh_estimate;
        fpfh_estimate.setInputCloud(keypoints);
        // It was commented
        fpfh_estimate.setRadiusSearch(support_radius_);

        // begin -- from Semantic localization project
        KdTreePtr tree(new KdTree());
        fpfh_estimate.setSearchMethod(tree);
        // It was uncommented
        //        fpfh_estimate.setKSearch(50);
        // end -- from Semantic localization project
        fpfh_estimate.setInputNormals(normals);
        fpfh_estimate.setSearchSurface(in);
        fpfh_estimate.setNumberOfThreads(8);
        fpfh_estimate.compute(*fpfhs);

        features.reserve(fpfhs->size());

        for (size_t j = 0; j < fpfhs->points.size(); j++)
        {
            BoWDescriptorPoint descriptor_vector(dimensionality);

            for (int idx = 0; idx < dimensionality; idx++)
            {
                descriptor_vector[idx] = fpfhs->points[j].histogram[idx];
            }

            features.push_back(std::move(descriptor_vector));
        }
    }

    void saveFeatures(std::vector<BoWDescriptorPoint> &features, std::string path)
    {
        DescriptorCloudPtr features_cloud(new pcl::PointCloud<DescriptorType>);
        features_cloud->resize(features.size());

        size_t j = 0;
        std::for_each(features.begin(), features.end(), [&j, &features_cloud, &dimensionality](BoWDescriptorPoint &feature)
                      {
            for(int idx = 0; idx < dimensionality; idx++)
            {
                features_cloud->points[j].descriptor[idx] = feature[idx];
            }
            j++; });

        pcl::io::savePCDFileASCII(path.c_str(), *features_cloud);

        std::cout << "Features were saved to file: " << path << "\n";
    }

    void loadFeatures(std::string path, std::vector<BoWDescriptorPoint> &features)
    {
        DescriptorCloudPtr features_cloud(new pcl::PointCloud<DescriptorType>);

        // Load descriptors from the file
        pcl::io::loadPCDFile(path.c_str(), *features_cloud);

        std::cout << "Descriptor was loaded from file " << path << "\n";

        features.reserve(features_cloud->size());

        for (size_t j = 0; j < features_cloud->points.size(); j++)
        {
            BoWDescriptorPoint descriptor_vector(dimensionality);

            if (!pcl::isFinite<DescriptorType>(features_cloud->points[j]))
                PCL_WARN("Point %d is NaN\n", static_cast<int>(j));

            for (int idx = 0; idx < dimensionality; idx++)
            {
                descriptor_vector[idx] = features_cloud->points[j].histogram[idx];
            }

            features.push_back(std::move(descriptor_vector));
        }
    }

    ~FeatureEstimatorFPFH() {}
};

template <typename PointType>
class FeatureEstimatorPFHRGB : public FeatureEstimator<PointType>
{
    using FeatureEstimator<PointType>::dimensionality;
    using FeatureEstimator<PointType>::support_radius_;
    using FeatureEstimator<PointType>::nan_indices_;
    int k_search_;

protected:
    using DescriptorType = pcl::PFHRGBSignature250;
    using DescriptorCloudPtr = pcl::PointCloud<DescriptorType>::Ptr;
    using DescriptorEstimator = FeatureEstimatorPFHRGB<PointType>;

public:
    using Ptr = boost::shared_ptr<DescriptorEstimator>;

    FeatureEstimatorPFHRGB()
    {
        dimensionality = 250;
        nan_indices_ = pcl::PointIndices::Ptr(new pcl::PointIndices);
    }

    void setKSearch(int k)
    {
        k_search_ = k;
    }

    void calculateFeatures(
        PointCloudPtr &in,
        PointCloudPtr &keypoints,
        SurfaceNormalsPtr &normals,
        std::vector<BoWDescriptorPoint> &features) noexcept
    {
        DescriptorCloudPtr pfhrgbs(new pcl::PointCloud<DescriptorType>());

        KdTreePtr tree(new KdTree);

        pcl::PFHRGBEstimation<PointType, NormalType, DescriptorType> pfhrgb_estimation;
        pfhrgb_estimation.setInputCloud(keypoints);
        pfhrgb_estimation.setInputNormals(normals);
        pfhrgb_estimation.setSearchSurface(in);
        pfhrgb_estimation.setSearchMethod(tree);
        pfhrgb_estimation.setRadiusSearch(support_radius_);
        pfhrgb_estimation.compute(*pfhrgbs);

        features.reserve(pfhrgbs->size());

        for (size_t j = 0; j < pfhrgbs->points.size(); j++)
        {
            BoWDescriptorPoint descriptor_vector(dimensionality);

            for (int idx = 0; idx < dimensionality; idx++)
            {
                descriptor_vector[idx] = pfhrgbs->points[j].histogram[idx];
            }

            features.push_back(std::move(descriptor_vector));
        }
    }

    void saveFeatures(std::vector<BoWDescriptorPoint> &features, std::string path)
    {
        DescriptorCloudPtr features_cloud(new pcl::PointCloud<DescriptorType>);
        features_cloud->resize(features.size());

        size_t j = 0;
        std::for_each(features.begin(), features.end(), [&j, &features_cloud, &dimensionality](BoWDescriptorPoint &feature)
                      {
            for(int idx = 0; idx < dimensionality; idx++)
            {
                features_cloud->points[j].descriptor[idx] = feature[idx];
            }
            j++; });

        pcl::io::savePCDFileASCII(path.c_str(), *features_cloud);

        std::cout << "Features were saved to file: " << path << "\n";
    }

    void loadFeatures(std::string path, std::vector<BoWDescriptorPoint> &features)
    {
        DescriptorCloudPtr features_cloud(new pcl::PointCloud<DescriptorType>);

        // Load descriptors from the file
        pcl::io::loadPCDFile(path.c_str(), *features_cloud);

        std::cout << "Descriptor was loaded from file " << path << "\n";

        features.reserve(features_cloud->size());

        for (size_t j = 0; j < features_cloud->points.size(); j++)
        {
            BoWDescriptorPoint descriptor_vector(dimensionality);

            if (!pcl::isFinite<DescriptorType>(features_cloud->points[j]))
                PCL_WARN("Point %d is NaN\n", static_cast<int>(j));

            for (int idx = 0; idx < dimensionality; idx++)
            {
                descriptor_vector[idx] = features_cloud->points[j].histogram[idx];
            }

            features.push_back(std::move(descriptor_vector));
        }
    }

    ~FeatureEstimatorPFHRGB() {}
};

template <typename PointType>
class FeatureEstimatorColorSHOT : public FeatureEstimator<PointType>
{
    using FeatureEstimator<PointType>::dimensionality;
    using FeatureEstimator<PointType>::support_radius_;
    using FeatureEstimator<PointType>::nan_indices_;

protected:
    using DescriptorType = pcl::SHOT1344;
    using DescriptorCloudPtr = pcl::PointCloud<FeatureType>::Ptr;
    using DescriptorEstimator = FeatureEstimatorColorSHOT<PointType>;

public:
    using Ptr = boost::shared_ptr<DescriptorEstimator>;

    FeatureEstimatorColorSHOT()
    {
        dimensionality = 1344;
        nan_indices_ = pcl::PointIndices::Ptr(new pcl::PointIndices);
    }

    void calculateFeatures(
        PointCloudPtr &in,
        PointCloudPtr &keypoints,
        SurfaceNormalsPtr &normals,
        std::vector<BoWDescriptorPoint> &features) noexcept
    {
        DescriptorCloudPtr cshots(new pcl::PointCloud<DescriptorType>());

        pcl::SHOTColorEstimation<PointType, NormalType> shotestimator;
        shotestimator.setInputCloud(keypoints);
        shotestimator.setInputNormals(normals);
        // computes the pointcloud resolution
        // sets the radius to three times the resolution
        shotestimator.setRadiusSearch(support_radius_);
        shotestimator.setSearchSurface(in);
        shotestimator.compute(*cshots);

        features.reserve(cshots->size());

        for (size_t j = 0; j < cshots->points.size(); j++)
        {
            BoWDescriptorPoint descriptor_vector(dimensionality);

            for (int idx = 0; idx < dimensionality; idx++)
            {
                descriptor_vector[idx] = cshots->points[j].descriptor[idx];
            }

            features.push_back(std::move(descriptor_vector));
        }
    }

    void saveFeatures(std::vector<BoWDescriptorPoint> &features, std::string path)
    {
        DescriptorCloudPtr features_cloud(new pcl::PointCloud<DescriptorType>);
        features_cloud->resize(features.size());

        size_t j = 0;
        std::for_each(features.begin(), features.end(), [&j, &features_cloud, &dimensionality](BoWDescriptorPoint &feature)
                      {
            for(int idx = 0; idx < dimensionality; idx++)
            {
                features_cloud->points[j].descriptor[idx] = feature[idx];
            }
            j++; });

        pcl::io::savePCDFileASCII(path.c_str(), *features_cloud);

        std::cout << "Features were saved to file: " << path << "\n";
    }

    void loadFeatures(std::string path, std::vector<BoWDescriptorPoint> &features)
    {
        DescriptorCloudPtr features_cloud(new pcl::PointCloud<DescriptorType>);

        // Load descriptors from the file
        pcl::io::loadPCDFile(path.c_str(), *features_cloud);

        std::cout << "Descriptor was loaded from file " << path << "\n";

        features.reserve(features_cloud->size());

        for (size_t j = 0; j < features_cloud->points.size(); j++)
        {
            BoWDescriptorPoint descriptor_vector(dimensionality);

            if (!pcl::isFinite<DescriptorCloud>(features_cloud->points[j]))
                PCL_WARN("Point %d is NaN\n", static_cast<int>(j));

            for (int idx = 0; idx < dimensionality; idx++)
            {
                descriptor_vector[idx] = features_cloud->points[j].descriptor[idx];
            }

            features.push_back(std::move(descriptor_vector));
        }
    }

    ~FeatureEstimatorColorSHOT() {}
};

#endif // FEATURE_ESTIMATOR_H
