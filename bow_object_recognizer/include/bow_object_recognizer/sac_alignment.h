#define USE_RANSAC_ALIGNMENT

#ifndef RANSAC_ALIGNMENT_H
#define RANSAC_ALIGNMENT_H

#define SAC_ALIGNMENT_CLASS_ACTIVE

#include <iostream>
#include <vector>
#include <Eigen/Core>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/registration/ia_ransac.h>
#include "typedefs.h"

#ifdef USE_RANSAC_ALIGNMENT
template <typename FeatureInT>
class FeatureCloud
{
public:
    // A bit of shorthand
    using DescriptorCloudIn = pcl::PointCloud<FeatureInT>;
    using DescriptorCloudInPtr = pcl::PointCloud<FeatureInT>::Ptr;

    FeatureCloud() {}

    ~FeatureCloud() {}

    void
    setId(std::string id)
    {
        id_ = id;
    }

    // Process the given cloud
    void
    setInputCloud(PointCloudPtr xyz)
    {
        xyz_ = xyz;
    }

    // Load and process the cloud in the given PCD file
    void
    loadInputCloud(const std::string &pcd_file)
    {
        xyz_ = PointCloudPtr(new PointCloud);
        pcl::io::loadPCDFile(pcd_file, *xyz_);
    }

    void
    setInputFeatures(DescriptorCloudInPtr features)
    {
        features_ = features;
    }

    // Load and process the feature cloud in the given PCD file
    void
    loadInputFeatures(const std::string &pcd_file)
    {
        features_ = DescriptorCloudInPtr(new DescriptorCloudIn);
        pcl::io::loadPCDFile(pcd_file, *features_);
    }

    void setLocalFeatures(DescriptorCloudInPtr) {}

    std::string
    getId()
    {
        return id_;
    }

    // Get a pointer to the cloud 3D points
    PointCloudPtr
    getPointCloud() const
    {
        return (xyz_);
    }

    // Get a pointer to the cloud of feature descriptors
    DescriptorCloudInPtr
    getLocalFeatures() const
    {
        return (features_);
    }

private:
    // Point cloud data
    PointCloudPtr xyz_;
    DescriptorCloudInPtr features_;
    std::string id_;
    pcl::PointIndices::Ptr nan_indices_;
};

#ifdef SAC_ALIGNMENT_CLASS_ACTIVE

template <typename FeatureInT>
class SACAlignment
{
public:
    // A struct for storing alignment results
    struct Result
    {
        float fitness_score;
        Eigen::Matrix4f final_transformation;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

    SACAlignment() : min_sample_distance_(0.05f),
                     max_correspondence_distance_(0.01f * 0.01f),
                     nr_iterations_(500) noexcept
    {
        // Intialize the parameters in the Sample Consensus Intial Alignment (SAC-IA) algorithm
        sac_ia_.setMinSampleDistance(min_sample_distance_);
        sac_ia_.setMaxCorrespondenceDistance(max_correspondence_distance_);
        sac_ia_.setMaximumIterations(nr_iterations_);
    }

    ~SACAlignment() {}

    // Set the given cloud as the target to which the templates will be aligned
    void
    setTargetCloud(FeatureCloud<FeatureInT> &target_cloud)
    {
        std::cout << "[SACAlignment:setTargetCloud] Set target cloud to SAC_IA\n";

        target_ = target_cloud;
        sac_ia_.setInputTarget(target_cloud.getPointCloud());
        PCL_INFO("Keypoints cloud has been added to SAC-IA\n");
        sac_ia_.setTargetFeatures(target_cloud.getLocalFeatures());
        PCL_INFO("Features cloud has been added to SAC-IA\n");
    }

    // Add the given cloud to the list of template clouds
    void
    addTemplateCloud(FeatureCloud<FeatureInT> &template_cloud)
    {
        templates_.push_back(template_cloud);
    }

    // Align the given template cloud to the target specified by setTargetCloud ()
    void
    align(FeatureCloud<FeatureInT> &template_cloud, SACAlignment<FeatureInT>::Result &result) noexcept
    {
        PCL_INFO("[SACAlignment::align] Align model %s\n", template_cloud.getId().c_str());
        sac_ia_.setInputCloud(template_cloud.getPointCloud());
        sac_ia_.setSourceFeatures(template_cloud.getLocalFeatures());

        PointCloud registration_output;
        sac_ia_.align(registration_output);

        PCL_INFO("SAC-IA alignment is complete\n");

        result.fitness_score = static_cast<float>(sac_ia_.getFitnessScore(max_correspondence_distance_));
        result.final_transformation = sac_ia_.getFinalTransformation();
    }

    // Align all of template clouds set by addTemplateCloud to the target specified by setTargetCloud ()
    std::vector<Result, Eigen::aligned_allocator<Result>>
    alignAll() noexcept
    {
        std::vector<Result, Eigen::aligned_allocator<Result>> results;

        results.resize(templates_.size());
        for (size_t i = 0; i < templates_.size(); ++i)
        {
            align(templates_[i], results[i]);
        }

        return results;
    }

    // Align all of template clouds to the target cloud to find the one with best alignment score
    int
    findBestAlignment(SACAlignment<FeatureInT>::Result &result) noexcept
    {
        // Align all of the templates to the target cloud
        std::vector<Result, Eigen::aligned_allocator<Result>> results;
        alignAll(results);

        // Find the template with the best (lowest) fitness score
        float lowest_score = std::numeric_limits<float>::infinity();
        int best_template = 0;
        for (size_t i = 0; i < results.size(); ++i)
        {
            const Result &r = results[i];
            if (r.fitness_score < lowest_score)
            {
                lowest_score = r.fitness_score;
                best_template = static_cast<int>(i);
            }
        }

        // Output the best alignment
        result = results[best_template];
        return (best_template);
    }

private:
    // A list of template clouds and the target to which they will be aligned
    std::vector<FeatureCloud<FeatureInT>> templates_;
    FeatureCloud<FeatureInT> target_;

    // The Sample Consensus Initial Alignment (SAC-IA) registration routine and its parameters
    pcl::SampleConsensusInitialAlignment<PointType, PointType, FeatureInT> sac_ia_;
    float min_sample_distance_;
    float max_correspondence_distance_;
    int nr_iterations_;
};

#endif

#endif

#endif // RANSAC_ALIGNMENT_H
