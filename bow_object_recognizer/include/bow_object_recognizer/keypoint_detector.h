#ifndef KEYPOINT_DETECTOR_H
#define KEYPOINT_DETECTOR_H

#include <iostream>
#include <vector>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/keypoints/uniform_sampling.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/range_image/range_image_planar.h>
#include <pcl/features/range_image_border_extractor.h>
#include <pcl/keypoints/narf_keypoint.h>
#include <pcl/keypoints/harris_3d.h>
#include <pcl/kdtree/kdtree_flann.h>
#include "typedefs.h"
// #include "common.h"

class KeypointDetector
{
public:
    using Ptr = boost::shared_ptr<KeypointDetector>;

    KeypointDetector() {}

    virtual void detectKeypoints(
        const PointCloudPtr &src,
        const PointCloudPtr &keypoints) = 0;

    inline void saveKeypoints(PointCloudPtr &keypoints, std::string path)
    {
        pcl::io::savePCDFileASCII(path.c_str(), *keypoints);
    }

    inline void loadKeypoints(std::string path, PointCloudPtr &keypoints)
    {
        // Load descriptors from the file
        pcl::io::loadPCDFile(path.c_str(), *keypoints);

        std::cout << "[KeypointDetector::loadKeypoints] Keypoints were loaded from file " << path << "\n";
    }

    ~KeypointDetector() {}
};

class UniformKeypointDetector : public KeypointDetector
{
public:
    using Ptr = boost::shared_ptr<UniformKeypointDetector>;

    UniformKeypointDetector()
    {
        std::cout << "UniformKeypointDetector::UniformKeypointDetector()\n";
    }

    inline void setRadius(float radius)
    {
        radius_ = radius;
    }

    double computeCloudResolution(const PointCloudConstPtr &cloud) noexcept
    {
        std::cout << "Compute cloud resolution ...\n";

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

    void detectKeypoints(
        const PointCloudPtr &src,
        const PointCloudPtr &keypoints) noexcept
    {
        std::cout << "---- Perform Uniform Sampling -----\n";

        double resolution = computeCloudResolution(src);

        std::cout << "Cloud resolution: " << resolution << "\n";

        pcl::UniformSampling<PointType> uniform_sampling;
        uniform_sampling.setInputCloud(src);
        uniform_sampling.setRadiusSearch(radius_);

        pcl::PointCloud<int> keypoint_idxes;
        uniform_sampling.compute(keypoint_idxes);

        pcl::copyPointCloud(*src, keypoint_idxes.points, *keypoints);

        std::cout << "[UniformKeypointDetector::detectKeypoints] Keypoints cloud has " << keypoints->points.size() << " points\n";
    }

    ~UniformKeypointDetector()
    {
    }

private:
    float radius_;
};

class ISSKeypointDetector : public KeypointDetector
{
    int salient_rad_factor_;
    int non_max_rad_factor_;

    using Ptr = boost::shared_ptr<ISSKeypointDetector>;

    ISSKeypointDetector()
    {
    }

    inline void setSalientRadiusFactor(int salient_rad_factor)
    {
        salient_rad_factor_ = salient_rad_factor;
    }

    inline void setNonMaxRadiusFactor(int non_max_rad_factor)
    {
        non_max_rad_factor_ = non_max_rad_factor;
    }

    double computeCloudResolution(const PointCloudConstPtr &cloud) noexcept
    {
        std::cout << "Compute cloud resolution ...\n";

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

    void detectKeypoints(
        const PointCloudPtr &src,
        const PointCloudPtr &keypoints) noexcept
    {
        // We assume that the keypoint detection is ISS
        std::cout << "---- Perform ISS ----\n";

        using KeypointDetector_ = pcl::ISSKeypoint3D<PointType, PointType>;
        KeypointDetector_ detector;
        detector.setInputCloud(src);
        KdTreePtr kdtree(new KdTree());
        detector.setSearchMethod(kdtree);

        double resolution = computeCloudResolution(src);

        std::cout << "Cloud resolution: " << resolution << "\n";

        double salient_radius = salient_rad_factor_ * resolution;
        double non_max_radius = non_max_rad_factor_ * resolution;

        PCL_INFO("Salient radius: %.3f\n", salient_radius);

        detector.setSalientRadius(salient_radius); // default - 6
        detector.setNonMaxRadius(non_max_radius);  // default - 4
        detector.setMinNeighbors(5);
        detector.setThreshold21(0.975);
        detector.setThreshold32(0.975);
        detector.setNumberOfThreads(8);

        detector.compute(*keypoints);

        std::cout << "[ISSKeypointDetector::detectKeypoints] Keypoints cloud has " << keypoints->points.size() << " points\n";
    }

    ~ISSKeypointDetector()
    {
    }
};

class SIFTKeypointDetector : public KeypointDetector
{
    float min_scale_;
    int nr_octaves_;
    int nr_scales_;
    float min_contrast_;
    float radius_;

public:
    using Ptr = boost::shared_ptr<SIFTKeypointDetector>;

    SIFTKeypointDetector()
    {
    }

    void setScales(float min_scale, int nr_octaves, int nr_scales)
    {
        min_scale_ = min_scale;
        nr_octaves_ = nr_octaves;
        nr_scales_ = nr_scales;
    }

    void setRadius(float radius)
    {
        radius_ = radius;
    }

    void setMinContrast(float min_contrast)
    {
        min_contrast_ = min_contrast;
    }

    void detectKeypoints(
        const PointCloudPtr &src,
        const PointCloudPtr &keypoints) noexcept
    {
        pcl::PointCloud<pcl::PointWithScale>::Ptr keypoints_tmp(new pcl::PointCloud<pcl::PointWithScale>);

        KdTreePtr tree(new KdTree);
        pcl::SIFTKeypoint<PointType, pcl::PointWithScale> detector;
        detector.setInputCloud(src);
        detector.setSearchSurface(src);
        detector.setSearchMethod(tree);
        detector.setScales(min_scale_, nr_octaves_, nr_scales_);
        detector.setMinimumContrast(min_contrast_);
        detector.compute(*keypoints_tmp);

        pcl::copyPointCloud(*keypoints_tmp, *keypoints);

        std::cout << "[SIFTKeypointDetector::detectKeypoints] Keypoints cloud has " << keypoints->points.size() << " points\n";
    }

    ~SIFTKeypointDetector()
    {
    }
};

class NARFKeypointDetector : public KeypointDetector
{
    float radius_;
    float support_size_;

public:
    using Ptr = boost::shared_ptr<NARFKeypointDetector>;

    NARFKeypointDetector()
    {
    }

    void setRadius(float radius)
    {
        radius_ = radius;
    }

    void setSupportSize(float s)
    {
        support_size_ = s;
    }

    void detectKeypoints(
        const PointCloudPtr &src,
        const PointCloudPtr &keypoints) noexcept
    {
        std::cout << "\n--------- Perform NARF -----------\n";

        Eigen::Affine3f scene_sensor_pose = Eigen::Affine3f(Eigen::Translation3f(src->sensor_origin_[0], src->sensor_origin_[1], src->sensor_origin_[2])) * Eigen::Affine3f(src->sensor_orientation_);

        boost::shared_ptr<pcl::RangeImage> range_image_ptr(new pcl::RangeImage);
        pcl::RangeImage &range_image = *range_image_ptr;
        range_image.createFromPointCloud(*src, pcl::deg2rad(0.5f), pcl::deg2rad(360.0f), pcl::deg2rad(180.0f), scene_sensor_pose, pcl::RangeImage::CAMERA_FRAME, 0.0, 0.0f, 1);

        range_image.setUnseenToMaxRange();

        pcl::RangeImageBorderExtractor range_image_border_extractor;
        pcl::NarfKeypoint detector;
        detector.setRangeImageBorderExtractor(&range_image_border_extractor);
        detector.setRangeImage(&range_image);
        detector.getParameters().support_size = support_size_; // 0.2f;
        detector.setRadiusSearch(radius_);

        pcl::PointCloud<int> keypoint_idxes;
        detector.compute(keypoint_idxes);

        pcl::copyPointCloud(*src, keypoint_idxes.points, *keypoints);
    }

    ~NARFKeypointDetector()
    {
    }
};

class Harris3DKeypointDetector : public KeypointDetector
{
    float threshold_;
    float radius_;

public:
    using Ptr = boost::shared_ptr<Harris3DKeypointDetector>;

    Harris3DKeypointDetector() {}

    void setRadius(float radius)
    {
        radius_ = radius;
    }

    void setThreshold(float threshold)
    {
        threshold_ = threshold;
    }

    void detectKeypoints(
        const PointCloudPtr &src,
        const PointCloudPtr &keypoints) noexcept
    {
        std::cout << "-------- Perform Harris3D -------\n";

        PointCloudNoColor keypoints_tmp;

        pcl::HarrisKeypoint3D<PointType, PointTypeNoColor> detector;
        detector.setNonMaxSupression(true);
        detector.setInputCloud(src);
        detector.setThreshold(threshold_);
        detector.setRadius(radius_); // 0.005
        detector.compute(keypoints_tmp);

        pcl::copyPointCloud(keypoints_tmp, *keypoints);

        // remove NaN points
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*keypoints, *keypoints, indices);

        std::cout << "[Harris3DKeypointDetector::detectKeypoints] Keypoints cloud has " << keypoints->points.size() << " points\n";
    }

    ~Harris3DKeypointDetector() {}
};

#endif // KEYPOINT_DETECTOR_H
