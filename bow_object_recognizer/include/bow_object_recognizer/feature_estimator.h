#ifndef FEATURE_ESTIMATOR_H
#define FEATURE_ESTIMATOR_H

#include <string>
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


template <typename PointInT>
class FeatureEstimator
{
protected:
    typedef pcl::Normal NormalT;
    typedef typename pcl::PointCloud<PointInT>::Ptr PointInTPtr;
    typedef pcl::PointCloud<NormalT>::Ptr NormalTPtr;
    typedef std::vector<float> feature_point;

    float support_radius_;
    pcl::PointIndices::Ptr nan_indices_;


public:
    typedef boost::shared_ptr<FeatureEstimator<PointInT> > Ptr;

    FeatureEstimator() {  }

    virtual void calculateFeatures(PointInTPtr& in, PointInTPtr& keypoints, NormalTPtr& normals, std::vector<feature_point>& features) = 0;

    virtual void saveFeatures(std::vector<feature_point>& features, std::string path) = 0;

    virtual void loadFeatures(std::string path, std::vector<feature_point>& features) = 0;

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

template <typename PointInT>
class FeatureEstimatorSHOT : public FeatureEstimator<PointInT>
{
    using FeatureEstimator<PointInT>::dimensionality;
    using FeatureEstimator<PointInT>::support_radius_;
    using FeatureEstimator<PointInT>::nan_indices_;

protected:
    typedef pcl::Normal NormalT;
    typedef pcl::SHOT352 FeatureT;
    typedef typename pcl::PointCloud<PointInT>::Ptr PointInTPtr;
    typedef typename pcl::PointCloud<PointInT>::ConstPtr PointInTConstPtr;
    typedef pcl::PointCloud<FeatureT>::Ptr FeatureTPtr;
    typedef pcl::PointCloud<NormalT>::Ptr NormalTPtr;
    typedef std::vector<float> feature_point;

public:
    typedef boost::shared_ptr<FeatureEstimatorSHOT<PointInT> > Ptr;

    FeatureEstimatorSHOT()
    {
        dimensionality = 352;
        nan_indices_ = pcl::PointIndices::Ptr(new pcl::PointIndices);
    }

    double computeCloudResolution(const PointInTConstPtr& cloud)
    {
        double resolution = 0.0;
        int number_of_points = 0;
        int nres;
        std::vector<int> indices(2);
        std::vector<float> squared_distances(2);
        typename pcl::search::KdTree<PointInT> tree;
        tree.setInputCloud(cloud);

        for (size_t i = 0; i < cloud->size(); ++i)
        {
            if (! pcl_isfinite((*cloud)[i].x))
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

    void calculateFeatures(PointInTPtr& in, PointInTPtr& keypoints, NormalTPtr& normals, std::vector<feature_point>& features)
    {
        std::cout << "Calculate features SHOT ...\n";

        FeatureTPtr shots (new pcl::PointCloud<FeatureT> ());

        double resolution = computeCloudResolution(in);

        pcl::SHOTEstimationOMP<PointInT, NormalT, FeatureT> shot_estimate;
        shot_estimate.setInputCloud(keypoints);
        shot_estimate.setRadiusSearch(support_radius_);

        shot_estimate.setInputNormals(normals);
        shot_estimate.setSearchSurface(in);
        shot_estimate.setNumberOfThreads(8);
        shot_estimate.compute(*shots);

        PCL_INFO("SHOT descriptors has %d points\n\n", (int)shots->points.size());


        features.reserve(shots->size());

        // Preprocess features: remove NaNs
        for(size_t j = 0; j < shots->points.size(); j++)
        {
            feature_point signature (dimensionality);

            if(!pcl_isfinite(shots->at(j).descriptor[0])) {
                nan_indices_->indices.push_back(j);
                continue;
            }

            for(int idx = 0; idx < dimensionality; idx++)
            {
                signature[idx] = shots->points[j].descriptor[idx];
            }

            features.push_back(signature);
        }

        PCL_INFO("SHOT descriptors has %d points after NaN removal\n\n", (int)features.size());

        std::cout << "[calculateFeatures] NaNs in scene keypoints: " << nan_indices_->indices.size() << "\n";

    }

    void saveFeatures(std::vector<feature_point>& features, std::string path)
    {
        FeatureTPtr features_cloud (new pcl::PointCloud<FeatureT>);
        features_cloud->resize(features.size());

        for(size_t j = 0; j < features.size(); j++)
        {
            for(int idx = 0; idx < dimensionality; idx++)
            {
                features_cloud->points[j].descriptor[idx] = features.at(j)[idx];
            }

        }

        pcl::io::savePCDFileASCII(path.c_str(), *features_cloud);

        std::cout << "Features were saved to file: " << path << "\n";
    }

    void loadFeatures(std::string path, std::vector<feature_point>& features)
    {
        FeatureTPtr features_cloud (new pcl::PointCloud<FeatureT>);

        // Load descriptors from the file
        pcl::io::loadPCDFile(path.c_str(), *features_cloud);

        PCL_INFO("Descriptor %s has size: %d\n", path.c_str(), features_cloud->size());

        features.reserve(features_cloud->size());

        for(size_t j = 0; j < features_cloud->points.size(); j++)
        {
            feature_point descr_vect (dimensionality);

            if(!pcl::isFinite<FeatureT>(features_cloud->points[j]))
                PCL_WARN("Point %d is NaN\n", (int)j);

            for(int idx = 0; idx < dimensionality; idx++)
            {
                descr_vect[idx] = features_cloud->points[j].descriptor[idx];
            }

            features.push_back(descr_vect);
        }
    }

    ~FeatureEstimatorSHOT() {}
};

template <typename PointInT>
class FeatureEstimatorFPFH : public FeatureEstimator<PointInT>
{
    using FeatureEstimator<PointInT>::dimensionality;
    using FeatureEstimator<PointInT>::support_radius_;
    using FeatureEstimator<PointInT>::nan_indices_;

protected:
    typedef pcl::Normal NormalT;
    typedef pcl::FPFHSignature33 FeatureT;
    typedef typename pcl::PointCloud<PointInT>::Ptr PointInTPtr;
    typedef pcl::PointCloud<FeatureT>::Ptr FeatureTPtr;
    typedef pcl::PointCloud<NormalT>::Ptr NormalTPtr;
    typedef std::vector<float> feature_point;

public:
    typedef boost::shared_ptr<FeatureEstimatorFPFH<PointInT> > Ptr;

    FeatureEstimatorFPFH() {
        dimensionality = 33;
        nan_indices_ = pcl::PointIndices::Ptr(new pcl::PointIndices);
    }

    void calculateFeatures(PointInTPtr& in, PointInTPtr& keypoints, NormalTPtr& normals, std::vector<feature_point>& features)
    {
        std::cout << "Calculate features FPFH ...\n";

        for(size_t i = 0; i < in->points.size(); i++)
        {
            if(!pcl::isFinite<PointInT>(in->points[i]))
                PCL_WARN("Point %d is NaN\n", (int)i);
        }

        FeatureTPtr fpfhs (new pcl::PointCloud<FeatureT> ());

        pcl::FPFHEstimationOMP<PointInT, NormalT, FeatureT> fpfh_estimate;
        fpfh_estimate.setInputCloud(keypoints);
        // It was commented
        fpfh_estimate.setRadiusSearch(support_radius_);

        // begin -- from Semantic localization project
        typename pcl::search::KdTree<PointInT>::Ptr tree(new pcl::search::KdTree<PointInT>());
        fpfh_estimate.setSearchMethod(tree);
        // It was uncommented
//        fpfh_estimate.setKSearch(50);
        // end -- from Semantic localization project
        fpfh_estimate.setInputNormals(normals);
        fpfh_estimate.setSearchSurface(in);
        fpfh_estimate.setNumberOfThreads(8);
        fpfh_estimate.compute(*fpfhs);

        features.reserve(fpfhs->size());

        for(size_t j = 0; j < fpfhs->points.size(); j++)
        {
            feature_point signature (dimensionality);

            for(int idx = 0; idx < dimensionality; idx++)
            {
                signature[idx] = fpfhs->points[j].histogram[idx];
            }

            features.push_back(signature);
        }

    }

    void saveFeatures(std::vector<feature_point>& features, std::string path)
    {
        FeatureTPtr features_cloud (new pcl::PointCloud<FeatureT>);
        features_cloud->resize(features.size());

        for(size_t j = 0; j < features.size(); j++)
        {
            for(int idx = 0; idx < dimensionality; idx++)
            {
                features_cloud->points[j].histogram[idx] = features.at(j)[idx];
            }
        }

        pcl::io::savePCDFileASCII(path.c_str(), *features_cloud);

        std::cout << "Features were saved to file: " << path << "\n";
    }

    void loadFeatures(std::string path, std::vector<feature_point>& features)
    {
        FeatureTPtr features_cloud (new pcl::PointCloud<FeatureT>);

        // Load descriptors from the file
        pcl::io::loadPCDFile(path.c_str(), *features_cloud);

        std::cout << "Descriptor was loaded from file " << path << "\n";

        features.reserve(features_cloud->size());

        for(size_t j = 0; j < features_cloud->points.size(); j++)
        {
            feature_point descr_vect (dimensionality);

            if(!pcl::isFinite<FeatureT>(features_cloud->points[j]))
                PCL_WARN("Point %d is NaN\n", (int)j);

            for(int idx = 0; idx < dimensionality; idx++)
            {
                descr_vect[idx] = features_cloud->points[j].histogram[idx];
            }

            features.push_back(descr_vect);
        }
    }

    ~FeatureEstimatorFPFH() {}
};

template <typename PointInT>
class FeatureEstimatorPFHRGB : public FeatureEstimator<PointInT>
{
    using FeatureEstimator<PointInT>::dimensionality;
    using FeatureEstimator<PointInT>::support_radius_;
    using FeatureEstimator<PointInT>::nan_indices_;
    int k_search_;

protected:
    typedef pcl::Normal NormalT;
    typedef pcl::PFHRGBSignature250 FeatureT;
    typedef typename pcl::PointCloud<PointInT>::Ptr PointInTPtr;
    typedef pcl::PointCloud<FeatureT>::Ptr FeatureTPtr;
    typedef pcl::PointCloud<NormalT>::Ptr NormalTPtr;
    typedef std::vector<float> feature_point;

public:
    typedef boost::shared_ptr<FeatureEstimatorPFHRGB<PointInT> > Ptr;

    FeatureEstimatorPFHRGB() {
        dimensionality = 250;
        nan_indices_ = pcl::PointIndices::Ptr(new pcl::PointIndices);
    }

    void setKSearch(int k)
    {
        k_search_ = k;
    }

    void calculateFeatures(PointInTPtr& in, PointInTPtr& keypoints, NormalTPtr& normals, std::vector<feature_point>& features)
    {
        FeatureTPtr pfhrgbs (new pcl::PointCloud<FeatureT> ());

        typename pcl::search::KdTree<PointInT>::Ptr tree (new pcl::search::KdTree<PointInT>);

        pcl::PFHRGBEstimation<PointInT, NormalT, FeatureT> pfhrgb_estimation;
        pfhrgb_estimation.setInputCloud (keypoints);
        pfhrgb_estimation.setInputNormals(normals);
        pfhrgb_estimation.setSearchSurface(in);
        pfhrgb_estimation.setSearchMethod (tree);
        pfhrgb_estimation.setRadiusSearch (support_radius_);
        pfhrgb_estimation.compute (*pfhrgbs);

        features.reserve(pfhrgbs->size());

        for(size_t j = 0; j < pfhrgbs->points.size(); j++)
        {
            feature_point signature (dimensionality);

            for(int idx = 0; idx < dimensionality; idx++)
            {
                signature[idx] = pfhrgbs->points[j].histogram[idx];
            }

            features.push_back(signature);
        }

    }

    void saveFeatures(std::vector<feature_point>& features, std::string path)
    {
        FeatureTPtr features_cloud (new pcl::PointCloud<FeatureT>);
        features_cloud->resize(features.size());

        for(size_t j = 0; j < features.size(); j++)
        {
            for(int idx = 0; idx < dimensionality; idx++)
            {
                features_cloud->points[j].histogram[idx] = features.at(j)[idx];
            }
        }

        pcl::io::savePCDFileASCII(path.c_str(), *features_cloud);

        std::cout << "Features were saved to file: " << path << "\n";
    }

    void loadFeatures(std::string path, std::vector<feature_point>& features)
    {
        FeatureTPtr features_cloud (new pcl::PointCloud<FeatureT>);

        // Load descriptors from the file
        pcl::io::loadPCDFile(path.c_str(), *features_cloud);

        std::cout << "Descriptor was loaded from file " << path << "\n";

        features.reserve(features_cloud->size());

        for(size_t j = 0; j < features_cloud->points.size(); j++)
        {
            feature_point descr_vect (dimensionality);

            if(!pcl::isFinite<FeatureT>(features_cloud->points[j]))
                PCL_WARN("Point %d is NaN\n", (int)j);

            for(int idx = 0; idx < dimensionality; idx++)
            {
                descr_vect[idx] = features_cloud->points[j].histogram[idx];
            }

            features.push_back(descr_vect);
        }
    }

    ~FeatureEstimatorPFHRGB() {}
};

template <typename PointInT>
class FeatureEstimatorColorSHOT : public FeatureEstimator<PointInT>
{
    using FeatureEstimator<PointInT>::dimensionality;
    using FeatureEstimator<PointInT>::support_radius_;
    using FeatureEstimator<PointInT>::nan_indices_;

protected:
    typedef pcl::Normal NormalT;
    typedef pcl::SHOT1344 FeatureT;
    typedef typename pcl::PointCloud<PointInT>::Ptr PointInTPtr;
    typedef pcl::PointCloud<FeatureT>::Ptr FeatureTPtr;
    typedef pcl::PointCloud<NormalT>::Ptr NormalTPtr;
    typedef std::vector<float> feature_point;

public:
    typedef boost::shared_ptr<FeatureEstimatorColorSHOT<PointInT> > Ptr;

    FeatureEstimatorColorSHOT() {
        dimensionality = 1344;
        nan_indices_ = pcl::PointIndices::Ptr(new pcl::PointIndices);
    }

    void calculateFeatures(PointInTPtr& in, PointInTPtr& keypoints, NormalTPtr& normals, std::vector<feature_point>& features)
    {
        FeatureTPtr cshots (new pcl::PointCloud<FeatureT> ());

        pcl::SHOTColorEstimation<PointInT, NormalT> shotestimator;
        shotestimator.setInputCloud(keypoints);
        shotestimator.setInputNormals(normals);
        //computes the pointcloud resolution
        //sets the radius to three times the resolution
        shotestimator.setRadiusSearch(support_radius_);
        shotestimator.setSearchSurface(in);
        shotestimator.compute(*cshots);

        features.reserve(cshots->size());

        for(size_t j = 0; j < cshots->points.size(); j++)
        {
            feature_point signature (dimensionality);

            for(int idx = 0; idx < dimensionality; idx++)
            {
                signature[idx] = cshots->points[j].descriptor[idx];
            }

            features.push_back(signature);
        }
    }

    void saveFeatures(std::vector<feature_point>& features, std::string path)
    {
        FeatureTPtr features_cloud (new pcl::PointCloud<FeatureT>);
        features_cloud->resize(features.size());

        for(size_t j = 0; j < features.size(); j++)
        {
            for(int idx = 0; idx < dimensionality; idx++)
            {
                features_cloud->points[j].descriptor[idx] = features.at(j)[idx];
            }

        }

        pcl::io::savePCDFileASCII(path.c_str(), *features_cloud);

        std::cout << "Features were saved to file: " << path << "\n";
    }

    void loadFeatures(std::string path, std::vector<feature_point>& features)
    {
        FeatureTPtr features_cloud (new pcl::PointCloud<FeatureT>);

        // Load descriptors from the file
        pcl::io::loadPCDFile(path.c_str(), *features_cloud);

        std::cout << "Descriptor was loaded from file " << path << "\n";

        features.reserve(features_cloud->size());

        for(size_t j = 0; j < features_cloud->points.size(); j++)
        {
            feature_point descr_vect (dimensionality);

            if(!pcl::isFinite<FeatureT>(features_cloud->points[j]))
                PCL_WARN("Point %d is NaN\n", (int)j);

            for(int idx = 0; idx < dimensionality; idx++)
            {
                descr_vect[idx] = features_cloud->points[j].descriptor[idx];
            }

            features.push_back(descr_vect);
        }
    }

    ~FeatureEstimatorColorSHOT() {}
};

#endif // FEATURE_ESTIMATOR_H
