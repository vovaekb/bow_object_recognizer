#ifndef NORMAL_ESTIMATOR_H
#define NORMAL_ESTIMATOR_H

#define DEBUG_UNIFORM_SAMPLING
#define DEBUG_KEYPOINT_DETECTOR_CLASS

#include <pcl/console/print.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/common/transforms.h>

using namespace std;

template<typename PointInT>
class NormalEstimator
{
    typedef typename pcl::PointCloud<PointInT>::Ptr PointInTPtr;
    typedef typename pcl::PointCloud<PointInT>::ConstPtr PointInTConstPtr;

public:
    typedef boost::shared_ptr<NormalEstimator> Ptr;
    bool do_scaling_;
    bool do_voxelizing_;

    float grid_resolution_;
    int norm_est_k_;
    float norm_rad_;

    NormalEstimator () {}

    inline void setDoScaling (float do_scaling)
    {
        do_scaling_ = do_scaling;
    }

    inline void setDoVoxelizing(bool do_voxelizing)
    {
        do_voxelizing_ = do_voxelizing;
    }

    inline void setGridResolution(float grid_resolution)
    {
        grid_resolution_ = grid_resolution;
    }

    inline void setNormalK(int k)
    {
        norm_est_k_ = k;
    }

    inline void setNormalRadius(float rad)
    {
        norm_rad_ = rad;
    }

    void estimate (PointInTPtr &in, PointInTPtr &out, pcl::PointCloud<pcl::Normal>::Ptr &normals)
    {
        if(do_scaling_)
        {
            Eigen::Matrix4f cloud_transform = Eigen::Matrix4f::Identity();

            cloud_transform(0,0) = 0.001;
            cloud_transform(1,1) = 0.001;
            cloud_transform(2,2) = 0.001;

            pcl::transformPointCloud(*in, *in, cloud_transform);
        }

        if(do_voxelizing_)
        {
            // Voxelize cloud
            float grid_size = grid_resolution_; // 0.0025 - value used in the 3d_rec_framework
            pcl::VoxelGrid<PointInT> voxel_grid;
            voxel_grid.setInputCloud(in);
            voxel_grid.setLeafSize(grid_size, grid_size, grid_size);

            PointInTPtr temp_cloud (new pcl::PointCloud<PointInT> ());
            voxel_grid.filter(*temp_cloud);

            in = temp_cloud;
        }

        // Remove NaNs
        vector<int> mapping;
        pcl::removeNaNFromPointCloud(*in, *out, mapping);

        std::cout << "[estimate] The number of points: " << out->points.size() << "\n";

        // Calculate normals

        typedef typename pcl::NormalEstimationOMP<PointInT, pcl::Normal> NormalEstimator_;
        NormalEstimator_ norm_est;
        norm_est.setKSearch(norm_est_k_);
        norm_est.setInputCloud(out);
        norm_est.compute(*normals);

        normals->is_dense = false;

        for(size_t i = 0; i < normals->points.size(); i++)
        {
            if(!pcl::isFinite<pcl::Normal>(normals->points[i]))
                PCL_WARN("Normal %d is NaN\n", (int)i);
        }

        mapping.clear();
        pcl::removeNaNNormalsFromPointCloud(*normals, *normals, mapping);
        if(mapping.size() > 0)
        {
            PointInTPtr cloud_tmp (new pcl::PointCloud<PointInT> ());
            pcl::copyPointCloud(*out, *cloud_tmp);
            pcl::copyPointCloud(*cloud_tmp, mapping, *out);
        }

        PCL_INFO("The number of normals: %d\n\n", (int)normals->points.size());
    }
};

#endif // NORMAL_ESTIMATOR_H
