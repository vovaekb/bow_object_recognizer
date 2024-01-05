#ifndef COMMON_H
#define COMMON_H

#include "typedefs.h"
#include "keypoint_detector.h"
#include "feature_estimator.h"

constexpr std::string training_dataset;

constexpr int vocabulary_size;
constexpr int norm_est_k;
constexpr float norm_rad;
constexpr float descr_rad;
constexpr float voxel_grid_size;
constexpr float score_thresh;
constexpr int best_models_k;
constexpr bool perform_voxelizing;
// Uniform sampling
constexpr float us_radius;
// ISS
constexpr int iss_salient_rad_factor;
constexpr int iss_non_max_rad_factor;
// Harris
constexpr float harris_thresh;
constexpr float harris_radius;
// General methods
constexpr std::string keypoint_detector;
constexpr std::string feature_descriptor;
constexpr std::string test_scenes_dir;
constexpr std::string test_scene;
constexpr std::string scene_name;
constexpr std::string gt_files_dir;
constexpr std::string gt_file_path;
constexpr std::string scene_descr_file_path;
constexpr bool use_partial_views;
constexpr bool apply_thresh;
constexpr bool apply_verification;
constexpr bool limit_matches;
constexpr BoWModelDescriptorExtractor::Ptr bow_extractor;
constexpr Source::Ptr training_source;
constexpr KeypointDetector::Ptr detector;
constexpr FeatureEstimator<PointType>::Ptr feature_estimator;
constexpr std::map<std::string, BoWDescriptor> training_bow_descriptors;
#endif // COMMON_H
