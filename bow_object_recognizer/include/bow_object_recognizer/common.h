#ifndef COMMON_H
#define COMMON_H

#include "typedefs.h"
#include "keypoint_detector.h"
#include "feature_estimator.h"

extern std::string training_dataset;

extern int vocabulary_size;
extern int norm_est_k;
extern float norm_rad;
extern float descr_rad;
extern float voxel_grid_size;
extern float score_thresh;
extern int best_models_k;
extern bool perform_voxelizing;
// Uniform sampling
extern float us_radius;
// ISS
extern int iss_salient_rad_factor;
extern int iss_non_max_rad_factor;
// Harris
extern float harris_thresh;
extern float harris_radius;
// General methods
extern std::string keypoint_detector;
extern std::string feature_descriptor;
extern std::string test_scenes_dir;
extern std::string test_scene;
extern std::string scene_name;
extern std::string gt_files_dir;
extern std::string gt_file_path;
extern std::string scene_descr_file_path;
extern bool use_partial_views;
extern bool apply_thresh;
extern bool apply_verification;
extern bool limit_matches;
extern BoWModelDescriptorExtractor::Ptr bow_extractor;
extern Source::Ptr training_source;
extern KeypointDetector::Ptr detector;
extern FeatureEstimator<PointType>::Ptr feature_estimator;
extern std::map<std::string, BoWDescriptor> training_bow_descriptors;
#endif // COMMON_H
