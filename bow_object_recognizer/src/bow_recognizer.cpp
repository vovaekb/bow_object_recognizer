/*
 * Author: Privalov Vladimir, iprivalov@fit.vutbr.cz
 * Date: May 2016
 */

#include <stdio.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <math.h>
#include <utility>
#include <numeric>
#include <pcl/console/parse.h>
#include <pcl/console/print.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/fpfh_omp.h>
#include <bow_object_recognizer/source.h>
#include <bow_object_recognizer/bagofwords.h>
#include <bow_object_recognizer/test_runner.h>
#include <bow_object_recognizer/persistence_utils.h>
#include <bow_object_recognizer/normal_estimator.h>
#include <bow_object_recognizer/sac_alignment.h>
#include <bow_object_recognizer/keypoint_detector.h>
#include <bow_object_recognizer/feature_estimator.h>
#include "pc_source/source.cpp"
#include "bagofwords.cpp"
#include "test_runner.cpp"
#include "typedefs.h"

using namespace std;

bool use_partial_views(false);
string training_dir;
string test_scene_path;
string gt_file_path;
std::string scene_descr_file_path;

std::string training_dataset;

string experiments_dir = "experiments/kinect_train_set_tests"; // "willow_train_set_tests" - for willow dataset; "experiments/correct_train_set_tests" - for RAM dataset
string keypoint_detector("us");
string flann_index_file("vocab_flann.idx");
string training_data_file("training_data.h5");

// Algorithm parameters

string test_name = "score_thresh_test";

// Shared parameters
auto vocabulary_size{500};
auto perform_voxelizing{false};

auto norm_est_k{10};
// float norm_rad (0.02f); // 0.01f // added
float descr_rad(0.05f); // 0.05f

float voxel_grid_size(0.01f);

// Uniform Sampling
float us_radius;
// ISS
auto iss_salient_rad_factor(6);
auto iss_non_max_rad_factor(4);
// Harris
float harris_thresh(0.0001);
float harris_radius(0.01);

// Threshold for rejecting wrong detections
float score_thresh(0.05); // 0.00015
auto best_models_k(5);
string feature_descriptor("fpfh");
string test_scenes_dir;
std::string test_scene;
string scene_name;
string gt_files_dir = "gt_files";
bool apply_thresh(false);
bool apply_verification(false);
bool limit_matches(false);
BoWModelDescriptorExtractor::Ptr bow_extractor;

Source::Ptr training_source;
KeypointDetector::Ptr detector;
FeatureEstimator<PointType>::Ptr feature_estimator;

std::map<std::string, BoWDescriptor> training_bow_descriptors;

bool run_tests(false);

std::vector<string> found_models;

void loadBoWModels()
{
    // Load BoW descriptors of training models
    cout << "Loading training BoW descriptors\n";

    string models_dir = "models";
    std::stringstream path_ss;
    path_ss << training_dir << "/" << models_dir;

    boost::filesystem::path models_path = path_ss.str();

    training_source->getModelsInDir(models_path);

    std::vector<Model> training_models = training_source->getModels();

    cout << training_models.size() << " training models has been loaded\n";

    for (auto &training_model : training_models)
    {
        string model_path = training_source->getModelDir(training_model);

        cout << "\n\nProcessing model " << training_model.model_id << "\n";

        if (use_partial_views)
        {
            for (auto &view_id : training_model.views)
            {
                std::stringstream bow_sample_id;
                bow_sample_id << training_model.model_id << "_" << view_id;

                std::stringstream bow_descr_path;
                bow_descr_path << model_path << "/views/" << view_id << "_bow_descr.txt";

                string bow_descr_file = bow_descr_path.str();

                BoWDescriptor &view_bow_descr = training_bow_descriptors[bow_sample_id.str()];
                PersistenceUtils::readVectorFromFile(bow_descr_file, view_bow_descr);
            }
        }
        else
        {
            cout << "\n\nLoading BoW descriptor for model " << training_model.model_id << "\n";

            std::stringstream bow_descr_path;
            // TODO: Add " << feature_descriptor << "
            bow_descr_path << model_path << "/bow_descr.txt";

            string bow_descr_file = bow_descr_path.str();

            BoWDescriptor &model_bow_descr = training_bow_descriptors[training_model.model_id];
            PersistenceUtils::readVectorFromFile(bow_descr_file, model_bow_descr);
        }
    }
}

void recognizeScene(PointCloudPtr &scene_cloud)
{
    std::vector<Model> training_models = training_source->getModels();

    std::vector<BoWDescriptorPoint> descriptors_vect;

    std::stringstream path_ss;
    path_ss << test_scenes_dir << "/" << feature_descriptor;

    path_ss << "/" << scene_name << ".pcd";
    scene_descr_file_path = path_ss.str();

    path_ss.str("");
    path_ss << test_scenes_dir << "/keypoints/" << scene_name << ".pcd";

    string scene_keypoints_file = path_ss.str();

    PointCloudPtr scene_keypoints(new PointCloud());

    if (!boost::filesystem::exists(scene_descr_file_path))
    {

        SurfaceNormalsPtr normals(new SurfaceNormals());

        NormalEstimator<PointType>::Ptr normal_estimator(new NormalEstimator<PointType>);

        normal_estimator->setDoScaling(false);
        normal_estimator->setGridResolution(voxel_grid_size);
        normal_estimator->setDoVoxelizing(perform_voxelizing);
        normal_estimator->setNormalK(norm_est_k);
        // normal_estimator->setNormalRadius(norm_rad);
        normal_estimator->estimate(scene_cloud, scene_cloud, normals);

        // Detect keypoints
        if (keypoint_detector.compare("us") == 0)
        {
            boost::shared_ptr<UniformKeypointDetector> cast_detector = boost::static_pointer_cast<UniformKeypointDetector>(detector);
            cast_detector->setRadius(0.02); // 0.01
        }
        else if (keypoint_detector.compare("iss") == 0)
        {
            boost::shared_ptr<ISSKeypointDetector> cast_detector = boost::static_pointer_cast<ISSKeypointDetector>(detector);
            cast_detector->setSalientRadiusFactor(iss_salient_rad_factor);
            cast_detector->setNonMaxRadiusFactor(iss_non_max_rad_factor);
        }
        else if (keypoint_detector.compare("harris") == 0)
        {
            boost::shared_ptr<Harris3DKeypointDetector> cast_detector = boost::static_pointer_cast<Harris3DKeypointDetector>(detector);
            cast_detector->setThreshold(harris_thresh);
            cast_detector->setRadius(harris_radius);
        }

        detector->detectKeypoints(scene_cloud, scene_keypoints);

        detector->saveKeypoints(scene_keypoints, scene_keypoints_file);

        //
        // Calculate the BoW descriptor for the scene
        //

        feature_estimator->setSupportRadius(descr_rad);

        feature_estimator->calculateFeatures(scene_cloud, scene_keypoints, normals, &descriptors_vect);

        // Save feature cloud to PCD and load it for using in SACAlignment
        feature_estimator->saveFeatures(descriptors_vect, scene_descr_file_path);
    }
    else
    {
        feature_estimator->loadFeatures(scene_descr_file_path, descriptors_vect);

        detector->loadKeypoints(scene_keypoints_file, scene_keypoints);
    }

    for (size_t i = 0; i < scene_keypoints->points.size(); i++)
    {
        if (!pcl::isFinite<PointType>(scene_keypoints->points[i]))
            PCL_WARN("Keypoint %d is NaN\n", static_cast<int>(i));
    }

    std::vector<float> query_bow_descriptor;

    bow_extractor->compute(descriptors_vect, query_bow_descriptor);

    // Calculate the number of models containing the word
    vocabulary_size = static_cast<int>(query_bow_descriptor.size());

    std::vector<float> word_occurrences;
    word_occurrences.resize(vocabulary_size, 0);

    std::map<string, BoWDescriptor>::iterator map_it;

    for (size_t idx = 0; idx < vocabulary_size; idx++)
    {
        for (map_it = training_bow_descriptors.begin(); map_it != training_bow_descriptors.end(); map_it++)
        {
            std::vector<float> descr = (*map_it).second;
            if (descr[idx] > 0)
                word_occurrences[idx] = word_occurrences[idx] + 1.f;
        }
    }

    int training_model_samples_number = training_source->getModelSamplesNumber();

    // Calculate idf-index
    for (size_t idx = 0; idx < query_bow_descriptor.size(); idx++)
    {
        auto word_occurrences_n = word_occurrences[idx];

        decltype(word_occurrences_n) word_idf = (word_occurrences_n != 0 ? log10(training_model_samples_number / word_occurrences_n) : 0);

        query_bow_descriptor[idx] = word_idf * query_bow_descriptor[idx];
    }

    //
    // Match scene against training models and rank matches
    //
    cout << "Match of the scene with the training models\n";

    vector<float> model_match_scores(training_models.size());

    if (use_partial_views)
    {
        for (auto &training_model : training_models)
        {
            vector<float> view_match_scores(training_model.views.size());

            cout << "Loading views for model " << training_model.model_id << "\n";

            for (auto &view_id : training_model.views)
            {
                std::stringstream bow_view_sample_key;
                bow_view_sample_key << training_model.model_id << "_" << view_id;

                BoWDescriptor view_bow_descriptor = training_bow_descriptors[bow_view_sample_key.str()];

                auto bow_dist = 0;
                for (size_t idx = 0; idx < view_bow_descriptor.size(); idx++)
                {
                    bow_dist = bow_dist + view_bow_descriptor[idx] * query_bow_descriptor[idx];
                }

                view_match_scores[j] = bow_dist;
            }

            std::sort(view_match_scores.begin(), view_match_scores.end(), std::greater<float>());

            auto best_view_score = view_match_scores[0];

            view_match_scores.clear();

            model_match_scores[i] = best_view_score;
        }
    }
    else
    {
        for (size_t i = 0; i < training_models.size(); i++)
        {
            string model_id = training_models[i].model_id;
            BoWDescriptor model_bow_descriptor = training_bow_descriptors[model_id];

            auto bow_dist = 0;
            for (size_t idx = 0; idx < model_bow_descriptor.size(); idx++)
            {
                bow_dist = bow_dist + model_bow_descriptor[idx] * query_bow_descriptor[idx];
            }

            model_match_scores[i] = bow_dist;
        }
    }

    vector<ModelScore> best_matches;
    sortModelSACAlignmentScores sort_model_sac_align_scores_op;

    size_t j = 0;
    std::for_each(model_match_scores.begin(), model_match_scores.end(), [&j, &training_models, &best_matches](float &model_match_score)
                  {
        Model model = training_models[j];
        ModelScore match;
        match.model_id = model.model_id;
        match.score = model_match_score;
        best_matches.push_back(std::move(match));
        j++; });

    model_match_scores.clear();

    // The best matches are ones with higher score (cosine tends to 1)
    // Display results of the matching
    if (best_matches.size())
    {
        // rank matches
        std::sort(best_matches.begin(), best_matches.end(),
                  [](const auto &d1, const auto &d2)
                  {
                      return d1.score > d2.score
                  });

        if (apply_verification)
        {
            cout << "Matches found: " << best_matches.size() << "\n";

            std::cout << "---------------------------------------\n";
            std::cout << "------ Geometric verification ---------\n";
            std::cout << "---------------------------------------\n";
            std::cout << "---------------------------------------\n";

            if (best_matches.size() > 10)
                best_matches.resize(10); // best models number

            //
            // Geometric verification
            //

            if (feature_descriptor.compare("fpfh") == 0)
            {
                // Initialize feature type
                using DescriptorType = pcl::FPFHSignature33;

                std::cout << "------------------- Set scene scloud to FeatureCloud ------------------\n";

                // Initialize FeatureCloud object
                FeatureCloud<DescriptorType> scene_feature_cloud;
                scene_feature_cloud.setInputCloud(scene_keypoints);
                scene_feature_cloud.loadInputFeatures(scene_descr_file_path);

                std::cout << "Create FeatureCloud objects for best candidates\n";

                std::vector<FeatureCloud<DescriptorType>> candidate_feature_clouds;

                SACAlignment<DescriptorType> alignment;

                for (auto &match : best_matches)
                {
                    FeatureCloud<DescriptorType> candidate_feature_cloud;

                    string model_id = match.model_id;
                    candidate_feature_cloud.setId(model_id);

                    string model_path = training_source->getModelDir(model_id);

                    path_ss.str("");
                    path_ss << model_path << "/" << feature_descriptor << "_descr.pcd";
                    string descr_path = path_ss.str();

                    candidate_feature_cloud.loadInputFeatures(descr_path);

                    path_ss.str("");
                    path_ss << model_path << "/keypoints.pcd";

                    string keypoints_cloud_path = path_ss.str();
                    candidate_feature_cloud.loadInputCloud(keypoints_cloud_path);

                    candidate_feature_clouds.push_back(std::move(candidate_feature_cloud));

                    alignment.addTemplateCloud(candidate_feature_cloud);
                }

                alignment.setTargetCloud(scene_feature_cloud);

                std::vector<SACAlignment<DescriptorType>::Result, Eigen::aligned_allocator<SACAlignment<DescriptorType>::Result>> alignment_results = alignment.alignAll();

                for (size_t i = 0; i < alignment_results.size(); i++)
                {
                    printf("Fitness score for model %d: %f\n", static_cast<int>(i), alignment_results[i].fitness_score);
                }

                for (size_t i = 0; i < best_matches.size(); i++)
                {
                    best_matches[i].sac_alignment_score = alignment_results[i].fitness_score;
                }

                std::cout << "Best matches alignment scores:\n";
                for (auto &match : best_matches)
                {
                    printf("sac_alignment_score for %s: %f\n", match.model_id.c_str(), match.sac_alignment_score);
                }

                std::sort(best_matches.begin(), best_matches.end(),
                          [](const auto &d1, const auto &d2)
                          {
                              return d1.sac_alignment_score < d2.sac_alignment_score
                          });

                std::cout << "Best matches alignment scores after ranking:\n";
                for (auto &match : best_matches)
                {
                    printf("sac_alignment_score for %s: %f\n", match.model_id.c_str(), match.sac_alignment_score);
                }
            }
            else if (feature_descriptor.compare("shot") == 0)
            {
                // Initialize feature type
                using DescriptorType = pcl::SHOT352;

                std::cout << "------------------- Set scene scloud to FeatureCloud ------------------\n";

                // Initialize FeatureCloud object
                FeatureCloud<DescriptorType> scene_feature_cloud;
                scene_feature_cloud.setInputCloud(scene_keypoints);
                scene_feature_cloud.loadInputFeatures(scene_descr_file_path);

                std::cout << "Create FeatureCloud objects for best candidates\n";

                std::vector<FeatureCloud<DescriptorType>> candidate_feature_clouds;

                SACAlignment<DescriptorType> alignment;

                for (auto &match : best_matches)
                {
                    FeatureCloud<DescriptorType> candidate_feature_cloud;

                    string model_id = match.model_id;
                    candidate_feature_cloud.setId(model_id);

                    string model_path = training_source->getModelDir(model_id);

                    path_ss.str("");
                    path_ss << model_path << "/" << feature_descriptor << "_descr.pcd";
                    string descr_path = path_ss.str();

                    candidate_feature_cloud.loadInputFeatures(descr_path);

                    path_ss.str("");
                    path_ss << model_path << "/keypoints.pcd";

                    string keypoints_cloud_path = path_ss.str();
                    candidate_feature_cloud.loadInputCloud(keypoints_cloud_path);

                    candidate_feature_clouds.push_back(std::move(candidate_feature_cloud));

                    alignment.addTemplateCloud(candidate_feature_cloud);
                }

                alignment.setTargetCloud(scene_feature_cloud);

                std::vector<SACAlignment<DescriptorType>::Result, Eigen::aligned_allocator<SACAlignment<DescriptorType>::Result>> alignment_results = alignment.alignAll();

                for (size_t i = 0; i < alignment_results.size(); i++)
                {
                    printf("Fitness score for model %d: %f\n", static_cast<int>(i), alignment_results[i].fitness_score);
                }

                for (size_t i = 0; i < best_matches.size(); i++)
                {
                    best_matches[i].sac_alignment_score = alignment_results[i].fitness_score;
                }

                std::cout << "Best matches alignment scores:\n";
                for (auto &match : best_matches)
                {
                    printf("sac_alignment_score for %s: %f\n", match.model_id.c_str(), match.sac_alignment_score);
                }

                std::sort(best_matches.begin(), best_matches.end(),
                          [](const auto &d1, const auto &d2)
                          {
                              return d1.sac_alignment_score < d2.sac_alignment_score
                          });

                std::cout << "Best matches alignment scores after ranking:\n";
                for (auto &match : best_matches)
                {
                    printf("sac_alignment_score for %s: %f\n", match.model_id.c_str(), match.sac_alignment_score);
                }
            }
            else if (feature_descriptor.compare("pfhrgb") == 0)
            {
                // Initialize feature type
                using DescriptorType = pcl::PFHRGBSignature250;

                std::cout << "------------------- Set scene scloud to FeatureCloud ------------------\n";

                // Initialize FeatureCloud object
                FeatureCloud<DescriptorType> scene_feature_cloud;
                scene_feature_cloud.setInputCloud(scene_keypoints);
                scene_feature_cloud.loadInputFeatures(scene_descr_file_path);

                std::cout << "Create FeatureCloud objects for best candidates\n";

                std::vector<FeatureCloud<DescriptorType>> candidate_feature_clouds;

                SACAlignment<DescriptorType> alignment;

                for (auto &match : best_matches)
                {
                    FeatureCloud<DescriptorType> candidate_feature_cloud;

                    string model_id = match.model_id;
                    candidate_feature_cloud.setId(model_id);

                    string model_path = training_source->getModelDir(model_id);

                    path_ss.str("");
                    path_ss << model_path << "/" << feature_descriptor << "_descr.pcd";
                    string descr_path = path_ss.str();

                    candidate_feature_cloud.loadInputFeatures(descr_path);

                    path_ss.str("");
                    path_ss << model_path << "/keypoints.pcd";

                    string keypoints_cloud_path = path_ss.str();
                    candidate_feature_cloud.loadInputCloud(keypoints_cloud_path);

                    candidate_feature_clouds.push_back(std::move(candidate_feature_cloud));

                    alignment.addTemplateCloud(candidate_feature_cloud);
                }

                alignment.setTargetCloud(scene_feature_cloud);

                std::vector<SACAlignment<DescriptorType>::Result, Eigen::aligned_allocator<SACAlignment<DescriptorType>::Result>> alignment_results = alignment.alignAll();

                for (size_t i = 0; i < alignment_results.size(); i++)
                {
                    printf("Fitness score for model %d: %f\n", static_cast<int>(i), alignment_results[i].fitness_score);
                }

                for (size_t i = 0; i < best_matches.size(); i++)
                {
                    best_matches[i].sac_alignment_score = alignment_results[i].fitness_score;
                }

                std::cout << "Best matches alignment scores:\n";
                for (auto &match : best_matches)
                {
                    printf("sac_alignment_score for %s: %f\n", match.model_id.c_str(), match.sac_alignment_score);
                }

                std::sort(best_matches.begin(), best_matches.end(),
                          [](const auto &d1, const auto &d2)
                          {
                              return d1.sac_alignment_score < d2.sac_alignment_score
                          });

                std::cout << "Best matches alignment scores after ranking:\n";
                for (auto &match : best_matches)
                {
                    printf("sac_alignment_score for %s: %f\n", match.model_id.c_str(), match.sac_alignment_score);
                }
            }
            else if (feature_descriptor.compare("cshot") == 0)
            {
                // Initialize feature type
                using DescriptorType = pcl::SHOT1344;

                std::cout << "------------------- Set scene scloud to FeatureCloud ------------------\n";

                // Initialize FeatureCloud object
                FeatureCloud<DescriptorType> scene_feature_cloud;
                scene_feature_cloud.setInputCloud(scene_keypoints);
                scene_feature_cloud.loadInputFeatures(scene_descr_file_path);

                std::cout << "Create FeatureCloud objects for best candidates\n";

                std::vector<FeatureCloud<DescriptorType>> candidate_feature_clouds;

                SACAlignment<DescriptorType> alignment;

                for (auto &match : best_matches)
                {
                    FeatureCloud<DescriptorType> candidate_feature_cloud;

                    string model_id = match.model_id;
                    candidate_feature_cloud.setId(model_id);

                    string model_path = training_source->getModelDir(model_id);

                    path_ss.str("");
                    path_ss << model_path << "/" << feature_descriptor << "_descr.pcd";
                    string descr_path = path_ss.str();

                    candidate_feature_cloud.loadInputFeatures(descr_path);

                    path_ss.str("");
                    path_ss << model_path << "/keypoints.pcd";

                    string keypoints_cloud_path = path_ss.str();
                    candidate_feature_cloud.loadInputCloud(keypoints_cloud_path);

                    candidate_feature_clouds.push_back(std::move(candidate_feature_cloud));

                    alignment.addTemplateCloud(candidate_feature_cloud);
                }

                alignment.setTargetCloud(scene_feature_cloud);

                std::vector<SACAlignment<DescriptorType>::Result, Eigen::aligned_allocator<SACAlignment<DescriptorType>::Result>> alignment_results = alignment.alignAll();

                for (size_t i = 0; i < alignment_results.size(); i++)
                {
                    printf("Fitness score for model %d: %f\n", static_cast<int>(i), alignment_results[i].fitness_score);
                }

                for (size_t i = 0; i < best_matches.size(); i++)
                {
                    best_matches[i].sac_alignment_score = alignment_results[i].fitness_score;
                }

                std::cout << "Best matches alignment scores:\n";
                for (auto &match : best_matches)
                {
                    printf("sac_alignment_score for %s: %f\n", match.model_id.c_str(), match.sac_alignment_score);
                }

                std::sort(best_matches.begin(), best_matches.end(),
                          [](const auto &d1, const auto &d2)
                          {
                              return d1.sac_alignment_score < d2.sac_alignment_score
                          });

                std::cout << "Best matches alignment scores after ranking:\n";
                for (auto &match : best_matches)
                {
                    printf("sac_alignment_score for %s: %f\n", match.model_id.c_str(), match.sac_alignment_score);
                }
            }
        }

        for (auto &match : best_matches)
        {
            cout << "Model " << match.model_id << " has matching score: " << match.score << "\n";
        }

        std::cout << "Estimate accuracy of recognition\n";

        int positives_n = 0;
        int negatives_n = 0;
        int tp_n = 0;
        int fp_n = 0;
        int tn_n = 0;
        int fn_n = 0;

        for (int i = 0; i < training_models.size(); i++)
        {
            string model_id = training_models[i].model_id;

            bool is_present = false;

            if (training_dataset.compare("ram") == 0)
            {
                is_present = PersistenceUtils::modelPresents(gt_file_path, model_id);
            }
            else if (training_dataset.compare("willow") == 0 || training_dataset.compare("tuw") == 0)
            {
                is_present = PersistenceUtils::modelPresents(gt_file_path, scene_name, model_id);
            }

            bool is_found = false;

            for (auto &match : best_matches)
            {
                std::string match_model_id = match.model_id;
                if (match_model_id == model_id)
                    is_found = true;
            }

            cout << "Model " << model_id << "\n";
            cout << "\t- is present: " << is_present << "\n";
            cout << "\t- is found: " << is_found << "\n\n";

            if (is_present)
            {
                if (is_found)
                {
                    tp_n++;
                    positives_n++;
                }
                else
                {
                    fn_n++;
                    negatives_n++;
                }
            }
            else
            {
                if (is_found)
                {
                    fp_n++;
                    positives_n++;
                }
                else
                {
                    tn_n++;
                    negatives_n++;
                }
            }
        }

        cout << "Positives: " << positives_n << "\n";
        cout << "Negatives: " << negatives_n << "\n";
        cout << "tp: " << tp_n << ", fp: " << fp_n << "\n";
        cout << "fn: " << fn_n << ", tn: " << tn_n << "\n";
    }
}

void runTestMode()
{
    std::cout << "**************** Run test mode *****************\n\n";

    TestRunner::Ptr test_runner(new TestRunner(experiments_dir, test_name, true));
    test_runner->initTests();
}

/*
 * Command line utilities
 */

void showHelp(char *filename)
{
    std::cout << "****************************************************************************************************\n";
    std::cout << "*                                                                                                  *\n";
    std::cout << "*                               BoW object recognizer - Usage Guide                                *\n";
    std::cout << "*                                                                                                  *\n";
    std::cout << "****************************************************************************************************\n";
    std::cout << "Using: " << filename << " --train_dir <training directory> [options]\n";
    std::cout << "options:\n";
    std::cout << "--test_scene <scene_cloud.pcd>:                       path to test scene cloud\n";
    std::cout << "--test_dir <dir>:                                     directory with test scenes for experiments\n";
    std::cout << "--gt_dir <dir>:                                       directory with ground truth data for experiments\n";
    std::cout << "--test_name <test_name>:                              test name / name for experiment directory\n";
    std::cout << "--descr <descriptor>:                                 descriptor to use (shot, fpfh, cshot, pfhrgb)\n";
    std::cout << "--keypoint_det <detector>:                            keypoint detector to use (iss, us)\n";
    std::cout << "--voc_size <voc_size>:                                vocabulary size (for reporting in experiment results only)\n";
    std::cout << "--vox_grid_size <vox_grid_size>:                      voxel grid size\n";
    std::cout << "--iss_sal_rad_factor <iss_sal_rad_factor>:            salience radius factor for ISS\n";
    std::cout << "--iss_non_max_rad_factor <iss_non_max_rad_factor>:    non maximum radius factor for ISS\n";
    std::cout << "--harris_thresh <harris_thresh>:                      outlier threshold for Harris\n";
    std::cout << "--harris_radius <harris_radius>:                      radius for Harris\n";
    std::cout << "--th score_thresh:                                     match threshold\n";
    std::cout << "-thresh:                                              apply threshold to match scores\n";
    std::cout << "-verify:                                              apply verification by SAC-IA to rank match scores\n";
    std::cout << "-limit:                                               limit matches by N (for testing spatial verification)\n";
    std::cout << "-voxelize:                                            perform voxelizing for training point clouds\n";
    std::cout << "-test:                                                run tests\n";
    std::cout << "-partial_views:                                       use partial views for object models\n";
    std::cout << "-h:                                                   show help\n";
}

void printParams()
{
    std::cout << "training dir: " << training_dir << "\n"
                                                     "test scene: "
              << test_scene << "\n"
                               "test name: "
              << test_name << "\n"
                              "feature descriptor: "
              << feature_descriptor << "\n"
                                       "voxel grid size: "
              << voxel_grid_size << "\n"
                                    "vocabulary size: "
              << vocabulary_size << "\n"
                                    "harris radius: "
              << harris_radius << "\n"
                                  "use partial views: "
              << use_partial_views << "\n"
                                      "Apply threshold: "
              << apply_thresh << "\n"
                                 "Apply verification: "
              << apply_verification << "\n"
                                       "Limit matches: "
              << limit_matches << "\n";
}

void parseCommandLine(int argc, char **argv)
{
    pcl::console::parse_argument(argc, argv, "--train_dir", training_dir);

    if (training_dir.compare("") == 0)
    {
        PCL_ERROR("The train_dir parameter is missing\n");
        showHelp(argv[0]);
        exit(-1);
    }

    pcl::console::parse_argument(argc, argv, "--test_scene", test_scene);

    pcl::console::parse_argument(argc, argv, "--test_name", test_name);

    if (pcl::console::find_switch(argc, argv, "-partial_views"))
        use_partial_views = true;

    if (pcl::console::find_switch(argc, argv, "-test"))
        run_tests = true;

    if (!run_tests && test_scene.compare("") == 0)
    {
        PCL_ERROR("The test_scene parameter is missing\n");
        showHelp(argv[0]);
        exit(-1);
    }

    pcl::console::parse_argument(argc, argv, "--test_dir", test_scenes_dir);

    pcl::console::parse_argument(argc, argv, "--gt_dir", gt_files_dir);

    pcl::console::parse_argument(argc, argv, "--descr", feature_descriptor);

    pcl::console::parse_argument(argc, argv, "--keypoint_det", keypoint_detector);

    pcl::console::parse_argument(argc, argv, "--voc_size", vocabulary_size);

    pcl::console::parse_argument(argc, argv, "--vox_grid_size", voxel_grid_size);

    pcl::console::parse_argument(argc, argv, "--iss_sal_rad_factor", iss_salient_rad_factor);

    pcl::console::parse_argument(argc, argv, "--iss_non_max_rad_factor", iss_non_max_rad_factor);

    pcl::console::parse_argument(argc, argv, "--harris_thresh", harris_thresh);

    pcl::console::parse_argument(argc, argv, "--harris_radius", harris_radius);

    pcl::console::parse_argument(argc, argv, "--th", score_thresh);

    if (pcl::console::find_switch(argc, argv, "-thresh"))
        apply_thresh = true;

    if (pcl::console::find_switch(argc, argv, "-verify"))
        apply_verification = true;

    if (pcl::console::find_switch(argc, argv, "-limit"))
        limit_matches = true;

    if (pcl::console::find_switch(argc, argv, "-voxelize"))
        perform_voxelizing = true;

    if (pcl::console::find_switch(argc, argv, "-h"))
    {
        showHelp(argv[0]);
        exit(0);
    }
}

int main(int argc, char **argv)
{
    pcl::console::print_highlight("\n\n***** Start object recognition *****\n");

    parseCommandLine(argc, argv);

    printParams();

    training_source = Source::Ptr(new Source());

    if (keypoint_detector == "us")
    {
        detector = KeypointDetector::Ptr(new UniformKeypointDetector());
    }
    else if (keypoint_detector == "iss")
    {
        detector = KeypointDetector::Ptr(new ISSKeypointDetector());
    }
    else if (keypoint_detector == "harris")
    {
        detector = KeypointDetector::Ptr(new Harris3DKeypointDetector());
    }

    if (feature_descriptor.compare("shot") == 0)
    {
        feature_estimator = FeatureEstimator<PointType>::Ptr(new FeatureEstimatorSHOT<PointType>);
    }
    else if (feature_descriptor.compare("fpfh") == 0)
    {
        feature_estimator = FeatureEstimator<PointType>::Ptr(new FeatureEstimatorFPFH<PointType>);
    }
    else if (feature_descriptor.compare("pfhrgb") == 0)
    {
        feature_estimator = FeatureEstimator<PointType>::Ptr(new FeatureEstimatorPFHRGB<PointType>);
    }
    else if (feature_descriptor.compare("cshot") == 0)
    {
        feature_estimator = FeatureEstimator<PointType>::Ptr(new FeatureEstimatorColorSHOT<PointType>);
    }

    if (training_dir.find("willow") != string::npos)
    {
        training_dataset = "willow";
    }
    else if (training_dir.find("tuw") != string::npos)
    {
        training_dataset = "tuw";
    }
    else
    {
        training_dataset = "ram";
    }

    PCL_INFO("training dataset: %s\n", training_dataset.c_str());

    if (training_dataset.compare("willow") == 0 && run_tests)
        experiments_dir = "experiments/willow_train_set_tests";
    else if (training_dataset.compare("tuw") == 0 && run_tests)
        experiments_dir = "experiments/tuw_train_set_tests";
    else if (training_dataset.compare("ram") == 0 && run_tests)
        experiments_dir = "experiments/kinect_train_set_tests";

    PCL_INFO("experiments directory: %s\n", experiments_dir.c_str());

    training_source->setPath(training_dir);

    training_source->setUseModelViews(use_partial_views);

    if (test_scene != "")
    {
        std::size_t pos = test_scene.find_last_of("/") + 1;
        scene_name = test_scene.substr(pos);

        scene_name = scene_name.substr(0, scene_name.find(".pcd"));

        pcl::console::print_info("scene_name: %s\n", scene_name.c_str());

        string gt_file = scene_name + ".txt";

        stringstream gt_path_ss;

        if (training_dataset.compare("ram") == 0)
        {
            gt_path_ss << gt_files_dir << "/" << gt_file;

            gt_file_path = gt_path_ss.str();
        }
        else if (training_dataset.compare("willow") == 0)
        {
            gt_files_dir = "willow_gt_files";
            gt_file_path = gt_files_dir;
        }
        else if (training_dataset.compare("tuw") == 0)
        {
            gt_files_dir = "tuw_gt_files";
            gt_file_path = gt_files_dir;
        }

        std::cout << "groundtruth path: " << gt_file_path << "\n";

        if (!boost::filesystem::exists(gt_file_path))
        {
            PCL_ERROR("Ground truth path %s doesn't exist\n", gt_file_path.c_str());
        }
    }

    stringstream path_ss;
    path_ss << training_dir << "/vocab_flann_" << feature_descriptor << ".idx";

    string flann_index_path = path_ss.str();

    path_ss.str("");
    path_ss << training_dir << "/training_data_" << feature_descriptor << ".h5";

    string training_data_file_path = path_ss.str();

    loadBoWModels();

    bow_extractor = BoWModelDescriptorExtractor::Ptr(new BoWModelDescriptorExtractor(flann_index_path, training_data_file_path));

    bow_extractor->loadMatcherIndex();

    if (run_tests)
    {
        runTestMode();
    }
    else
    {
        // Load test scene
        PointCloudPtr scene_cloud(new PointCloud());
        pcl::io::loadPCDFile(test_scene.c_str(), *scene_cloud);

        printf("scene cloud has size: %d\n", static_cast<int>(scene_cloud->points.size()));

        recognizeScene(scene_cloud);
    }

    return 0;
}
