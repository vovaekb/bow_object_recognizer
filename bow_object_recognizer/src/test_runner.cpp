#define DEBUG_LOADING_TRAINING_MODELS
#define NEW_CALC_METHOD
#define CALC_RUNTIME
#define SAVE_TEST_SCENE_BOW

#include <math.h>
#include <time.h>
#include <algorithm>
#include <pcl/common/time.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/fpfh_omp.h>
#include <bow_object_recognizer/common.h>
#include <bow_object_recognizer/test_runner.h>
#include <bow_object_recognizer/normal_estimator.h>
#include <bow_object_recognizer/keypoint_detector.h>
#include <bow_object_recognizer/source.h>
#include <bow_object_recognizer/persistence_utils.h>
#include <bow_object_recognizer/sac_alignment.h>


string scene_keypoints_file;

TestRunner::TestRunner(const std::string &tests_base_path, std::string test_setup_name, bool single_case_mode):
    tests_base_path_(tests_base_path), test_setup_name_(test_setup_name), single_case_mode_(single_case_mode)
{
    start_thresh_ = 0.000000064;
    // 0.005 - max threshold value for the case of global estimation of precision due to the fact that higher threshold values give no positives thus are not useful
    // 0.625 - max threshold value for constructing ROC curve
    end_thresh_ = 0.005;
    tests_base_path_ = tests_base_path_ + "/" + test_setup_name_;

    std::cout << "tests_base_path: " << tests_base_path_ << "\n";
}

void TestRunner::initTests() {
    std::cout << "[TestRunner::initTests]\n";

    if(!boost::filesystem::exists(tests_base_path_))
        boost::filesystem::create_directories(tests_base_path_);

    if(single_case_mode_)
    {
        // Set fixed params
        runTestCase(0);
    }
}

void TestRunner::runTestCase(const int test_num) {
    std::stringstream test_path_ss;
    test_path_ss << tests_base_path_;

    if(test_num)
    {
        test_path_ss << "/test_" << test_num;
    }

    test_case_dir_ = test_path_ss.str();

    if(!boost::filesystem::exists(test_case_dir_))
        boost::filesystem::create_directory(test_case_dir_);

    std::cout << "test case dir: " << test_case_dir_ << "\n";

    test_path_ss << "/setup.txt";

    std::string setup_file = test_path_ss.str();

    test_path_ss.str("");

    test_path_ss << tests_base_path_ << "/scores.txt";

    test_scores_file_ = test_path_ss.str();

    test_path_ss.str("");

    test_path_ss << tests_base_path_ << "/runtime.txt";

    time_result_file_ = test_path_ss.str();


    // Write test case params setup to file
    output_.open(setup_file.c_str());
    output_ << "Experiment: " << test_setup_name_ << "\n\n";
    output_ << "vocabulary size: " << vocabulary_size << "\n";
    output_ << "descriptor: " << feature_descriptor << "\n";

    output_.close();

#ifndef NEW_CALC_METHOD
    // Extreme case for ROC curve
    setScoreThreshold(0);

    // Another cases
    for(float th = start_thresh_; th <= end_thresh_+0.001; th *= 5)
    {
        setScoreThreshold(th);
    }
#endif
#ifdef NEW_CALC_METHOD
    iterateTestScenes();
#endif
}

void TestRunner::setScoreThreshold(const float score)
{
    score_thresh = score;

    PCL_INFO("match score threshold: %.3f\n", score_thresh);

    std::stringstream results_path_ss;
    results_path_ss << tests_base_path_ << "/th_" << score_thresh << ".txt";

    test_output_file_ = results_path_ss.str();

    results_path_ss.str("");

    iterateTestScenes();

    std::cout << "\n";
}

void TestRunner::iterateTestScenes() {
    PCL_INFO("Iterate test scenes ...\n");

    boost::filesystem::path test_scenes_path = test_scenes_dir;
    boost::filesystem::directory_iterator end_itr;

    for(boost::filesystem::directory_iterator iter (test_scenes_path); iter != end_itr; ++iter)
    {
        if(boost::filesystem::extension(iter->path()) == ".pcd")
        {
            test_scene = (iter->path()).string();

            pcl::console::print_debug("Load the test scene: %s\n", test_scene.c_str());

            std::size_t pos = test_scene.find_last_of("/") + 1;
            scene_name = test_scene.substr( pos );

            scene_name = scene_name.substr(0, scene_name.find(".pcd"));

            std::cout << "scene_name: " << scene_name << "\n";

            // Specify path to ground truth file
            string gt_file = scene_name + ".txt";

            std::stringstream path_ss;
            if(training_dataset.compare("ram") == 0)
            {
                path_ss << gt_files_dir << "/" << gt_file;
                gt_file_path = path_ss.str();
            }
            else if(training_dataset.compare("willow") == 0)
            {
                gt_files_dir = "willow_gt_files";
                gt_file_path = gt_files_dir;
            }
            else if(training_dataset.compare("tuw") == 0)
            {
                gt_files_dir = "tuw_gt_files";
                gt_file_path = gt_files_dir;
            }

            std::cout << "groundtruth path: " << gt_file_path << "\n";

            if(!boost::filesystem::exists(gt_file_path))
            {
                PCL_ERROR("Ground truth path %s doesn't exist\n", gt_file_path.c_str());
            }

            path_ss.str("");

            path_ss << test_scenes_dir << "/" << feature_descriptor;

            string descr_path = path_ss.str();

            if(!boost::filesystem::exists(descr_path))
            {
                boost::filesystem::create_directory(descr_path);
            }

            path_ss << "/" << scene_name << ".pcd";
            scene_descr_file_path = path_ss.str();

            path_ss.str("");
            path_ss << test_scenes_dir << "/keypoints/" << scene_name << ".pcd";

            scene_keypoints_file = path_ss.str();

            runDetector();
        }
    }

#ifdef CALC_RUNTIME
    // Calculate avarage time over all the scenes
    double max_time = *std::max_element(calc_durations.begin(), calc_durations.end());
    double average_time = std::accumulate(calc_durations.begin(), calc_durations.end(), 0.0)/calc_durations.size();

    output_.open(time_result_file_.c_str(), std::ios::app);
    output_ << "max time: " << max_time << "ms\n";
    output_ << "avg time: " << average_time << "ms\n";
    output_.close();

    calc_durations.clear();
#endif
}

void TestRunner::runDetector() {
#ifdef CALC_RUNTIME
    pcl::StopWatch sw;
#endif

    std::vector<Model> training_models = training_source->getModels();

    PointInTPtr scene_cloud (new pcl::PointCloud<PointInT> ());
    pcl::io::loadPCDFile(test_scene.c_str(), *scene_cloud);


    std::vector<feature_point> descriptors_vect;

    PointInTPtr scene_keypoints (new pcl::PointCloud<PointInT> ());

    // Calculate features
    if(!boost::filesystem::exists(scene_descr_file_path))
    {
        NormalTPtr normals (new pcl::PointCloud<NormalT> ());

        NormalEstimator<PointInT>::Ptr normal_estimator (new NormalEstimator<PointInT>);
        normal_estimator->setDoScaling(false);
        normal_estimator->setGridResolution(voxel_grid_size);
        normal_estimator->setDoVoxelizing(perform_voxelizing);
        normal_estimator->setNormalK(norm_est_k);
        //normal_estimator->setNormalRadius(norm_rad);
        normal_estimator->estimate(scene_cloud, scene_cloud, normals);

        if(keypoint_detector.compare("us") == 0)
        {
            boost::shared_ptr<UniformKeypointDetector> cast_detector = boost::static_pointer_cast<UniformKeypointDetector>(detector);
            cast_detector->setRadius(0.02); // 0.01);
        }
        else if(keypoint_detector.compare("iss") == 0)
        {
            boost::shared_ptr<ISSKeypointDetector> cast_detector = boost::static_pointer_cast<ISSKeypointDetector>(detector);
            cast_detector->setSalientRadiusFactor(iss_salient_rad_factor);
            cast_detector->setNonMaxRadiusFactor(iss_non_max_rad_factor);
        }
        else if(keypoint_detector.compare("harris") == 0)
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
        feature_estimator->calculateFeatures(scene_cloud, scene_keypoints, normals, descriptors_vect);

        feature_estimator->saveFeatures(descriptors_vect, scene_descr_file_path);
    }
    else
    {
        detector->loadKeypoints(scene_keypoints_file, scene_keypoints);

        // Load features from file
        feature_estimator->loadFeatures(scene_descr_file_path, descriptors_vect);
    }

    pcl::PointIndices::Ptr nan_indices = feature_estimator->getNaNIndices();

    // Remove keypoints corresponding to NaN feature points
    if(nan_indices->indices.size() > 0)
    {
        pcl::ExtractIndices<PointInT> extract;
        extract.setInputCloud(scene_keypoints);
        extract.setIndices(nan_indices);
        extract.filter(*scene_keypoints);
    }

    for (size_t i = 0; i < scene_keypoints->size(); ++i)
    {
        if (! pcl_isfinite((*scene_keypoints)[i].x))
        {
            PCL_WARN("Scene keypoint %d has NaN in x\n", (int)i);
        }
        else if(! pcl_isfinite((*scene_keypoints)[i].y))
        {
            PCL_WARN("Scene keypoint %d has NaN in y\n", (int)i);
        }
        else if(! pcl_isfinite((*scene_keypoints)[i].z))
        {
            PCL_WARN("Scene keypoint %d has NaN in z\n", (int)i);
        }
    }


    //
    // Calculate the BoW descriptor for the scene
    //
    bow_vector query_bow_descriptor;

    bow_extractor->compute(descriptors_vect, query_bow_descriptor);

    vocabulary_size = (int)query_bow_descriptor.size();

    // Calculate the number of models containing the word
    std::vector<float> word_occurrences;
    word_occurrences.resize(vocabulary_size, 0);

    std::map<string, bow_vector >::iterator map_it;


    for(size_t idx = 0; idx < vocabulary_size; idx++)
    {
        for(map_it = training_bow_descriptors.begin(); map_it != training_bow_descriptors.end(); map_it++)
        {
            std::vector<float> descr = (*map_it).second;
            if(descr[idx] > 0)
                word_occurrences[idx] = word_occurrences[idx] + 1.f;
        }
    }

    int training_model_samples_number = training_source->getModelSamplesNumber();

    // Calculate idf-index
    for(size_t idx = 0; idx < query_bow_descriptor.size(); idx++)
    {
        float word_occurrences_n = word_occurrences[idx];

        float word_idf = (word_occurrences_n != 0 ? log10(training_model_samples_number / word_occurrences_n) : 0);

        query_bow_descriptor[idx] = word_idf * query_bow_descriptor[idx];
    }

#ifdef SAVE_TEST_SCENE_BOW
    std::stringstream scene_path_ss;

    scene_path_ss << test_scenes_dir << "/bow/" << scene_name << ".txt";

    std::string scene_bow_file = scene_path_ss.str();

    cout << "Write BoW vector to file " << scene_bow_file << "\n";

    PersistenceUtils::writeVectorToFile(scene_bow_file, query_bow_descriptor);

#endif

    //
    // Match scene against training models and rank matches
    //

    std::vector<float> model_match_scores ( training_models.size() );

    if(use_partial_views)
    {
        for(size_t i = 0; i < training_models.size(); i++)
        {
            Model training_model = training_models[i];
            std::vector<float> view_match_scores (training_model.views.size());

            std::cout << "Loading views for model " << training_model.model_id << "\n";

            for(size_t j = 0; j < training_model.views.size(); j++)
            {
                std::string view_id = training_model.views[j];

                std::stringstream bow_view_sample_key;
                bow_view_sample_key << training_model.model_id << "_" << view_id;

                bow_vector view_bow_descriptor = training_bow_descriptors[bow_view_sample_key.str()];

                float bow_dist = 0;
                for(size_t idx = 0; idx < view_bow_descriptor.size(); idx++)
                {
                    bow_dist = bow_dist + view_bow_descriptor[idx] * query_bow_descriptor[idx];
                }

                view_match_scores[j] = bow_dist;
            }

            std::sort(view_match_scores.begin(), view_match_scores.end(), std::greater<float>());

            float best_view_score = view_match_scores[0];

            view_match_scores.clear();

            model_match_scores[i] = best_view_score;
        }
    }
    else
    {
        for(size_t i = 0; i < training_models.size(); i++)
        {
            std::string model_id = training_models[i].model_id;
            bow_vector model_bow_descriptor = training_bow_descriptors[model_id];

            float bow_dist = 0;
            for(size_t idx = 0; idx < model_bow_descriptor.size(); idx++)
            {
                bow_dist = bow_dist + model_bow_descriptor[idx] * query_bow_descriptor[idx];
            }

            model_match_scores[i] = bow_dist;
        }

    }


    std::vector<model_score> best_matches;
    sortModelScores sort_model_scores_op;
    sortModelSACAlignmentScores sort_model_sac_align_scores_op;

    for(size_t i = 0; i < model_match_scores.size(); i++)
    {
//        if(apply_thresh && match_scores[i] < score_thresh) continue;
        Model model = training_models[i];
        model_score match;
        match.model_id = model.model_id;
        match.score = model_match_scores[i];
        best_matches.push_back(match);
    }

#ifdef CALC_RUNTIME
    double recogn_time = sw.getTime();
    calc_durations.push_back(recogn_time);
#endif

    model_match_scores.clear();

    // The best matches are ones with higher score (cosine tends to 1)
    // Display results of the matching

    std::sort(best_matches.begin(), best_matches.end(), sort_model_scores_op);

    if(limit_matches)
    {
        if(apply_verification)
        {
            std::cout << "---------------------------------------\n";
            std::cout << "------ Geometric verification ---------\n";
            std::cout << "---------------------------------------\n";
            std::cout << "---------------------------------------\n";


            if(best_matches.size() > 10)
                best_matches.resize(10); // best models number

            //
            // Geometric verification
            //

            std::stringstream path_ss;

            if(feature_descriptor.compare("fpfh") == 0)
            {
                // Initialize feature type
                typedef pcl::FPFHSignature33 FeatureInT;

                std::cout << "------------------- Set scene scloud to FeatureCloud ------------------\n";

                // Initialize FeatureCloud object
                FeatureCloud<FeatureInT> scene_feature_cloud;
                scene_feature_cloud.setInputCloud(scene_keypoints);
                scene_feature_cloud.loadInputFeatures(scene_descr_file_path);

                std::cout << "Create FeatureCloud objects for best candidates\n";

                SACAlignment<FeatureInT> alignment;

                for(size_t i = 0; i < best_matches.size(); i++)
                {
                    FeatureCloud<FeatureInT> candidate_feature_cloud;

                    string model_id = best_matches[i].model_id;
                    candidate_feature_cloud.setId(model_id);

                    string model_path = training_source->getModelDir(model_id);

                    path_ss.str("");
                    path_ss << model_path << "/" << feature_descriptor << "_descr.pcd";
                    string descr_path = path_ss.str();

                    pcl::PointCloud<FeatureInT>::Ptr candidate_features (new pcl::PointCloud<FeatureInT>);
                    pcl::io::loadPCDFile (descr_path, *candidate_features);

                    // Remove NaNs
                    pcl::PointIndices::Ptr nan_points (new pcl::PointIndices);
                    for(size_t j = 0; j < candidate_features->points.size(); j++)
                    {
                        if(!pcl_isfinite(candidate_features->at(j).histogram[0])) {
                            PCL_WARN("Point %d is NaN\n", (int)j);
                            nan_points->indices.push_back(j);
                        }
                    }

                    PCL_INFO("Processing candidate: %s\n", model_id.c_str());

                    // Remove NaN feature points
                    if(nan_points->indices.size() > 0)
                    {
                        pcl::ExtractIndices<FeatureInT> extract;
                        extract.setInputCloud(candidate_features);
                        extract.setIndices(nan_points);
                        extract.setNegative(true);
                        extract.filter(*candidate_features);
                    }

                    candidate_feature_cloud.setInputFeatures(candidate_features);

                    path_ss.str("");
                    path_ss << model_path << "/keypoints.pcd";

                    string keypoints_cloud_path = path_ss.str();

                    PointInTPtr keypoints (new pcl::PointCloud<PointInT>);
                    pcl::io::loadPCDFile(keypoints_cloud_path, *keypoints);

                    PCL_INFO("Candidate keypoints: %d\n", (int)keypoints->points.size());

                    // Remove keypoints corresponding to NaN feature points
                    if(nan_points->indices.size() > 0)
                    {
                        pcl::ExtractIndices<PointInT> extract;
                        extract.setInputCloud(keypoints);
                        extract.setIndices(nan_points);
                        extract.setNegative(true);
                        extract.filter(*keypoints);
                    }

                    PCL_INFO("Candidate keypoints after removal NaNs: %d\n", (int)keypoints->points.size());

                    candidate_feature_cloud.setInputCloud(keypoints);

                    alignment.addTemplateCloud( candidate_feature_cloud );
                }

                alignment.setTargetCloud(scene_feature_cloud);

                std::vector<SACAlignment<FeatureInT>::Result, Eigen::aligned_allocator<SACAlignment<FeatureInT>::Result> > alignment_results = alignment.alignAll();

                for(size_t i = 0; i < alignment_results.size(); i++)
                {
                    printf ("Fitness score for model %d: %f\n", (int)i, alignment_results[i].fitness_score);
                }

                for(size_t i = 0; i < best_matches.size(); i++)
                {
                    best_matches[i].sac_alignment_score = alignment_results[i].fitness_score;
                }

                std::cout << "Best matches alignment scores:\n";
                for(size_t i = 0; i < best_matches.size(); i++)
                {
                    printf ("sac_alignment_score for %s: %f\n", best_matches[i].model_id.c_str(), best_matches[i].sac_alignment_score);
                }

                std::sort(best_matches.begin(), best_matches.end(), sort_model_sac_align_scores_op);

                std::cout << "Best matches alignment scores after ranking:\n";
                for(size_t i = 0; i < best_matches.size(); i++)
                {
                    printf ("sac_alignment_score for %s: %f\n", best_matches[i].model_id.c_str(), best_matches[i].sac_alignment_score);
                }
            }
            else if(feature_descriptor.compare("shot") == 0)
            {
                // Initialize feature type
                typedef pcl::SHOT352 FeatureInT;

                std::cout << "------------------- Set scene scloud to FeatureCloud ------------------\n";

                // Initialize FeatureCloud object
                FeatureCloud<FeatureInT> scene_feature_cloud;
                scene_feature_cloud.setInputCloud(scene_keypoints); //scene_cloud);
                scene_feature_cloud.loadInputFeatures(scene_descr_file_path);

                pcl::PointCloud<FeatureInT>::Ptr scene_features = scene_feature_cloud.getLocalFeatures();

                for(size_t j = 0; j < scene_features->points.size(); j++)
                {
                    int dimensionality = 352;

                    for(int idx = 0; idx < dimensionality; idx++)
                    {
                        if(!pcl_isfinite(scene_features->at(j).descriptor[idx])) PCL_WARN("Feature point %d has NaN in component %d\n", (int)j, (int)idx);
                    }
                }


                std::cout << "Create FeatureCloud objects for best candidates\n";

                SACAlignment<FeatureInT> alignment;

                for(size_t i = 0; i < best_matches.size(); i++)
                {
                    FeatureCloud<FeatureInT> candidate_feature_cloud;

                    string model_id = best_matches[i].model_id;
                    candidate_feature_cloud.setId(model_id);

                    string model_path = training_source->getModelDir(model_id);

                    path_ss.str("");
                    path_ss << model_path << "/" << feature_descriptor << "_descr.pcd";
                    string descr_path = path_ss.str();

                    pcl::PointCloud<FeatureInT>::Ptr candidate_features (new pcl::PointCloud<FeatureInT>);
                    pcl::io::loadPCDFile (descr_path, *candidate_features);

                    // Remove NaNs
                    pcl::PointIndices::Ptr nan_points (new pcl::PointIndices);
                    for(size_t j = 0; j < candidate_features->points.size(); j++)
                    {
                        if(!pcl_isfinite(candidate_features->at(j).descriptor[0])) {
                            PCL_WARN("Point %d is NaN\n", (int)j);
                            nan_points->indices.push_back(j);
                        }
                    }

                    PCL_INFO("Processing candidate: %s\n", model_id.c_str());

                    // Remove NaN feature points
                    if(nan_points->indices.size() > 0)
                    {
                        pcl::ExtractIndices<FeatureInT> extract;
                        extract.setInputCloud(candidate_features);
                        extract.setIndices(nan_points);
                        extract.setNegative(true);
                        extract.filter(*candidate_features);
                    }

                    candidate_feature_cloud.setInputFeatures(candidate_features);

                    path_ss.str("");
                    path_ss << model_path << "/keypoints.pcd";

                    string keypoints_cloud_path = path_ss.str();

                    PointInTPtr keypoints (new pcl::PointCloud<PointInT>);
                    pcl::io::loadPCDFile(keypoints_cloud_path, *keypoints);

                    PCL_INFO("Candidate keypoints: %d\n", (int)keypoints->points.size());

                    // Remove keypoints corresponding to NaN feature points
                    if(nan_points->indices.size() > 0)
                    {
                        pcl::ExtractIndices<PointInT> extract;
                        extract.setInputCloud(keypoints);
                        extract.setIndices(nan_points);
                        extract.setNegative(true);
                        extract.filter(*keypoints);
                    }

                    candidate_feature_cloud.setInputCloud(keypoints);

                    alignment.addTemplateCloud( candidate_feature_cloud );
                }

                alignment.setTargetCloud(scene_feature_cloud);

                std::vector<SACAlignment<FeatureInT>::Result, Eigen::aligned_allocator<SACAlignment<FeatureInT>::Result> > alignment_results = alignment.alignAll();

                for(size_t i = 0; i < alignment_results.size(); i++)
                {
                    printf ("Fitness score for model %d: %f\n", (int)i, alignment_results[i].fitness_score);
                }

                for(size_t i = 0; i < best_matches.size(); i++)
                {
                    best_matches[i].sac_alignment_score = alignment_results[i].fitness_score;
                }

                std::cout << "Best matches alignment scores:\n";
                for(size_t i = 0; i < best_matches.size(); i++)
                {
                    printf ("sac_alignment_score for %s: %f\n", best_matches[i].model_id.c_str(), best_matches[i].sac_alignment_score);
                }

                std::sort(best_matches.begin(), best_matches.end(), sort_model_sac_align_scores_op);

                std::cout << "Best matches alignment scores after ranking:\n";
                for(size_t i = 0; i < best_matches.size(); i++)
                {
                    printf ("sac_alignment_score for %s: %f\n", best_matches[i].model_id.c_str(), best_matches[i].sac_alignment_score);
                }
            }
            else if(feature_descriptor.compare("pfhrgb") == 0)
            {
                // Initialize feature type
                typedef pcl::PFHRGBSignature250 FeatureInT;

                std::cout << "------------------- Set scene scloud to FeatureCloud ------------------\n";

                // Initialize FeatureCloud object
                FeatureCloud<FeatureInT> scene_feature_cloud;
                scene_feature_cloud.setInputCloud(scene_keypoints);
                scene_feature_cloud.loadInputFeatures(scene_descr_file_path);

                std::cout << "Create FeatureCloud objects for best candidates\n";

                SACAlignment<FeatureInT> alignment;

                for(size_t i = 0; i < best_matches.size(); i++)
                {
                    FeatureCloud<FeatureInT> candidate_feature_cloud;

                    string model_id = best_matches[i].model_id;
                    candidate_feature_cloud.setId(model_id);

                    string model_path = training_source->getModelDir(model_id);

                    path_ss.str("");
                    path_ss << model_path << "/" << feature_descriptor << "_descr.pcd";
                    string descr_path = path_ss.str();

                    pcl::PointCloud<FeatureInT>::Ptr candidate_features (new pcl::PointCloud<FeatureInT>);
                    pcl::io::loadPCDFile (descr_path, *candidate_features);

                    // Remove NaNs
                    pcl::PointIndices::Ptr nan_points (new pcl::PointIndices);
                    for(size_t j = 0; j < candidate_features->points.size(); j++)
                    {
                        if(!pcl_isfinite(candidate_features->at(j).histogram[0])) {
                            PCL_WARN("Point %d is NaN\n", (int)j);
                            nan_points->indices.push_back(j);
                        }
                    }

                    PCL_INFO("Processing candidate: %s\n", model_id.c_str());

                    // Remove NaN feature points
                    if(nan_points->indices.size() > 0)
                    {
                        pcl::ExtractIndices<FeatureInT> extract;
                        extract.setInputCloud(candidate_features);
                        extract.setIndices(nan_points);
                        extract.setNegative(true);
                        extract.filter(*candidate_features);
                    }

                    candidate_feature_cloud.setInputFeatures(candidate_features);

                    path_ss.str("");
                    path_ss << model_path << "/keypoints.pcd";

                    string keypoints_cloud_path = path_ss.str();

                    PointInTPtr keypoints (new pcl::PointCloud<PointInT>);
                    pcl::io::loadPCDFile(keypoints_cloud_path, *keypoints);

                    PCL_INFO("Candidate keypoints: %d\n", (int)keypoints->points.size());

                    // Remove keypoints corresponding to NaN feature points
                    if(nan_points->indices.size() > 0)
                    {
                        pcl::ExtractIndices<PointInT> extract;
                        extract.setInputCloud(keypoints);
                        extract.setIndices(nan_points);
                        extract.setNegative(true);
                        extract.filter(*keypoints);
                    }

                    candidate_feature_cloud.setInputCloud(keypoints);

                    alignment.addTemplateCloud( candidate_feature_cloud );
                }

                alignment.setTargetCloud(scene_feature_cloud);

                std::vector<SACAlignment<FeatureInT>::Result, Eigen::aligned_allocator<SACAlignment<FeatureInT>::Result> > alignment_results = alignment.alignAll();

                for(size_t i = 0; i < alignment_results.size(); i++)
                {
                    printf ("Fitness score for model %d: %f\n", (int)i, alignment_results[i].fitness_score);
                }

                for(size_t i = 0; i < best_matches.size(); i++)
                {
                    best_matches[i].sac_alignment_score = alignment_results[i].fitness_score;
                }

                std::cout << "Best matches alignment scores:\n";
                for(size_t i = 0; i < best_matches.size(); i++)
                {
                    printf ("sac_alignment_score for %s: %f\n", best_matches[i].model_id.c_str(), best_matches[i].sac_alignment_score);
                }

                std::sort(best_matches.begin(), best_matches.end(), sort_model_sac_align_scores_op);

                std::cout << "Best matches alignment scores after ranking:\n";
                for(size_t i = 0; i < best_matches.size(); i++)
                {
                    printf ("sac_alignment_score for %s: %f\n", best_matches[i].model_id.c_str(), best_matches[i].sac_alignment_score);
                }
            }
            else if(feature_descriptor.compare("cshot") == 0)
            {
                // Initialize feature type
                typedef pcl::SHOT1344 FeatureInT;

                std::cout << "------------------- Set scene scloud to FeatureCloud ------------------\n";

                // Initialize FeatureCloud object
                FeatureCloud<FeatureInT> scene_feature_cloud;
                scene_feature_cloud.setInputCloud(scene_keypoints);
                scene_feature_cloud.loadInputFeatures(scene_descr_file_path);

                std::cout << "Create FeatureCloud objects for best candidates\n";


                SACAlignment<FeatureInT> alignment;

                for(size_t i = 0; i < best_matches.size(); i++)
                {
                    FeatureCloud<FeatureInT> candidate_feature_cloud;

                    string model_id = best_matches[i].model_id;
                    candidate_feature_cloud.setId(model_id);

                    string model_path = training_source->getModelDir(model_id);

                    path_ss.str("");
                    path_ss << model_path << "/" << feature_descriptor << "_descr.pcd";
                    string descr_path = path_ss.str();

                    pcl::PointCloud<FeatureInT>::Ptr candidate_features (new pcl::PointCloud<FeatureInT>);
                    pcl::io::loadPCDFile (descr_path, *candidate_features);

                    // Remove NaNs
                    pcl::PointIndices::Ptr nan_points (new pcl::PointIndices);
                    for(size_t j = 0; j < candidate_features->points.size(); j++)
                    {
                        if(!pcl_isfinite(candidate_features->at(j).descriptor[0])) {
                            PCL_WARN("Point %d is NaN\n", (int)j);
                            nan_points->indices.push_back(j);
                        }
                    }

                    PCL_INFO("Processing candidate: %s\n", model_id.c_str());

                    // Remove NaN feature points
                    if(nan_points->indices.size() > 0)
                    {
                        pcl::ExtractIndices<FeatureInT> extract;
                        extract.setInputCloud(candidate_features);
                        extract.setIndices(nan_points);
                        extract.setNegative(true);
                        extract.filter(*candidate_features);
                    }

                    candidate_feature_cloud.setInputFeatures(candidate_features);

                    path_ss.str("");
                    path_ss << model_path << "/keypoints.pcd";

                    string keypoints_cloud_path = path_ss.str();

                    PointInTPtr keypoints (new pcl::PointCloud<PointInT>);
                    pcl::io::loadPCDFile(keypoints_cloud_path, *keypoints);

                    PCL_INFO("Candidate keypoints: %d\n", (int)keypoints->points.size());

                    // Remove keypoints corresponding to NaN feature points
                    if(nan_points->indices.size() > 0)
                    {
                        pcl::ExtractIndices<PointInT> extract;
                        extract.setInputCloud(keypoints);
                        extract.setIndices(nan_points);
                        extract.setNegative(true);
                        extract.filter(*keypoints);
                    }

                    PCL_INFO("Candidate keypoints after removal NaNs: %d\n", (int)keypoints->points.size());

                    candidate_feature_cloud.setInputCloud(keypoints);
                    alignment.addTemplateCloud( candidate_feature_cloud );
                }

                alignment.setTargetCloud(scene_feature_cloud);

                std::vector<SACAlignment<FeatureInT>::Result, Eigen::aligned_allocator<SACAlignment<FeatureInT>::Result> > alignment_results = alignment.alignAll();

                for(size_t i = 0; i < alignment_results.size(); i++)
                {
                    printf ("Fitness score for model %d: %f\n", (int)i, alignment_results[i].fitness_score);
                }

                for(size_t i = 0; i < best_matches.size(); i++)
                {
                    best_matches[i].sac_alignment_score = alignment_results[i].fitness_score;
                }

                std::cout << "Best matches alignment scores:\n";
                for(size_t i = 0; i < best_matches.size(); i++)
                {
                    printf ("sac_alignment_score for %s: %f\n", best_matches[i].model_id.c_str(), best_matches[i].sac_alignment_score);
                }

                std::sort(best_matches.begin(), best_matches.end(), sort_model_sac_align_scores_op);

                std::cout << "Best matches alignment scores after ranking:\n";
                for(size_t i = 0; i < best_matches.size(); i++)
                {
                    printf ("sac_alignment_score for %s: %f\n", best_matches[i].model_id.c_str(), best_matches[i].sac_alignment_score);
                }
            }

        }

        best_matches.resize(5);

        output_.open(test_scores_file_.c_str(), std::ios::app);
        output_ << scene_name << "\n";

        for(int i = 0; i < best_matches.size(); i++)
        {
            std::string model_id = best_matches[i].model_id;

            bool is_present = false;

            if(training_dataset.compare("ram") == 0)
            {
                is_present = PersistenceUtils::modelPresents(gt_file_path, model_id);
            }
            else if(training_dataset.compare("willow") == 0 || training_dataset.compare("tuw") == 0)
            {
                is_present = PersistenceUtils::modelPresents(gt_file_path, scene_name, model_id);
            }

            cout << model_id << ": " << is_present << "\n"; // " - " << match.score << "\n";

            cout << best_matches[i].score << ":" << is_present << "\n";

            output_ << best_matches[i].score << ":" << is_present << "\n";
        }

        output_ << "\n";
        output_.close();
    }
    else
    {
        cout << "\nFinal matches\n";

        // Save match scores to file

        output_.open(test_scores_file_.c_str(), std::ios::app);
        output_ << scene_name << "\n";

        for(size_t i = 0; i < training_models.size(); i++)
        {
            std::string train_model_id = training_models[i].model_id;
            bool is_present = false;

            if(training_dataset.compare("ram") == 0)
            {
                is_present = PersistenceUtils::modelPresents(gt_file_path, train_model_id);
            }
            else if(training_dataset.compare("willow") == 0 || training_dataset.compare("tuw") == 0)
            {
                is_present = PersistenceUtils::modelPresents(gt_file_path, scene_name, train_model_id);
            }

            model_score match;
            for(int j = 0; j < best_matches.size(); j++)
            {
                if(best_matches[j].model_id == train_model_id)
                {
                    match = best_matches[j];
                    break;
                }
            }

            cout << train_model_id << ": " << is_present << "\n"; // " - " << match.score << "\n";

            output_ << match.score << ":" << is_present << "\n";
        }

        output_ << "\n";
        output_.close();
    }

    std::cout << "\n\n";
}
