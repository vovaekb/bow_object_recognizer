/*
 * Author: Privalov Vladimir, iprivalov@fit.vutbr.cz
 * Date: May 2016
 */


#include <stdio.h>
#include <vector>
#include <math.h>
#include <ctime>
#include <fstream>

#include <pcl/common/time.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>
#include <pcl/common/time.h>
#include <pcl/console/parse.h>
#include <pcl/console/print.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/pfh.h>
#include <pcl/features/pfhrgb.h>
#include <bow_object_recognizer/source.h>
#include <bow_object_recognizer/bagofwords.h>
#include <bow_object_recognizer/normal_estimator.h>
#include <bow_object_recognizer/keypoint_detector.h>
#include <bow_object_recognizer/feature_estimator.h>
#include "pc_source/source.cpp"
#include "bagofwords.cpp"

using namespace std;

typedef pcl::PointXYZRGB PointInT;
typedef pcl::Normal NormalT;
typedef pcl::PointCloud<PointInT>::Ptr PointInTPtr;
typedef pcl::PointCloud<PointInT>::ConstPtr PointInTConstPtr;
typedef pcl::PointCloud<NormalT>::Ptr NormalTPtr;
typedef std::vector<float> feature_point;

// Global parameters
string training_dir;

std::string training_dataset;

string feature_descriptor ("fpfh");
string keypoint_detector ("us");
string vocabulary_file;
string kmeans_centers_init_flag ("random_centers");
int vocabulary_size (500);
bool use_partial_views (false);
bool perform_scaling (false);
bool perform_voxelizing (false);
bool save_descriptors (false);

string word_freq_file ("word_frequencies.txt");
string calc_time_file ("calc_time.txt");
string models_word_freq_file ("models_word_frequences.txt");
string scene_word_freq_file ("scene_word_frequences.txt");


int descr_length;

// Algorithm parameters
float voxel_grid_size (0.001f);
int norm_est_k (10);
float descr_rad (0.05f); // 0.02f
float norm_rad (0.02f); // 0.01f

// ISS
int iss_salient_rad_factor (6); //  5
int iss_non_max_rad_factor (4); // 3
// SIFT
float sift_min_scale (0.025);
int sift_nr_octaves (2);
int sift_nr_scales (3);
float sift_min_contrast (1);
float sift_radius (0.2);
// Harris
float harris_radius (0.01);
float harris_thresh (0.0001);

int pfhrgb_k (50);

float us_radius;

vector<Model> training_models;
vector<string> training_scenes;

//Source training_source;
Source::Ptr training_source;
KeypointDetector::Ptr detector;
FeatureEstimator<PointInT>::Ptr feature_estimator;

void trainVocabulary(string vocabulary_path)
{
    cout << "\nTrain vocabulary\n";

    BoWTrainer::Ptr bow_trainer(new BoWTrainer(vocabulary_size));

    bow_trainer->setCentersInitFlag(kmeans_centers_init_flag);


    if(keypoint_detector.compare("us") == 0)
    {
        boost::shared_ptr<UniformKeypointDetector> cast_detector = boost::static_pointer_cast<UniformKeypointDetector>(detector);
        cast_detector->setRadius(0.01);
    }
    else if(keypoint_detector.compare("iss") == 0)
    {
        boost::shared_ptr<ISSKeypointDetector> cast_detector = boost::static_pointer_cast<ISSKeypointDetector>(detector);
        cast_detector->setSalientRadiusFactor(iss_salient_rad_factor);
        cast_detector->setNonMaxRadiusFactor(iss_non_max_rad_factor);
    }
    else if(keypoint_detector.compare("sift") == 0)
    {
        boost::shared_ptr<SIFTKeypointDetector> cast_detector = boost::static_pointer_cast<SIFTKeypointDetector>(detector);
        cast_detector->setRadius(sift_radius);
        cast_detector->setScales(sift_min_scale, sift_nr_octaves, sift_nr_scales);
        cast_detector->setMinContrast(sift_min_contrast);
    }
    else if(keypoint_detector.compare("harris") == 0)
    {
        boost::shared_ptr<Harris3DKeypointDetector> cast_detector = boost::static_pointer_cast<Harris3DKeypointDetector>(detector);
        cast_detector->setThreshold(harris_thresh);
        cast_detector->setRadius(harris_radius);
    }


    // Iterate over all training models
    for(auto & training_model : training_models)
    {
        printf("\n---------------- Loading model %s ---------------\n", training_model.model_id.c_str());

        string model_path = training_source->getModelDir(training_model);

        if(use_partial_views)
        {
            for(auto & view_id : training_model.views)
            {
                std::stringstream view_descr_file;

                view_descr_file << model_path << "/views/" << view_id << "_" << feature_descriptor << "_descr.pcd";

                string descr_file = view_descr_file.str();

                if(!boost::filesystem::exists(descr_file))
                {
                    std::cout << "Descriptor for view " << view_id << " does not exist\n";

                    cout << "----------- Process view cloud -------------\n";

                    PointInTPtr view_cloud (new pcl::PointCloud<PointInT> ());
                    NormalTPtr normals (new pcl::PointCloud<NormalT> ());
                    PointInTPtr keypoints (new pcl::PointCloud<PointInT> ());

                    std::stringstream view_file;

                    view_file << model_path << "/views/" << view_id << ".pcd";
                    pcl::io::loadPCDFile(view_file.str(), *view_cloud);

                    std::cout << "View cloud has " << view_cloud->points.size() << " points\n";


                    NormalEstimator<PointInT>::Ptr normal_estimator (new NormalEstimator<PointInT>);
                    normal_estimator->setDoScaling(perform_scaling);
                    normal_estimator->setDoVoxelizing(perform_voxelizing);
                    normal_estimator->setGridResolution(voxel_grid_size);
                    normal_estimator->setNormalK(norm_est_k);
                    //normal_estimator->setNormalRadius(norm_rad);
                    normal_estimator->estimate(view_cloud, view_cloud, normals);


                    // Extract keypoints
                    detector->detectKeypoints(view_cloud, keypoints);

                    std::stringstream keypoints_file;
                    keypoints_file << model_path << "/views/" << view_id << "_keypoints.pcd";


                    detector->saveKeypoints(keypoints, keypoints_file.str());

                    std::vector<feature_point> descr_vector;

                    feature_estimator->setSupportRadius(descr_rad);

                    feature_estimator->calculateFeatures(view_cloud, keypoints, normals, descr_vector);

                    if(!descr_vector.size()) continue;

                    if(save_descriptors)
                    {
                        feature_estimator->saveFeatures(descr_vector, descr_file);
                    }

                    cout << "Pass descriptors to the BoW trainer\n\n";

                    for(auto & descriptor : descr_vector)
                    {
                        bow_trainer->add(descriptor);
                    }
                }
                else
                {
                    cout << "-------------- Load descriptors from file -----------------------\n";

                    // Load descriptors from file
                    std::vector<feature_point> descr_vector;
                    feature_estimator->loadFeatures(descr_file, descr_vector);

                    cout << "Pass descriptors to the BoW trainer\n\n";

                    for(auto & descriptor : descr_vector)
                    {
                        bow_trainer->add(descriptor);
                    }
                }
            }
        }
        else
        {
            PointInTPtr model_cloud (new pcl::PointCloud<PointInT> ());
            PointInTPtr model_processed (new pcl::PointCloud<PointInT> ());

            pcl::io::loadPCDFile(training_model.cloud_path.c_str(), *model_cloud);

            std::stringstream path_ss;
            path_ss << model_path << "/" << feature_descriptor << "_descr.pcd";
            string descr_file = path_ss.str();

            std::cout << "Cloud has " << model_cloud->points.size() << "\n";


            // Process cloud and calculate descriptors
            if(!boost::filesystem::exists(descr_file))
            {
                cout << "----------- Process cloud -------------\n";

                NormalTPtr normals (new pcl::PointCloud<NormalT> ());
                PointInTPtr keypoints (new pcl::PointCloud<PointInT> ());

                NormalEstimator<PointInT>::Ptr normal_estimator (new NormalEstimator<PointInT>);
                normal_estimator->setDoScaling(perform_scaling);
                normal_estimator->setDoVoxelizing(perform_voxelizing);
                normal_estimator->setGridResolution(voxel_grid_size);
                normal_estimator->setNormalK(norm_est_k);
                //normal_estimator->setNormalRadius(norm_rad);
                normal_estimator->estimate(model_cloud, model_processed, normals);


                // Extract keypoints
                detector->detectKeypoints(model_cloud, keypoints);

                path_ss.str("");

                path_ss << model_path << "/keypoints.pcd";

                string keypoints_file = path_ss.str();

                detector->saveKeypoints(keypoints, keypoints_file);

                // Save the cloud processed

                path_ss.str("");
                path_ss << model_path << "/model_processed.pcd";

                string processed_path = path_ss.str();

                if(!boost::filesystem::exists(processed_path))
                {
                    pcl::io::savePCDFileASCII(processed_path.c_str(), *model_processed);
                }


                std::vector<feature_point> descr_vector;

                feature_estimator->setSupportRadius(descr_rad);

                feature_estimator->calculateFeatures(model_processed, keypoints, normals, descr_vector);

                if(!descr_vector.size()) continue;

                if(save_descriptors)
                {
                    feature_estimator->saveFeatures(descr_vector, descr_file);
                }

                cout << "Pass descriptors to the BoW trainer\n\n";

                for(auto & descriptor : descr_vector)
                {
                    bow_trainer->add(descriptor);
                }

            }
            else
            {
                cout << "-------------- Load descriptors from file -----------------------\n";

                // Load descriptors from file
                std::vector<feature_point> descr_vector;
                feature_estimator->loadFeatures(descr_file, descr_vector);

                cout << "Pass descriptors to the BoW trainer\n\n";

                for(auto & descriptor : descr_vector)
                {
                    bow_trainer->add(descriptor);
                }

            }
        }

    }


    // Setup parameters for keypoint detector
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
    else if(keypoint_detector.compare("sift") == 0)
    {
        boost::shared_ptr<SIFTKeypointDetector> cast_detector = boost::static_pointer_cast<SIFTKeypointDetector>(detector);
        cast_detector->setRadius(sift_radius);
        cast_detector->setScales(sift_min_scale, sift_nr_octaves, sift_nr_scales);
        cast_detector->setMinContrast(sift_min_contrast);
    }
    if(keypoint_detector.compare("harris") == 0)
    {
        boost::shared_ptr<Harris3DKeypointDetector> cast_detector = boost::static_pointer_cast<Harris3DKeypointDetector>(detector);
        cast_detector->setThreshold(harris_thresh);
        cast_detector->setRadius(harris_radius);
    }


    for(auto & scene_file : training_scenes)
    {
        PointInTPtr scene_cloud (new pcl::PointCloud<PointInT> ());

        pcl::io::loadPCDFile(scene_file.c_str(), *scene_cloud);

        stringstream path_ss;
        vector<string> strs;
        boost::split(strs, scene_file, boost::is_any_of("/"));
        string scene_name = strs[strs.size() - 1];

        path_ss << training_dir << "/scenes/descrs/" << feature_descriptor;

        string descr_dir = path_ss.str();

        if(!boost::filesystem::exists(descr_dir))
            boost::filesystem::create_directory(descr_dir);

        path_ss << "/" << scene_name;
        string descr_file = path_ss.str();

        if(!boost::filesystem::exists(descr_file))
        {
            NormalTPtr normals (new pcl::PointCloud<NormalT> ());
            PointInTPtr keypoints (new pcl::PointCloud<PointInT> ());

            NormalEstimator<PointInT>::Ptr normal_estimator (new NormalEstimator<PointInT>);

            normal_estimator->setDoScaling(false);
            normal_estimator->setDoVoxelizing(perform_voxelizing);
            normal_estimator->setGridResolution(voxel_grid_size);
            normal_estimator->setNormalK(norm_est_k);
            //normal_estimator->setNormalRadius(norm_rad);
            normal_estimator->estimate(scene_cloud, scene_cloud, normals);

            // Detect keypoints
            detector->detectKeypoints(scene_cloud, keypoints);

            std::vector<feature_point> descr_vector;

            feature_estimator->setSupportRadius(descr_rad);

            feature_estimator->calculateFeatures(scene_cloud, keypoints, normals, descr_vector);

            if(!descr_vector.size()) continue;

            feature_estimator->saveFeatures(descr_vector, descr_file);

            for(auto & descriptor : descr_vector)
            {
                bow_trainer->add(descriptor);
            }

        }
        else
        {
            cout << "-------------- Load descriptors from file -----------------------\n";

            // Load descriptors from file
            std::vector<feature_point> descr_vector;
            feature_estimator->loadFeatures(descr_file, descr_vector);

            for(auto & descriptor : descr_vector.views)
            {
                bow_trainer->add(descriptor);
            }

        }
    }


    cout << "-------- Clustering ---------------\n";

    vector<feature_point> vocabulary = bow_trainer->cluster();

    printf("Vocabulary file path: %s\n", vocabulary_path.c_str());

    feature_estimator->saveFeatures(vocabulary, vocabulary_path);
}

/*
  * Command line utilities
*/

void showHelp(char* filename) {
    std::cout << "**************************************************************************************************\n";
    std::cout << "*                                                                                                *\n";
    std::cout << "*                             Building visual vocabulary - Usage Guide                           *\n";
    std::cout << "*                                                                                                *\n";
    std::cout << "**************************************************************************************************\n";
    std::cout << "Usage: " << filename << " --train_dir <training directory> [options]\n";
    std::cout << "options:\n";
    std::cout << "--descr <descriptor>:                                 descriptor to use (shot, fpfh, pfh, pfhrgb)\n";
    std::cout << "--descr_rad <descriptor>:                             descriptor radius\n";
    std::cout << "--keypoint_det <detector>:                            keypoint detector to use (iss, us)\n";
    std::cout << "--voc_size <vocabulary size>:                         the number of visual words in the vocabulary\n";
    std::cout << "--vox_grid_size <vox_grid_size>:                      voxel grid size\n";
    std::cout << "--iss_sal_rad_factor <iss_sal_rad_factor>:            salience radius factor for ISS\n";
    std::cout << "--iss_non_max_rad_factor <iss_non_max_rad_factor>:    non maximum radius factor for ISS\n";
    std::cout << "--sift_min_scale <sift_min_scale>:                    minimum scale for SIFT\n";
    std::cout << "--sift_nr_octaves <sift_nr_octaves>:                  nr octaves for SIFT\n";
    std::cout << "--sift_nr_scales <sift_nr_scales>:                    nr scales for SIFT\n";
    std::cout << "--sift_min_contrast <sift_min_contrast>:              minimum contrast for SIFT\n";
    std::cout << "--sift_radius <sift_radius>:                          radius for SIFT\n";
    std::cout << "--harris_radius <harris_radius>:                      radius for Harris\n";
    std::cout << "--harris_thresh <harris_thresh>:                      outlier threshold for Harris\n";
    std::cout << "--pfhrgb_k <pfhrgb_k>:                                number k for k-neighbor search in PFHRGB estimator\n";
    std::cout << "--centers_init <centers_init_flag>:                   flag for initial centers sampling for kmeans\n";
    std::cout << "-partial_views:                                       use partial views for object models\n";
    std::cout << "-scale:                                               perform scaling for training models\n";
    std::cout << "-voxelize:                                            perform voxelizing for training point clouds\n";
    std::cout << "-save_descr:                                          save descriptors for training models\n";
    std::cout << "-h:                                                   show help\n\n";
}

void printParams()
{
    cout << "training dir: " << training_dir.c_str() << "\n"
            "descriptor: " << feature_descriptor.c_str() << "\n"
            "descriptor radius: " << descr_rad << "\n"
            "keypoint detector: " << keypoint_detector.c_str() << "\n"
            "vocabulary size: " << vocabulary_size << "\n"
            "use partial views: " << use_partial_views << "\n"
            "perform scale: " << (perform_scaling ? "true" : "false") << "\n"
            "perform voxelizing: " << (perform_voxelizing ? "true" : "false") << "\n"
            "save descriptors: " << (save_descriptors ? "true" : "false") << "\n"
            "voxel grid size: " << voxel_grid_size << "\n"
            "harris threshold: " << harris_thresh << "\n"
            "kmeans center initialization: " << kmeans_centers_init_flag << "\n"
            "harris radius: " << harris_radius << "\n";
}

void parseCommandLine(int argc, char** argv)
{
    pcl::console::parse_argument(argc, argv, "--train_dir", training_dir);

    if(training_dir.compare("") == 0)
    {
        PCL_ERROR("The train_dir parameter is missing\n");
        showHelp(argv[0]);
        exit(-1);
    }

    pcl::console::parse_argument(argc, argv, "--descr", feature_descriptor);

    pcl::console::parse_argument(argc, argv, "--descr_rad", descr_rad);

    pcl::console::parse_argument(argc, argv, "--keypoint_det", keypoint_detector);

    pcl::console::parse_argument(argc, argv, "--voc_size", vocabulary_size);

    if(pcl::console::find_switch(argc, argv, "-partial_views"))
        use_partial_views = true;

    if(pcl::console::find_switch(argc, argv, "-scale"))
        perform_scaling = true;

    if(pcl::console::find_switch(argc, argv, "-voxelize"))
        perform_voxelizing = true;

    if(pcl::console::find_switch(argc, argv, "-save_descr"))
        save_descriptors = true;

    pcl::console::parse_argument(argc, argv, "--vox_grid_size", voxel_grid_size);

    pcl::console::parse_argument(argc, argv, "--iss_sal_rad_factor", iss_salient_rad_factor);

    pcl::console::parse_argument(argc, argv, "--iss_non_max_rad_factor", iss_non_max_rad_factor);

    pcl::console::parse_argument(argc, argv, "--sift_min_scale", sift_min_scale);
    pcl::console::parse_argument(argc, argv, "--sift_nr_octaves", sift_nr_octaves);
    pcl::console::parse_argument(argc, argv, "--sift_nr_scales", sift_nr_scales);
    pcl::console::parse_argument(argc, argv, "--sift_min_contrast", sift_min_contrast);
    pcl::console::parse_argument(argc, argv, "--sift_radius", sift_radius);

    pcl::console::parse_argument(argc, argv, "--harris_radius", harris_radius);
    pcl::console::parse_argument(argc, argv, "--harris_thresh", harris_thresh);

    pcl::console::parse_argument(argc, argv, "--pfhrgb_k", pfhrgb_k);

    pcl::console::parse_argument(argc, argv, "--centers_init", kmeans_centers_init_flag);

    if(pcl::console::find_switch(argc, argv, "-h"))
    {
        showHelp(argv[0]);
        exit(0);
    }
}

int main(int argc, char** argv)
{
    parseCommandLine(argc, argv);

    printParams();

    vocabulary_file = "vocabulary_" + feature_descriptor + ".pcd";

    cout << "Vocabulary file: " << vocabulary_file << "\n";

    // Load models from the training directory
    training_source = Source::Ptr(new Source());

    if(keypoint_detector == "us")
    {
        detector = KeypointDetector::Ptr( new UniformKeypointDetector() );
    }
    else if(keypoint_detector == "iss")
    {
        detector = KeypointDetector::Ptr( new ISSKeypointDetector() );
    }
    else if(keypoint_detector == "sift")
    {
        detector = KeypointDetector::Ptr( new SIFTKeypointDetector() );
    }
    else if(keypoint_detector == "harris")
    {
        detector = KeypointDetector::Ptr( new Harris3DKeypointDetector() );
    }

    if(feature_descriptor.compare("shot") == 0)
    {
        feature_estimator = FeatureEstimator<PointInT>::Ptr( new FeatureEstimatorSHOT<PointInT> );
    }
    else if(feature_descriptor.compare("fpfh") == 0)
    {
        feature_estimator = FeatureEstimator<PointInT>::Ptr( new FeatureEstimatorFPFH<PointInT> );
    }
    else if(feature_descriptor.compare("pfhrgb") == 0)
    {
        feature_estimator = FeatureEstimator<PointInT>::Ptr( new FeatureEstimatorPFHRGB<PointInT> );
    }
    else if(feature_descriptor.compare("cshot") == 0)
    {
        feature_estimator = FeatureEstimator<PointInT>::Ptr( new FeatureEstimatorColorSHOT<PointInT> );
    }

    if(training_dir.find("willow") != string::npos)
    {
        training_dataset = "willow";
    }
    else if(training_dir.find("tuw") != string::npos)
    {
        training_dataset = "tuw";
    }
    else
    {
        training_dataset = "ram";
    }

    PCL_INFO("training dataset: %s\n", training_dataset.c_str());

    if(training_dataset.compare("ram") == 0)
        perform_scaling = true;

    training_source->setPath(training_dir);

    training_source->setUseModelViews(use_partial_views);

    training_source->printPath();

    string models_dir = "models";

    stringstream path_ss;
    path_ss << training_dir << "/" << models_dir;

    boost::filesystem::path models_path = path_ss.str();

    training_source->getModelsInDir(models_path);

    training_models = training_source->getModels();

    // Load scenes
    string scenes_dir = "scenes";
    path_ss.str("");

    path_ss << training_dir << "/" << scenes_dir;

    boost::filesystem::path scenes_path = path_ss.str();

    training_source->getScenesInDir(scenes_path);

    training_scenes = training_source->getScenes();

    path_ss.str("");

    path_ss << training_dir << "/" << vocabulary_file;

    string vocabulary_path = path_ss.str();

    std::cout << "Vocabulary path: " << vocabulary_path << "\n";

    trainVocabulary(vocabulary_path);

    return 0;
}
