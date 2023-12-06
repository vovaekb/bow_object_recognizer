/*
 * Author: Privalov Vladimir, iprivalov@fit.vutbr.cz
 * Date: May 2016
 */

#include <stdio.h>
#include <vector>
#include <fstream>
#include <math.h>
#include <flann/flann.h>
#include <pcl/io/pcd_io.h>
#include <pcl/console/parse.h>
#include <pcl/console/print.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/fpfh.h>
#include <bow_object_recognizer/source.h>
#include <bow_object_recognizer/bagofwords.h>
#include <bow_object_recognizer/feature_estimator.h>
#include <bow_object_recognizer/persistence_utils.h>
#include "pc_source/source.cpp"
#include "bagofwords.cpp"

using namespace std;

typedef pcl::PointXYZRGB PointInT;
typedef vector<float> feature_point;
typedef std::pair<string, std::vector<float> > bow_model_sample;

bool use_partial_views (false);
string training_dir;
string feature_descriptor ("fpfh");
string vocabulary_file;
string flann_index_file ("vocab_flann.idx");
string training_data_file ("training_data.h5");
string word_freq_file ("word_frequencies.txt");

// Algorithm parameters
string knn_search_index_params ("linear");

vector<Model> training_models;

Source::Ptr training_source;
FeatureEstimator<PointInT>::Ptr feature_estimator;


void showHelp(char* filename) {
    std::cout << "*******************************************************************\n";
    std::cout << "*                                                                 *\n";
    std::cout << "*             Training BoW models - Usage Guide                   *\n";
    std::cout << "*                                                                 *\n";
    std::cout << "*******************************************************************\n";
    std::cout << "Using: " << filename << " --train_dir <training directory> [options]\n";
    std::cout << "options:\n";
    std::cout << "--descr <descriptor>:  descriptor to use (shot, fpfh, cshot, pfhrgb)\n";
    std::cout << "--index_params <search_index_params>:    knn-search index parameters\n";
    std::cout << "-partial_views:                          use partial views for object models\n";
    std::cout << "-h:                    show help\n\n";
}

void printParams()
{
    std::cout << "training dir: " << training_dir.c_str() << "\n"
              "descriptor: " << feature_descriptor.c_str() << "\n"
              "use partial views: " << use_partial_views << "\n"
              "knn search index params: " << knn_search_index_params << "\n";
}

void trainModelBowDescriptors(Model& training_model, std::map<string, bow_vector> bow_descriptors)
{
    string model_path = training_source->getModelDir(training_model);

    if(use_partial_views)
    {
        for(size_t j = 0; j < training_model.views.size(); j++)
        {
            std::stringstream descr_file;

            string view_id = training_model.views[j];

            std::cout << "Processing view " << view_id << "\n";

            std::stringstream bow_sample_id;
            bow_sample_id << training_model.model_id << "_" << view_id;

            descr_file << model_path << "/views/" << view_id << "_" << feature_descriptor << "_descr.pcd";

            vector<feature_point> descriptors_vect;

            feature_estimator->loadFeatures(descr_file.str(), descriptors_vect);

            bow_vector& view_bow_descr = bow_descriptors[bow_sample_id.str()];
            bow_extractor->compute(descriptors_vect, view_bow_descr);
        }
    }
    else
    {
        std::stringstream descr_file;

        std::stringstream bow_sample_id;
        bow_sample_id << training_model.model_id;

        descr_file << model_path << "/" << feature_descriptor << "_descr.pcd";

        vector<feature_point> descriptors_vect;

        feature_estimator->loadFeatures(descr_file.str(), descriptors_vect);

        std::cout << "Compute boW for model " << training_model.model_id << "\n";

        bow_vector& model_bow_descr = bow_descriptors[bow_sample_id.str()];
        bow_extractor->compute(descriptors_vect, model_bow_descr);
    }
}

void trainBowDescriptors(BoWModelDescriptorExtractor::Ptr& bow_extractor, std::vector<feature_point>& vocabulary)
{
    cout << "[trainBowDescriptors] Loading models ...\n";

    // Load model descriptors

    string models_dir = "models";

    stringstream path_ss;
    path_ss << training_dir << "/" << models_dir;

    boost::filesystem::path models_path = path_ss.str();

    training_source->getModelsInDir(models_path);

    training_models = training_source->getModels();

    typedef std::vector<float> bow_vector;
    std::map<string, bow_vector> bow_descriptors;

    int training_model_samples_number = training_source->getModelSamplesNumber();

    std::cout << "Training model samples number is: " << training_model_samples_number << "\n";
    std::cout << training_models.size() << " training models have been loaded\n";

    int threads_num = 4;
    std::vector<std::thread> threads;
    int training_models_chunk_size = object_templates.size() / threads_num;
    int training_models_number = training_models.size();

    for (int i = 0; i < threads_num; i++)
    {
        threads.emplace_back([&]() {
            int start = i * training_models_chunk_size;
            int end = (i == threads_num - 1) ? training_models_number : (i + 1) * training_models_chunk_size;
            for (int j = start; j < end; j++)
            {
                auto training_model = training_models[j];

                std::cout << "\n\nProcessing model " << training_model.model_id << "\n";
                trainModelBowDescriptors(&training_model, &bow_descriptors);
            }
        });
    }

    for (auto &&t : threads)
    {
        if (t.joinable())
        {
            t.join();
        }

    }
    threads.clear();

    // Calculate the number of models containing the word
    std::vector<float> word_occurrences;
    word_occurrences.resize(vocabulary.size(), 0);
    for(size_t idx = 0; idx < vocabulary.size(); idx++)
    {
        for(auto it : bow_descriptors)
        {
            std::vector<float> descr = it.second;
            if(descr[idx] > 0)
                word_occurrences[idx] = word_occurrences[idx] + 1.f;
        }
    }

    cout << "Review words occurrences vector\n";

    for(auto & word_occurrence : word_occurrences)
    {
        printf("%d ", static_cast<int>(word_occurrence));
    }

    std::cout << "\n";


    for(auto & training_model : training_models)
    {
        string model_path = training_source->getModelDir(training_model);

        if(use_partial_views)
        {
            for(auto & view_id : training_model.views)
            {
                std::stringstream bow_sample_key;
                bow_sample_key << training_model.model_id << "_" << view_id;

                bow_vector& view_bow_descr = bow_descriptors[bow_sample_key.str()];

                for(size_t idx = 0; idx < view_bow_descr.size(); idx++)
                {
                    float word_occurrences_n = word_occurrences[idx];

                    float word_idf = (word_occurrences_n != 0 ? log10(training_model_samples_number / word_occurrences_n) : 0);

                    view_bow_descr[idx] = word_idf * view_bow_descr[idx];
                }

                // Write vector to file
                std::stringstream bow_descr_path;

                bow_descr_path << model_path << "/views/" << view_id << "_bow_descr.txt";

                cout << "Write vector to file " << bow_descr_path.str() << "\n";

                string bow_descr_file = bow_descr_path.str();

                PersistenceUtils::writeVectorToFile(bow_descr_file, view_bow_descr);

            }
        }
        else
        {
            // Calculation of idf weighted BoW descriptor
            bow_vector& model_bow_descr = bow_descriptors[training_model.model_id];

            for(size_t idx = 0; idx < model_bow_descr.size(); idx++)
            {
                float word_occurrences_n = word_occurrences[idx];

                float word_idf = (word_occurrences_n != 0 ? log10(training_model_samples_number / word_occurrences_n) : 0); // models_n

                model_bow_descr[idx] = word_idf * model_bow_descr[idx];
            }

            std::stringstream bow_descr_path;

            bow_descr_path << model_path << "/bow_descr.txt";

            cout << "Write vector to file " << bow_descr_path.str() << "\n";

            string bow_descr_file = bow_descr_path.str();

            PersistenceUtils::writeVectorToFile(bow_descr_file, model_bow_descr);
        }

    }

//    bow_extractor->clear();
    bow_extractor.reset();

}

int main(int argc, char** argv)
{
    pcl::console::print_highlight("\n\n***** Start training BoW descriptors *****\n");

    pcl::console::parse_argument(argc, argv, "--train_dir", training_dir);

    if(training_dir.compare("") == 0)
    {
        PCL_ERROR("The train_dir parameter is missing\n");
        showHelp(argv[0]);
        return -1;
    }

    pcl::console::parse_argument(argc, argv, "--descr", feature_descriptor);

    pcl::console::parse_argument(argc, argv, "--index_params", knn_search_index_params);

    if(pcl::console::find_switch(argc, argv, "-partial_views"))
        use_partial_views = true;

    if(pcl::console::find_switch(argc, argv, "-h"))
    {
        showHelp(argv[0]);
        return 0;
    }

    printParams();

    vocabulary_file = "vocabulary_" + feature_descriptor + ".pcd";

    cout << "Vocabulary file: " << vocabulary_file << "\n";

    training_source = Source::Ptr(new Source());
    training_source->setPath(training_dir);

    training_source->setUseModelViews(use_partial_views);

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

    vector<feature_point> vocabulary;

    stringstream path_ss;
    path_ss << training_dir << "/vocab_flann_" << feature_descriptor << ".idx";

    string flann_index_path = path_ss.str();

    path_ss.str("");
    path_ss << training_dir << "/training_data_" << feature_descriptor <<".h5";

    string training_data_file_path = path_ss.str();

    // remove flann index file is exists
    if(boost::filesystem::exists(flann_index_path))
    {
        if(remove(flann_index_path.c_str()) != 0)
            perror("Error deleting old flann index file");
        else
            cout << "Old flann index file successfully deleted\n";
    }

    if(boost::filesystem::exists(training_data_file_path))
    {
        if(remove(training_data_file_path.c_str()) != 0)
            perror("Error deleting old training data file");
        else
            cout << "Old training data file successfully deleted\n";
    }

    BoWModelDescriptorExtractor::Ptr bow_extractor(new BoWModelDescriptorExtractor (flann_index_path, training_data_file_path));

    path_ss.str("");
    path_ss << training_dir << "/" << vocabulary_file;
    string vocabulary_file_path = path_ss.str();

    cout << "DescriptorExtractor class is initialized\n";

    feature_estimator->loadFeatures(vocabulary_file_path, vocabulary);

    std::cout << "\nVocabulary size: " << vocabulary.size() << "\n";

    bow_extractor->setSearchIndexParams(knn_search_index_params);
    bow_extractor->setVocabulary(vocabulary);

    trainBowDescriptors(bow_extractor, vocabulary);

    return 0;
}
