#ifndef BAGOFWORDS_H
#define BAGOFWORDS_H

/**
 * Classes:
 * BoWTrainer
 * BoWModelDescriptorExtractor
 * DescriptorMatcher
 * Structures:
 * DMatch
 */

#include <vector>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <flann/flann.h>
#include "source.h"

typedef std::vector<float> feature_point;

struct DMatch
{
    DMatch(int _query_idx, int _train_idx, float _distance);
    int query_idx;
    int train_idx;
    float distance;
};

/*
 * Class for training of a 'bag of visual words' vocabulary from a set of descriptors
 *
 */
class BoWTrainer
{
public:
    typedef pcl::PointXYZRGB PointInT;
    typedef boost::shared_ptr<BoWTrainer> Ptr;

    explicit BoWTrainer(int cluster_count) noexcept;

    // Add descriptor to the training set of descriptors
    void add(feature_point descriptor);

    void setVocabPath(std::string &vocabulary_path);

    void setCentersInitFlag(std::string centers_init_flag);

    // Actual performing of k-means
    std::vector<feature_point> cluster() noexcept;

    std::vector<feature_point> getDescriptors();

protected:
    // Training descriptors
    std::vector<feature_point> descriptors_;

    // Vocabulary size
    int cluster_count_;
    // The number of training descriptors
    int size_;
    std::string vocab_file_path_;

    // kmeans centers init flag
    int kmeans_centers_init_flag_;
};

/*
 * Flann descriptor matcher
 *
 */

class DescriptorMatcher
{
public:
    typedef flann::ChiSquareDistance<float> DistT;
    typedef boost::shared_ptr<DescriptorMatcher> Ptr;

    DescriptorMatcher(std::string index_file_path, std::string training_data_file_path);

    ~DescriptorMatcher();
    void add(std::vector<feature_point> &descriptors);
    void setSearchIndexParams(std::string search_index_params);
    void clear();
    void train();
    void match(std::vector<feature_point> &query_descriptors, std::vector<DMatch> &matches);
    void knnMatch(std::vector<feature_point> &query_descriptors, std::vector<std::vector<DMatch>> &matches, int knn);
    void convertMatches(std::vector<std::vector<DMatch>> &knn_matches, std::vector<DMatch> &matches);
    void loadIndex(int &data_length);

protected:
    flann::Index<DistT> *index_;
    flann::Matrix<float> data;
    std::vector<feature_point> train_descriptors_;
    int added_desc_count_;
    std::string index_file_path_;
    std::string training_data_file_path_;
    std::string search_index_params_;
};

/*
 * Class to compute image descriptor using bag of visual words.
 *
 */
class BoWModelDescriptorExtractor
{
public:
    typedef pcl::PointXYZRGB PointInT;
    typedef pcl::PointCloud<PointInT>::Ptr PointInTPtr;
    typedef std::vector<float> bow_vector;
    typedef boost::shared_ptr<BoWModelDescriptorExtractor> Ptr;

    BoWModelDescriptorExtractor(std::string index_file_path, std::string data_file_path);

    ~BoWModelDescriptorExtractor();

    void setVocabulary(std::vector<feature_point> vocabulary);
    void setDescriptorSize(int descriptor_size);
    void setSearchIndexParams(std::string search_index_params);
    std::vector<feature_point> getVocabulary();
    int descriptorSize();
    void loadMatcherIndex();
    //    void clear();
    void compute(std::vector<feature_point> model_descriptors, bow_vector &bow_model_descriptor);

protected:
    DescriptorMatcher::Ptr dmatcher_;
    // vocabulary size
    int descriptor_size_;
};

#endif // BAGOFWORDS_H
