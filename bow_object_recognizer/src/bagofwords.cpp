#define USE_OPENCV_KMEANS
// #define DEBUG_DESCRIPTORS_COUNT
#define DISABLE_DEBUG_LOGS

#include <string_view>
#include <utility>
#include <iostream>
#include <stdio.h>
#include <cmath>
#include <math.h>
#include <ctime>
#include <flann/io/hdf5.h>
#include <pcl/console/print.h>
#include <pcl/common/time.h>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <bow_object_recognizer/bagofwords.h>

using namespace cv;

int isnan_float(float f) { return (f != f); }

DMatch::DMatch(int _query_idx, int _train_idx, float _distance) : query_idx(_query_idx), train_idx(_train_idx), distance(_distance) {}

BoWTrainer::BoWTrainer(int cluster_count)
{
    cluster_count_ = cluster_count;
    size_ = 0;
}

void BoWTrainer::add(feature_point descriptor)
{
    descriptors_.push_back(std::move(descriptor));
    size_ = size_ + 1;
}

void BoWTrainer::setCentersInitFlag(std::string_view centers_init_flag)
{
    if (centers_init_flag == "random_centers")
    {
        kmeans_centers_init_flag_ = KMEANS_RANDOM_CENTERS;

        PCL_INFO("KMEANS_RANDOM_CENTERS initialization flag is set");
    }
    else
    {
        kmeans_centers_init_flag_ = KMEANS_PP_CENTERS;

        PCL_INFO("KMEANS_PP_CENTERS initialization flag is set");
    }
}

std::vector<feature_point> BoWTrainer::cluster()
{
    std::cout << "Clustering descriptors ...\n";

    std::cout << "Descriptors count: " << descriptors_.size() << "\n";

    int descr_length = static_cast<int>(descriptors_[0].size());

    std::vector<feature_point> vocabulary;

#ifndef DEBUG_DESCRIPTORS_COUNT

    // Convert input data to OpenCV data
    Mat points(size_, descr_length, CV_32F);
    Mat labels;
    Mat centers(cluster_count_, descr_length, points.type());

    for (int i = 0; i < points.rows; i++)
    {
        feature_point descriptor = descriptors_[i];

        for (int j = 0; j < points.cols; j++)
        {
            points.at<float>(i, j) = descriptor[j];
        }
    }

    // TermCriteria::EPS+TermCriteria::COUNT, 10, 1.0
    cv::kmeans(points, cluster_count_, labels, cv::TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 10000, 0.0001), 3, kmeans_centers_init_flag_, centers);

    for (int i = 0; i < centers.rows; i++)
    {
        feature_point center_descriptor(descr_length);

        if (isinf(centers.at<float>(i, 0)))
        {
            printf("centroid %d has 'inf' values\n", i);
            continue;
        }

        for (int j = 0; j < centers.cols; j++)
        {
            center_descriptor[j] = centers.at<float>(i, j);
        }

        vocabulary.push_back(std::move(center_descriptor));
    }

    printf("\nvocabulary size: %d\n", (int)vocabulary.size());
#endif

    return vocabulary;
}

std::vector<feature_point> BoWTrainer::getDescriptors()
{
    return descriptors_;
}

// DescriptorMatcher
DescriptorMatcher::DescriptorMatcher(std::string_view index_file_path, std::string_view training_data_file_path) : index_file_path_(index_file_path), training_data_file_path_(training_data_file_path) {}

DescriptorMatcher::~DescriptorMatcher()
{
    clear();
}

void DescriptorMatcher::add(std::vector<feature_point> &descriptors)
{
    train_descriptors_.emplace(train_descriptors_.end(), std::move(descriptors));
}

void DescriptorMatcher::setSearchIndexParams(std::string search_index_params)
{
    search_index_params_ = search_index_params;
}

void DescriptorMatcher::clear()
{
    train_descriptors_.clear();

    if (data.rows != 0 || data.cols != 0)
        delete[] data.ptr();
}

void DescriptorMatcher::train()
{
    // Fill a flann matrix with training data and build the flann index
    data = flann::Matrix<float>(new float[train_descriptors_.size() * train_descriptors_[0].size()], train_descriptors_.size(), train_descriptors_[0].size());

    size_t i = 0;
    std::for_each(train_descriptors_.begin(), train_descriptors_.end(), [&i, &data](feature_point &descriptor)
                  {
        for(size_t j = 0; j < descriptor.size(); j++)
        {
            data[i][j] = descriptor[j];
        }
        j++; });

    if (search_index_params_ == "linear")
    {
        PCL_INFO("Linear index params were applied\n");

        index_ = new flann::Index<DistT>(data, flann::LinearIndexParams());
    }
    else
    {
        PCL_INFO("KdTree index params were applied\n");

        index_ = new flann::Index<DistT>(data, flann::KDTreeIndexParams(4));
    }
    index_->buildIndex();

    flann::save_to_file(data, training_data_file_path_, "training_data");

    index_->save(index_file_path_);
}

void DescriptorMatcher::match(std::vector<feature_point> &query_descriptors, std::vector<DMatch> &matches)
{
    std::vector<std::vector<DMatch>> knn_matches;

    knnMatch(query_descriptors, knn_matches, 1);
    convertMatches(knn_matches, matches);
}

void DescriptorMatcher::knnMatch(std::vector<feature_point> &query_descriptors, std::vector<std::vector<DMatch>> &matches, int knn)
{
    matches.clear();

    // FLANN kNN search
    for (size_t i = 0; i < query_descriptors.size(); i++)
    {
        feature_point descriptor = query_descriptors[i];
        int descr_length = (int)descriptor.size();

        flann::Matrix<float> p = flann::Matrix<float>(new float[descr_length], 1, descr_length);

        for (int idx = 0; idx < descr_length; idx++)
        {
            p[0][idx] = descriptor[idx];
        }

        flann::Matrix<int> indices(new int[knn], 1, knn);
        flann::Matrix<float> distances(new float[knn], 1, knn);
        index_->knnSearch(p, indices, distances, knn, flann::SearchParams(512));

        std::vector<DMatch> descr_matches;
        for (size_t j = 0; j < knn; j++)
        {
            DMatch match(i, indices[0][j], distances[0][j]);
            descr_matches.push_back(std::move(match));
        }

        matches.push_back(std::move(descr_matches));

        delete[] p.ptr();
    }
}

void DescriptorMatcher::convertMatches(std::vector<std::vector<DMatch>> &knn_matches, std::vector<DMatch> &matches)
{
    matches.clear();
    matches.reserve(knn_matches.size());

    for (auto &match : knn_matches)
    {
        if (!match.empty())
            matches.push_back(match[0]);
    }
}

void DescriptorMatcher::loadIndex(int &data_length)
{
    std::cout << "Loading training data from file " << training_data_file_path_ << "\n";

    flann::load_from_file(data, training_data_file_path_, "training_data");

    data_length = (int)data.rows;

    std::cout << "Loading index from file " << index_file_path_ << "\n";

    index_ = new flann::Index<DistT>(data, flann::SavedIndexParams(index_file_path_));
    index_->buildIndex();
}

// BoWModelDescriptorExtractor
BoWModelDescriptorExtractor::BoWModelDescriptorExtractor(std::string index_file_path, std::string data_file_path)
{
    dmatcher_ = DescriptorMatcher::Ptr(new DescriptorMatcher(index_file_path, data_file_path));
}

void BoWModelDescriptorExtractor::setVocabulary(std::vector<feature_point> vocabulary)
{
    dmatcher_->clear();
    dmatcher_->add(vocabulary);

    setDescriptorSize(vocabulary.size());

    dmatcher_->train();
}

void BoWModelDescriptorExtractor::setDescriptorSize(int descriptor_size)
{
    descriptor_size_ = descriptor_size;
}

void BoWModelDescriptorExtractor::setSearchIndexParams(std::string search_index_params)
{
    dmatcher_->setSearchIndexParams(search_index_params);
}

int BoWModelDescriptorExtractor::descriptorSize()
{
    return descriptor_size_;
}

void BoWModelDescriptorExtractor::loadMatcherIndex()
{
    int descriptor_size;
    dmatcher_->loadIndex(descriptor_size);

    setDescriptorSize(descriptor_size);

    printf("[BoWModelDescriptorExtractor::loadMatcherIndex] vocabulary size: %d\n", descriptor_size);
}

BoWModelDescriptorExtractor::~BoWModelDescriptorExtractor()
{
    descriptor_size_ = 0;
    dmatcher_.reset();
}

void BoWModelDescriptorExtractor::compute(std::vector<feature_point> model_descriptors, bow_vector &bow_model_descriptor)
{
    bow_model_descriptor.clear();
    bow_model_descriptor.resize(descriptor_size_, 0);

    std::vector<DMatch> matches;
    dmatcher_->match(model_descriptors, matches);

    for (auto &match : matches)
    {
        int train_idx = match.train_idx;

        bow_model_descriptor[train_idx] = bow_model_descriptor[train_idx] + 1.f;
    }

    // TF (term frequency) metric
    float descriptor_size = static_cast<float>(model_descriptors.size());
    for (auto &value : bow_model_descriptor)
    {
        value = value / descriptor_size;
    }
}
