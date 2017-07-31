#ifndef TEST_RUNNER_H
#define TEST_RUNNER_H

#include <string>
#include <vector>
#include <iostream>
#include <fstream>

typedef pcl::PointXYZRGB PointInT;
typedef pcl::PointCloud<PointInT>::Ptr PointInTPtr;
typedef pcl::Normal NormalT;
typedef pcl::PointCloud<NormalT>::Ptr NormalTPtr;

typedef std::vector<float> feature_point;
typedef std::vector<float> bow_vector;
typedef std::pair<std::string, float> bow_match;


class TestRunner
{
public:
    typedef boost::shared_ptr<TestRunner> Ptr;
    TestRunner(const std::string& tests_base_path, std::string test_setup_name, bool single_case_mode);

    // Iterate through all the combinations of the values of the test parameters
    void initTests();

    // Run an experiment with the fixed values of test parameters
    // Iterate over all the test scenes and run detector on every single scene
    void runTestCase(const int test_num);

    void setScoreThreshold(const float score);

    void iterateTestScenes();

    // Run the detector on given scene and calculate the evaluation metrics,
    // write the results to the output file
    void runDetector();

private:
    std::string tests_base_path_;
    std::string test_setup_name_;
    std::string test_case_dir_;
    std::string test_output_file_;
    std::string time_result_file_;
    std::string test_scores_file_;
    float start_thresh_;
    float end_thresh_;
    std::ofstream output_;
    bool single_case_mode_;

    std::vector<double> calc_durations;
};

#endif // TEST_RUNNER_H
