#ifndef SOURCE_H
#define SOURCE_H

#include <vector>
#include <boost/filesystem.hpp>

/**
  * Basic class for 3D point cloud model representation
  **/
class Model
{
public:
    std::string model_id;
    std::string cloud_path;
    std::vector<std::string> views;
};


/**
  * Abstract data source class, manages filesystem, load models and scenes for training etc.
  */
class Source
{
protected:
    typedef pcl::PointXYZRGB PointInT;
    typedef pcl::PointCloud<PointInT>::Ptr PointInTPtr;

    std::string training_path_;
    std::vector<Model> models_;
    std::vector<std::string> scenes_;

    int model_samples_number_;

    bool use_model_views_;

public:
    typedef boost::shared_ptr<Source> Ptr;

    Source();

    // Pass the path to the directory to store training data
    void setPath(std::string& path);

    void setUseModelViews(bool use_model_views);

    void printPath();

    std::string getModelDir(Model m) const;

    std::string getModelDir(std::string model_id) const;

    int getModelSamplesNumber();

    void getModelsInDir(boost::filesystem::path &models_path);

    void getScenesInDir(boost::filesystem::path &scenes_path);

    void loadModelFeatures(std::string model_id, std::string feature_type);

    void loadModelBoWVector(std::string model_id);

    std::vector<Model> getModels();

    std::vector<std::string> getScenes();
};

/*
 * Struct for storing matching scores
 **/
struct model_score
{
    std::string model_id;
    std::string view_id;
    float score;
    float sac_alignment_score;
};

struct sortModelScores
{
    bool operator() (const model_score& d1, const model_score& d2)
    {
        return d1.score > d2.score;
    }
};

struct sortModelSACAlignmentScores
{
    bool operator() (const model_score& d1, const model_score& d2)
    {
        return d1.sac_alignment_score < d2.sac_alignment_score;
    }
};
#endif // SOURCE_H
