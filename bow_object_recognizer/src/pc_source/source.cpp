#include <stdio.h>
#include <iostream>
#include <vector>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>
#include <bow_object_recognizer/source.h>

using namespace std;

Source::Source() {}

void Source::setPath(std::string &path)
{
    training_path_ = path;
}

void Source::setUseModelViews(bool use_model_views)
{
    use_model_views_ = use_model_views;

    if(use_model_views_)
        model_samples_number_ = 0;
}

void Source::printPath()
{
    printf("training_path: %s\n", training_path_.c_str());
}

int Source::getModelSamplesNumber()
{
    return model_samples_number_;
}

string Source::getModelDir(Model m) const
{
    std::stringstream dir;
    dir << training_path_ << "/models/" << m.model_id;
    return dir.str();
}

string Source::getModelDir(std::string model_id) const
{
    std::stringstream dir;
    dir << training_path_ << "/models/" << model_id;
    return dir.str();
}

void Source::getModelsInDir(boost::filesystem::path &models_path)
{
    cout << "\n\n[Source::getModelsInDir] Loading models from the directory " << models_path << "\n";

    if(!boost::filesystem::exists(models_path) && !boost::filesystem::is_directory(models_path))
        return;

    for (boost::filesystem::directory_iterator it (models_path); it != boost::filesystem::directory_iterator (); ++it)
    {
      if (boost::filesystem::is_directory (it->status ()) && (it->path().filename()).string() != "views")
      {
          boost::filesystem::path curr_path = it->path();
          getModelsInDir(curr_path);
      }

      string file = (it->path().filename()).string();

      if (boost::filesystem::is_regular_file (it->status ()) && boost::filesystem::extension (it->path ()) == ".pcd" && file.substr(0,8) == "3D_model")
      {
          vector<string> strs;
          boost::split(strs, models_path.string(), boost::is_any_of("/"));
          string model_id = strs[strs.size() - 1];

          Model model;
          model.model_id = model_id;
          model.cloud_path = (it->path()).string();

          printf("Model id: %s\n", model.model_id.c_str());

          if(use_model_views_)
          {
              std::stringstream path_ss;
              path_ss << models_path.string() << "/views";
              string views_dir = path_ss.str();

              for (boost::filesystem::directory_iterator view_it (views_dir); view_it != boost::filesystem::directory_iterator (); ++view_it)
              {                  
                  string view_file = (view_it->path().filename()).string();

                  if (boost::filesystem::is_regular_file (view_it->status ()) && boost::filesystem::extension (view_it->path ()) == ".pcd" && !strstr(view_file.c_str(), "keypoints") && !strstr(view_file.c_str(), "descr"))
                  {
                      boost::split(strs, view_file, boost::is_any_of("."));

                      string view_id = strs[0];
                      model.views.push_back(view_id);

                      model_samples_number_ = model_samples_number_ + 1;
                  }

              }

          }

          models_.push_back(model);
      }
    }

    if(!use_model_views_)
        model_samples_number_ = (int)models_.size();
}

void Source::getScenesInDir(boost::filesystem::path &scenes_path)
{
    cout << "\n[Source::getScenesInDir] Loading scenes from the directory " << scenes_path << "\n";

    if(!boost::filesystem::exists(scenes_path) && !boost::filesystem::is_directory(scenes_path))
        return;

    for (boost::filesystem::directory_iterator it (scenes_path); it != boost::filesystem::directory_iterator (); ++it)
    {
      if (boost::filesystem::is_regular_file (it->status ()) && boost::filesystem::extension (it->path ()) == ".pcd")
      {
          string cloud_path = (it->path()).string();
          scenes_.push_back(cloud_path);
      }
    }
}

vector<Model> Source::getModels()
{
    return models_;
}

vector<string> Source::getScenes()
{
    return scenes_;
}
