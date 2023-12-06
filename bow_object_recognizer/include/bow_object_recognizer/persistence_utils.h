#ifndef PERSISTENCE_UTILS_H
#define PERSISTENCE_UTILS_H

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

namespace PersistenceUtils
{
    inline bool writeVectorToFile(std::string& file, std::vector<float>& vect)
    {
        std::cout << "[writeVectorToFile] Write vector to file " << file << "\n";

        std::ofstream out(file.c_str());
        if(!out)
        {
            std::cout << "Cannot open file.\n";
            return false;
        }

        for(auto & el : vect)
        {
            out << el << " ";
        }

        out.close();
        return true;
    }

    inline bool readVectorFromFile(std::string& file, std::vector<float>& vect)
    {
        std::ifstream in;
        in.open(file.c_str(), std::ifstream::in);

        if(!in)
        {
            std::cout << "Cannot open file.\n";
            return false;
        }

        char linebuf[1024];
        in.getline(linebuf, 1024);
        std::string line (linebuf);
        std::vector<std::string> strs;
        boost::split(strs, line, boost::is_any_of(" "));

        for(auto & str : strs)
        {
            float val = static_cast<float> (atof(str.c_str()));
            vect.push_back(val);
        }

        return true;
    }

    inline bool modelPresents (std::string file, std::string model)
    {
        std::ifstream in(file.c_str());

        if(in.is_open())
        {
            while(in.good())
            {
                std::string line;
                getline(in, line);

                if(line.substr(0, model.size()) == model)
                {
                    in.close();
                    return true;
                }
            }
            in.close();
            return false;
        }
    }

    inline bool modelPresents (std::string path, std::string scene, std::string model)
    {
        std::stringstream ss;

        ss << scene << "_" << model;

        std::string search_str = ss.str();
        int str_length = static_cast<int>(search_str.size());

        boost::filesystem::path gt_files_path = path;

        for (boost::filesystem::directory_iterator it (gt_files_path);
         it != boost::filesystem::directory_iterator (); 
         ++it)
        {
            std::string file = (it->path().filename()).string();

            if (boost::filesystem::is_regular_file (it->status ()) &&
             file.substr(0,str_length) == search_str &&
              file.find("occlusion") == std::string::npos)
            {
                return true;
            }
        }

        return false;
    }
}

#endif // PERSISTENCE_UTILS_H
