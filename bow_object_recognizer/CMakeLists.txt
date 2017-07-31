cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

#set(CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS}") #"-g")

# Add -std=c++0x ${CMAKE_CXX_FLAGS}

set(SUBSUBSYS_NAME bow_object_recognizer)

# we need FindFLANN.cmake
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

project(bow_object_recognizer)

find_package(PCL 1.2 REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package(HDF5)
if(HDF5_FOUND)
find_package(FLANN)
  include_directories(${FLANN_INCLUDE_DIRS})

  include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

  # add executables and link libraries
  add_executable(build_vocabulary src/build_vocabulary.cpp)
  add_executable(bow_train src/bow_train.cpp)
  add_executable(bow_recognizer src/bow_recognizer.cpp)

  target_link_libraries(build_vocabulary libvtkCommon.so libvtkFiltering.so ${PCL_LIBRARIES} ${OpenCV_LIBS} ${HDF5_hdf5_LIBRARY})
  target_link_libraries(bow_train ${PCL_LIBRARIES} ${FLANN_LIBRARIES} ${OpenCV_LIBS} ${HDF5_hdf5_LIBRARY})
  target_link_libraries(bow_recognizer ${PCL_LIBRARIES} ${FLANN_LIBRARIES} ${OpenCV_LIBS} ${HDF5_hdf5_LIBRARY})
endif(HDF5_FOUND)
