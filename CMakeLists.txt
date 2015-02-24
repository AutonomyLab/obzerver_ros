cmake_minimum_required(VERSION 2.8.3)
project(obzerver_ros)

# CMAKE_PREFIX_PATH has higher priority than $ENV{CMAKE_PREFIX_PATH}
# catkin workspace's setup file sets the latter, so messing with
# the former will break catkin

set(CUSTOM_OPENCV_DIR "/local_home/opencv/opencv-2.4.10.1/install")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  image_transport
  sensor_msgs
  cv_bridge
  message_generation
  roslint
)

## System dependencies are found with CMake's conventions
find_package(Boost 1.45 REQUIRED COMPONENTS program_options)

find_package(OpenCV REQUIRED PATHS ${CUSTOM_OPENCV_DIR} NO_DEFAULT_PATH)

# TODO: Inherit obzerver libs and include_dirs
include(ExternalProject)
ExternalProject_Add(libobzerver
  #GIT_REPOSITORY git@bitbucket.org:AutonomyLab/obzerver.git
  GIT_REPOSITORY /home/mani/Dev/obzerver
  GIT_TAG master
  PREFIX ${CATKIN_DEVEL_PREFIX}
  CMAKE_ARGS
    -DWITH_TESTS=0
    -DCMAKE_PREFIX_PATH=${CUSTOM_OPENCV_DIR}
    -DCMAKE_BUILD_TYPE=Release
    -DEXTERNAL_DEVEL_SPACE=${CATKIN_DEVEL_PREFIX}
    -DCMAKE_INSTALL_PREFIX=${CATKIN_DEVEL_PREFIX}
  BUILD_COMMAND make
  INSTALL_COMMAND make install
#  BUILD_IN_SOURCE 1
)

## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

## Generate messages in the 'msg' folder
add_message_files(
 FILES
 Track.msg
 Tracks.msg
)

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES obzerver_ros
  CATKIN_DEPENDS message_runtime
#  DEPENDS system_lib
)

###########
## Build ##
###########

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIR}
  ${CATKIN_DEVEL_PREFIX}/include
)

link_directories(
  ${CATKIN_DEVEL_PREFIX}/lib
)

## Declare a cpp executable
add_executable(obzerver_node src/obzerver_node.cpp)

## Specify libraries to link a library or executable target against
add_dependencies(obzerver_node ${PROJECT_NAME}_generate_messages_cpp libobzerver)
target_link_libraries(obzerver_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${Boost_LIBRARIES}
  obzerver
  glog # TODO: This is an inherited dependency from libobzerver, both the folder and libname should be inherited
)

## roslint

roslint_cpp()

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
# install(TARGETS obzerver_ros obzerver_ros_node
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_obzerver_ros.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
