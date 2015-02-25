/*
 * (C) Copyright 2014 Autonomy Lab (Simon Fraser University).
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Contributors:
 *     Mani Monajjemi <mmonajje@sfu.ca>
 */
#include <string>

#include "ros/ros.h"
#include "image_transport/image_transport.h"
#include "cv_bridge/cv_bridge.h"
#include "sensor_msgs/image_encodings.h"
#include "sensor_msgs/RegionOfInterest.h"

#include "obzerver/periodicity_app.hpp"
#include "obzerver_ros/Tracks.h"

class ObzerverROS
{
public:
  template<class T>
  static void GetParam(const ros::NodeHandle& nh,
                       const std::string& param_name, T& var, const T& default_value)
  {
    nh.param<T>(param_name, var, default_value);
    ROS_INFO_STREAM("[OBR] Param " << param_name << " : " << var);
  }

  void CopyTracksToMsg(const std::vector<obz::Track>& tracks,
                       obzerver_ros::Tracks& target_tracks_msg)
  {
    target_tracks_msg.tracks.resize(tracks.size());
    for (std::size_t i = 0; i < tracks.size(); i++)
    {
      single_track_msg_.uid = tracks[i].uid;
      single_track_msg_.roi.x_offset = tracks[i].GetBB().x;
      single_track_msg_.roi.y_offset = tracks[i].GetBB().y;
      single_track_msg_.roi.width = tracks[i].GetBB().width;
      single_track_msg_.roi.height = tracks[i].GetBB().height;
      single_track_msg_.status = obzerver_ros::Track::STATUS_TRACKING;
      single_track_msg_.dominant_freq = tracks[i].dom_freq;
      single_track_msg_.spectrum = tracks[i].avg_spectrum;
      single_track_msg_.displacement = tracks[i].displacement;
      target_tracks_msg.tracks[i] = single_track_msg_;
    }
  }

protected:
  ros::NodeHandle ros_nh_;
  ros::NodeHandle private_nh_;
  image_transport::ImageTransport ros_it_;
  image_transport::Subscriber sub_image_;

  image_transport::Publisher pub_debug_image_;
  image_transport::Publisher pub_stablized_image_;
  image_transport::Publisher pub_diff_image_;
  image_transport::Publisher pub_simmat_image_;
  ros::Publisher pub_all_tracks_;
  ros::Publisher pub_per_tracks_;

  std::string param_cfg_file_;
  std::string param_log_file_;
  double param_downsample_factor_;

  bool enable_debug_image_;
  bool enable_stablized_image_;
  bool enable_diff_image_;
  bool enable_simmat_image_;

  cv::Mat frame_input_;

  unsigned int last_seq_;
  sensor_msgs::ImageConstPtr cache_msg_;
  cv_bridge::CvImageConstPtr frame_input_cvptr_;
  cv_bridge::CvImage frame_debug_cvi_;
  cv_bridge::CvImage frame_sim_cvi_;
  cv_bridge::CvImage frame_stab_cvi_;
  cv_bridge::CvImage frame_diff_cvi_;

  obzerver_ros::Track  single_track_msg_;
  obzerver_ros::Tracks all_tracks_msg_;
  obzerver_ros::Tracks per_tracks_msg_;

  bool do_downsample_;
  std::size_t frame_counter_;

  obz::app::PeriodicityApp papp;
  std::vector<obz::Track> tracks;

  void UpdateParams()
  {
    GetParam<bool>(private_nh_, "enable_debug_image", enable_debug_image_, false);
    GetParam<bool>(private_nh_, "enable_stablized_image", enable_stablized_image_, false);
    GetParam<bool>(private_nh_, "enable_diff_image", enable_diff_image_, false);
    GetParam<bool>(private_nh_, "enable_simmat_image", enable_simmat_image_, false);

    GetParam<std::string>(private_nh_, "obz_configfile", param_cfg_file_, std::string(""));
    GetParam<std::string>(private_nh_, "obz_logfile", param_log_file_, std::string(""));
    GetParam<double>(private_nh_, "downsample_factor", param_downsample_factor_, 1.0);
  }

  void Process()
  {
    // Do nothing when there is no new frame
    if (!cache_msg_ || last_seq_ == cache_msg_->header.seq) return;

    StepBenchmarker::GetInstance().reset();
    last_seq_ = cache_msg_->header.seq;

    try
    {
      if (sensor_msgs::image_encodings::isColor(cache_msg_->encoding))
      {
        ROS_WARN_ONCE("[OBR] Input image is BGR8");
        frame_input_cvptr_ = cv_bridge::toCvShare(
              cache_msg_, sensor_msgs::image_encodings::BGR8);
      }
      else
      {
        ROS_WARN_ONCE("[OBR] Input image is MONO8");
        frame_input_cvptr_ = cv_bridge::toCvShare(cache_msg_, sensor_msgs::image_encodings::MONO8);
      }

      frame_input_ = frame_input_cvptr_->image;  // No Copy For Now
      TICK("ML_Frame_Copy");
      if (do_downsample_)
      {
        cv::resize(frame_input_, frame_input_, cv::Size(0, 0),
                   param_downsample_factor_, param_downsample_factor_, cv::INTER_CUBIC);
        TICK("ML_Downsampling");
      }

      LOG(INFO) << "Frame: " << frame_counter_ << " [" << frame_input_.cols << " x " << frame_input_.rows << "]";

      papp.Update(frame_input_);

      all_tracks_msg_.header.stamp = ros::Time::now();
      all_tracks_msg_.header.frame_id = frame_input_cvptr_->header.frame_id;
      all_tracks_msg_.max_width = frame_input_cvptr_->image.cols;
      all_tracks_msg_.max_height = frame_input_cvptr_->image.rows;
      per_tracks_msg_.header.stamp = ros::Time::now();
      per_tracks_msg_.header.frame_id = frame_input_cvptr_->header.frame_id;
      per_tracks_msg_.max_width = frame_input_cvptr_->image.cols;
      per_tracks_msg_.max_height = frame_input_cvptr_->image.rows;

      // tracks will be cleared
      papp.GetMOTPtr()->CopyAllTracks(tracks);
      CopyTracksToMsg(tracks, all_tracks_msg_);
      ROS_ASSERT(tracks.size() == all_tracks_msg_.tracks.size());

      // tracks will be cleared
      papp.GetPeriodicTracks(tracks);
      CopyTracksToMsg(tracks, per_tracks_msg_);
      ROS_ASSERT(tracks.size() == per_tracks_msg_.tracks.size());

      ROS_DEBUG_STREAM_THROTTLE(1, "All tracks: " << all_tracks_msg_.tracks.size()
                               << " Periodic Tracks: " << per_tracks_msg_.tracks.size());

      // Publish
      pub_all_tracks_.publish(all_tracks_msg_);
      pub_per_tracks_.publish(per_tracks_msg_);

      if (
          enable_stablized_image_ &&
          pub_stablized_image_.getNumSubscribers() > 0 &&
          papp.GetCTCstPtr()->GetStablizedRGB().data
          )
      {
        frame_stab_cvi_.image = papp.GetCTCstPtr()->GetStablizedRGB();
        frame_stab_cvi_.header.frame_id = frame_input_cvptr_->header.frame_id;
        frame_stab_cvi_.header.stamp = frame_input_cvptr_->header.stamp;
        frame_stab_cvi_.encoding = "bgr8";
        papp.GetMOTPtr()->DrawTracks(frame_stab_cvi_.image);
        pub_stablized_image_.publish(frame_stab_cvi_.toImageMsg());
      }

      if (
          enable_diff_image_ &&
          pub_diff_image_.getNumSubscribers() > 0 &&
          papp.GetCTCstPtr()->GetLatestDiff().data
          )
      {
        frame_diff_cvi_.image = papp.GetCTCstPtr()->GetLatestDiff();
        frame_diff_cvi_.header.frame_id = frame_input_cvptr_->header.frame_id;
        frame_diff_cvi_.header.stamp = frame_input_cvptr_->header.stamp;
        frame_diff_cvi_.encoding = "mono8";
        pub_diff_image_.publish(frame_diff_cvi_.toImageMsg());
      }

      frame_counter_++;

      TICK("Visualization");
      LOG(INFO) << StepBenchmarker::GetInstance().getstr();
    }
    catch (const cv_bridge::Exception& e)
    {
      ROS_ERROR("[OBR] cv_bridge exception: %s", e.what());
    }
  }

  void ImageCallback(const sensor_msgs::ImageConstPtr& msg)
  {
    cache_msg_ = msg;
  }

public:
  ObzerverROS(const int queue_size, ros::NodeHandle& ros_nh):
    ros_nh_(ros_nh),
    private_nh_("~"),
    ros_it_(ros_nh),
    last_seq_(-1),
    frame_counter_(0)
  {
    UpdateParams();

    boost::program_options::variables_map vm;
    if (!papp.Init(param_cfg_file_, param_log_file_, false, std::string(""), vm))
    {
      throw std::runtime_error("Error initializing obz::PeriodicityApp");
    }

    sub_image_ = ros_it_.subscribe(
          "camera/image_raw",
          queue_size,
          &ObzerverROS::ImageCallback,
          this);

    pub_all_tracks_ = ros_nh_.advertise<obzerver_ros::Tracks>("obzerver/tracks/all", 30);
    pub_per_tracks_ = ros_nh_.advertise<obzerver_ros::Tracks>("obzerver/tracks/periodic", 30);

    if (enable_debug_image_)
    {
      ROS_INFO("[OBR] debug_image is enabled.");
      pub_debug_image_ = ros_it_.advertise("obzerver/debug_image", 1);
    }

    if (enable_diff_image_)
    {
      ROS_INFO("[OBR] diff_image is enabled.");
      pub_diff_image_ = ros_it_.advertise("obzerver/diff_image", 1);
    }

    if (enable_stablized_image_)
    {
      ROS_INFO("[OBR] stablized_image is enabled.");
      pub_stablized_image_ = ros_it_.advertise("obzerver/stablized_image", 1);
    }

    if (enable_simmat_image_)
    {
      ROS_INFO("[OBR] simmat_image is enabled.");
      pub_simmat_image_ = ros_it_.advertise("obzerver/simmat_image", 1);
    }

    do_downsample_ = param_downsample_factor_ < 1.0 && param_downsample_factor_ > 0.0;
  }

  virtual void spin()
  {
    // TODO(mani-monaj): Check if fixed freq. is better
    //    ROS_INFO("[OBR] Setting ROS Loop Rate to %f hz", param_fps_);
    ros::Rate rate(30.0);
    while (ros::ok())
    {
      spinOnce();
      if (!rate.sleep())
      {
        ROS_WARN("[OBR] Can not catch up with input image rate.");
      }
    }
  }

  virtual void spinOnce()
  {
    Process();
    ros::spinOnce();
  }
};

int main(int argc, char* argv[])
{
  /* ROS Stuff */
  ros::init(argc, argv, "obzerver_ros");
  ros::NodeHandle ros_nh;
  int param_queue_size;

  ObzerverROS::GetParam<int>(ros_nh, "queue_size", param_queue_size, 1);


  try
  {
    ObzerverROS obzerver_ros(param_queue_size, ros_nh);
    obzerver_ros.spin();
  }
  catch (std::runtime_error& ex)
  {
    ROS_FATAL_STREAM("[OBZ] Runtime error: " << ex.what());
    return 1;
  }
  catch (cv::Exception& ex)
  {
    ROS_FATAL_STREAM("[OBZ] OpenCV Exception: " << ex.what());
    return 2;
  }


  return 0;
}
