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
#include "std_msgs/Bool.h"

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
  bool paused;

  ros::NodeHandle ros_nh_;
  ros::NodeHandle private_nh_;
  image_transport::ImageTransport ros_it_;
  image_transport::Subscriber sub_image_;
  ros::Subscriber sub_enable_;

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

  std::shared_ptr<obz::app::PeriodicityApp> papp_ptr_;
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

    GetParam<bool>(private_nh_, "start_paused", paused, false);
  }

  void Process()
  {
    // Do nothing when there is no new frame
    if (paused || !cache_msg_ || last_seq_ == cache_msg_->header.seq) return;
    if (!papp_ptr_) throw std::runtime_error("Obzerver app is NULL");

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

      papp_ptr_->Update(frame_input_);

      all_tracks_msg_.header.stamp = ros::Time::now();
      all_tracks_msg_.header.frame_id = frame_input_cvptr_->header.frame_id;
      all_tracks_msg_.max_width = frame_input_cvptr_->image.cols;
      all_tracks_msg_.max_height = frame_input_cvptr_->image.rows;
      per_tracks_msg_.header.stamp = ros::Time::now();
      per_tracks_msg_.header.frame_id = frame_input_cvptr_->header.frame_id;
      per_tracks_msg_.max_width = frame_input_cvptr_->image.cols;
      per_tracks_msg_.max_height = frame_input_cvptr_->image.rows;

      // tracks will be cleared
      papp_ptr_->GetMOTPtr()->CopyAllTracks(tracks);
      CopyTracksToMsg(tracks, all_tracks_msg_);
      ROS_ASSERT(tracks.size() == all_tracks_msg_.tracks.size());

      // tracks will be cleared
      papp_ptr_->GetPeriodicTracks(tracks);
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
          papp_ptr_->GetCTCstPtr()->GetStablizedRGB().data
          )
      {
        frame_stab_cvi_.image = papp_ptr_->GetCTCstPtr()->GetStablizedRGB();
        frame_stab_cvi_.header.frame_id = frame_input_cvptr_->header.frame_id;
        frame_stab_cvi_.header.stamp = frame_input_cvptr_->header.stamp;
        frame_stab_cvi_.encoding = "bgr8";
        papp_ptr_->GetMOTPtr()->DrawTracks(frame_stab_cvi_.image);
        pub_stablized_image_.publish(frame_stab_cvi_.toImageMsg());
      }

      if (
          enable_diff_image_ &&
          pub_diff_image_.getNumSubscribers() > 0 &&
          papp_ptr_->GetCTCstPtr()->GetLatestDiff().data
          )
      {
        frame_diff_cvi_.image = papp_ptr_->GetCTCstPtr()->GetLatestDiff();
        frame_diff_cvi_.header.frame_id = frame_input_cvptr_->header.frame_id;
        frame_diff_cvi_.header.stamp = frame_input_cvptr_->header.stamp;
        frame_diff_cvi_.encoding = "mono8";
        pub_diff_image_.publish(frame_diff_cvi_.toImageMsg());
      }

      if (
          enable_debug_image_ &&
          pub_debug_image_.getNumSubscribers() > 0 &&
          papp_ptr_->GetCTCstPtr()->GetStablizedRGB().data
          )
      {
        frame_debug_cvi_.image = papp_ptr_->GetCTCstPtr()->GetStablizedRGB().clone();
        if (papp_ptr_->GetCTCstPtr()->GetTrackedFeaturesCurr().size()) {
          obz::util::DrawFeaturePointsTrajectory(frame_debug_cvi_.image,
                                                 papp_ptr_->GetCTCstPtr()->GetHomographyOutliers(),
                                                 papp_ptr_->GetCTCstPtr()->GetTrackedFeaturesPrev(),
                                                 papp_ptr_->GetCTCstPtr()->GetTrackedFeaturesCurr(),
                                                 2,
                                                 CV_RGB(0,0,255), CV_RGB(255, 0, 0), CV_RGB(255, 0, 0));
          papp_ptr_->GetRECstPtr()->DrawROIs(frame_debug_cvi_.image, true);
        }
        frame_debug_cvi_.header.frame_id = frame_input_cvptr_->header.frame_id;
        frame_debug_cvi_.header.stamp = frame_input_cvptr_->header.stamp;
        frame_debug_cvi_.encoding = "bgr8";
        pub_debug_image_.publish(frame_debug_cvi_.toImageMsg());
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
    Process();
  }

  void EnableCallback(const std_msgs::BoolConstPtr& msg)
  {
    const bool en = static_cast<bool>(msg->data);
    ROS_WARN_STREAM("[OBR] Request to " << (en ? "Enable" : "Disable"));

    if (paused && en)
    {
      ReinitObzerverApp();
    }

    if (!paused && !en)
    {
       // TODO: Cleanup??
    }

    paused = !en;
  }

  void ReinitObzerverApp()
  {
    ROS_WARN("[OBR] Reinitializing obzerver ...");
    boost::program_options::variables_map vm;
    papp_ptr_ = std::make_shared<obz::app::PeriodicityApp>();
    if (!papp_ptr_->Init(param_cfg_file_, param_log_file_, false, std::string(""), vm))
    {
      throw std::runtime_error("Error initializing obz::PeriodicityApp");
    }
  }

public:
  ObzerverROS(const int queue_size, ros::NodeHandle& ros_nh):
    paused(false),
    ros_nh_(ros_nh),
    private_nh_("~"),
    ros_it_(ros_nh),
    last_seq_(-1),
    frame_counter_(0),
    papp_ptr_()
  {
    UpdateParams();

    if (!paused)
    {
      ReinitObzerverApp();
    }
    else
    {
      ROS_ERROR("[OBR] Starting in pause state");
    }

    sub_image_ = ros_it_.subscribe(
          "image_raw",
          queue_size,
          &ObzerverROS::ImageCallback,
          this);

    sub_enable_ = ros_nh_.subscribe("obzerver/enable", 10, &ObzerverROS::EnableCallback, this);

    pub_all_tracks_ = ros_nh_.advertise<obzerver_ros::Tracks>("obzerver/tracks/all", 30);
    pub_per_tracks_ = ros_nh_.advertise<obzerver_ros::Tracks>("obzerver/tracks/periodic", 30);

    if (enable_debug_image_)
    {
      ROS_INFO("[OBR] debug_image is enabled.");
      pub_debug_image_ = ros_it_.advertise("obzerver/debug/image_raw", 1);
    }

    if (enable_diff_image_)
    {
      ROS_INFO("[OBR] diff_image is enabled.");
      pub_diff_image_ = ros_it_.advertise("obzerver/diff/image_raw", 1);
    }

    if (enable_stablized_image_)
    {
      ROS_INFO("[OBR] stablized_image is enabled.");
      pub_stablized_image_ = ros_it_.advertise("obzerver/stablized/image_raw", 1);
    }

    if (enable_simmat_image_)
    {
      ROS_INFO("[OBR] simmat_image is enabled.");
      pub_simmat_image_ = ros_it_.advertise("obzerver/simmat/image_raw", 1);
    }

    do_downsample_ = param_downsample_factor_ < 1.0 && param_downsample_factor_ > 0.0;
  }

  virtual void spin()
  {
    // TODO(mani-monaj): Check if fixed freq. is better
    //    ROS_INFO("[OBR] Setting ROS Loop Rate to %f hz", param_fps_);
//    ros::Rate rate(20.0);
//    while (ros::ok())
//    {
//      spinOnce();
//      if (!rate.sleep())
//      {
//        ROS_WARN("[OBR] Can not catch up with input image rate.");
//      }
//    }
    ros::spin();
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
  ros::NodeHandle ros_nh_priv("~");
  int param_queue_size;

  ObzerverROS::GetParam<int>(ros_nh_priv, "queue_size", param_queue_size, 1);

  try
  {
    ObzerverROS obzerver_ros(param_queue_size, ros_nh);
    obzerver_ros.spin();
  }
  catch (const ros::Exception& ex)
  {
    ROS_ERROR_STREAM("[OBZ] ROS Exception: " << ex.what());
  }
  catch (const std::runtime_error& ex)
  {
    ROS_ERROR_STREAM("[OBZ] Runtime error: " << ex.what());
  }
  catch (const cv::Exception& ex)
  {
    ROS_ERROR_STREAM("[OBZ] OpenCV Exception: " << ex.what());
  }

  return 0;
}
