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

#include "glog/logging.h"

#include "opencv2/features2d/features2d.hpp"

#include "obzerver/logger.hpp"
#include "obzerver/utility.hpp"
#include "obzerver/benchmarker.hpp"
#include "obzerver/camera_tracker.hpp"
#include "obzerver/object_tracker.hpp"

#include "obzerver_ros/object.h"

class ObzerverROS
{
public:
  template<class T>
  static void GetParam(const ros::NodeHandle& nh, const std::string& param_name, T& var, const T& default_value)
  {
    nh.param<T>(param_name, var, default_value);
    ROS_INFO_STREAM("[OBR] Param " << param_name << " : " << var);
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
  ros::Publisher pub_object_;

  std::string param_log_file_;

  double param_downsample_factor_;
  double param_fps_;

  int param_max_features_;
  int param_num_particles_;
  int param_hist_len_;
  int param_pylk_winsize_;
  int param_pylk_iters_;
  double param_pylk_eps_;
  int param_ffd_threshold_;
  int param_skip_dc_term_count_;

  bool enable_debug_image_;
  bool enable_stablized_image_;
  bool enable_diff_image_;
  bool enable_simmat_image_;

  cv::Mat frame_input_;
  cv::Mat frame_gray_;
  unsigned int last_seq_;
  sensor_msgs::ImageConstPtr cache_msg_;
  cv_bridge::CvImageConstPtr frame_input_cvptr_;
  cv_bridge::CvImage frame_debug_cvi_;
  cv_bridge::CvImage frame_sim_cvi_;
  cv_bridge::CvImage frame_stab_cvi_;
  cv_bridge::CvImage frame_diff_cvi_;
  obzerver_ros::object out_object_;

  bool do_convert_;
  bool do_downsample_;
  cv::Ptr<cv::FeatureDetector> feature_detector_;
  cv::Ptr<CameraTracker> camera_tracker_;
  cv::Ptr<ObjectTracker> object_tracker_;
  std::size_t frame_counter_;

  StepBenchmarker& ticker_;

  void UpdateParams()
  {
    GetParam<bool>(private_nh_, "enable_debug_image", enable_debug_image_, false);
    GetParam<bool>(private_nh_, "enable_stablized_image", enable_stablized_image_, false);
    GetParam<bool>(private_nh_, "enable_diff_image", enable_diff_image_, false);
    GetParam<bool>(private_nh_, "enable_simmat_image", enable_simmat_image_, false);

    GetParam<std::string>(private_nh_, "obz_logfile", param_log_file_, std::string(""));
    GetParam<double>(private_nh_, "downsample_factor", param_downsample_factor_, 1.0);
    GetParam<double>(private_nh_, "fps", param_fps_, 30.0);
    GetParam<int>(private_nh_, "max_features", param_max_features_, 300);
    GetParam<int>(private_nh_, "num_particles", param_num_particles_, 1000);
    GetParam<int>(private_nh_, "history_length", param_hist_len_, 90);
    GetParam<int>(private_nh_, "pylk_winsize", param_pylk_winsize_, 30);
    GetParam<int>(private_nh_, "pylk_iterations", param_pylk_iters_, 30);
    GetParam<double>(private_nh_, "pylk_eps", param_pylk_eps_, 0.01);
    GetParam<int>(private_nh_, "ffd_threshold", param_ffd_threshold_, 30);
    GetParam<int>(private_nh_, "skip_dc_term_count", param_skip_dc_term_count_, 1);
  }

  void Process()
  {
    ticker_.reset();

    // Do nothing when there is no new frame
    if (!cache_msg_ || last_seq_ == cache_msg_->header.seq) return;
    last_seq_ = cache_msg_->header.seq;

    float _f = -1.0;
    try
    {
      if (sensor_msgs::image_encodings::isColor(cache_msg_->encoding))
      {
        ROS_WARN_ONCE("[OBR] Input image is BGR8");
        frame_input_cvptr_ = cv_bridge::toCvShare(cache_msg_, sensor_msgs::image_encodings::BGR8);
        do_convert_ = true;
      }
      else
      {
        ROS_WARN_ONCE("[OBR] Input image is MONO8");
        frame_input_cvptr_ = cv_bridge::toCvShare(cache_msg_, sensor_msgs::image_encodings::MONO8);
        do_convert_ = false;
      }

      frame_input_ = frame_input_cvptr_->image;  // No Copy For Now
      ticker_.tick("Frame Copy");
      if (do_downsample_)
      {
        cv::resize(frame_input_, frame_input_, cv::Size(0, 0),
                   param_downsample_factor_, param_downsample_factor_, cv::INTER_CUBIC);
        ticker_.tick("Downsampling.");
      }

      frame_gray_ = frame_input_;  // No Copy for now
      if (do_convert_)
      {
        cv::cvtColor(frame_input_, frame_gray_, cv::COLOR_BGR2GRAY);
        ticker_.tick("Frame 2 Gray");
      }

      LOG(INFO) << "Frame: " << frame_counter_ << " [" << frame_input_.cols << " x " << frame_input_.rows << "]";

      const bool ct_success = camera_tracker_->Update(frame_gray_, frame_input_);

      out_object_.header.stamp = ros::Time::now();
      out_object_.header.frame_id = frame_input_cvptr_->header.frame_id;
      out_object_.status = 0;
      if (!ct_success)
      {
        LOG(WARNING) << "Camera Tracker Failed";
        // TODO(mani-monaj)
      }
      else
      {
        object_tracker_->Update(camera_tracker_->GetStablizedGray(),
                               camera_tracker_->GetLatestDiff(),
                               camera_tracker_->GetLatestSOF(),
                               camera_tracker_->GetLatestCameraTransform());

        out_object_.status = object_tracker_->GetStatus();

        LOG(INFO) << "Tracking status: " << object_tracker_->GetStatus();
        if (object_tracker_->IsTracking())
        {
           // TODO(mani-monaj)
          _f = object_tracker_->GetObject().GetPeriodicity().GetDominantFrequency(param_skip_dc_term_count_);
          LOG(INFO) << "Object: "
                    << object_tracker_->GetObjectBoundingBox()
                    << " Periodicity:"
                    << _f;
          out_object_.roi.x_offset = object_tracker_->GetObjectBoundingBox().tl().x;
          out_object_.roi.y_offset = object_tracker_->GetObjectBoundingBox().tl().y;
          out_object_.roi.width = object_tracker_->GetObjectBoundingBox().width;
          out_object_.roi.height = object_tracker_->GetObjectBoundingBox().height;
          out_object_.max_width = frame_input_.cols;
          out_object_.max_height = frame_input_.rows;
          out_object_.spectrum = object_tracker_->GetObject().GetPeriodicity().GetSpectrum();
          out_object_.dominant_freq = _f;
          out_object_.displacement = 0.0;  // TODO(mani-monaj)
        }

        // Publish

        if (pub_object_.getNumSubscribers() > 0)
        {
          pub_object_.publish(out_object_);
        }

        if (
          enable_stablized_image_ &&
          pub_stablized_image_.getNumSubscribers() > 0 &&
          camera_tracker_->GetStablizedGray().data
        )
        {
          frame_stab_cvi_.image = camera_tracker_->GetStablizedGray();
          frame_stab_cvi_.header.frame_id = frame_input_cvptr_->header.frame_id;
          frame_stab_cvi_.header.stamp = frame_input_cvptr_->header.stamp;
          frame_stab_cvi_.encoding = "mono8";
          object_tracker_->DrawParticles(frame_stab_cvi_.image);
          pub_stablized_image_.publish(frame_stab_cvi_.toImageMsg());
        }

        if (
          enable_debug_image_ &&
          pub_debug_image_.getNumSubscribers() > 0 &&
          frame_input_.data
        )
        {
          frame_debug_cvi_.image = frame_input_;
          frame_debug_cvi_.header.frame_id = frame_input_cvptr_->header.frame_id;
          frame_debug_cvi_.header.stamp = frame_input_cvptr_->header.stamp;
          frame_debug_cvi_.encoding = do_convert_ ? "bgr8" : "mono8";

          drawFeaturePointsTrajectory(frame_debug_cvi_.image,
                                      camera_tracker_->GetHomographyOutliers(),
                                      camera_tracker_->GetTrackedFeaturesPrev(),
                                      camera_tracker_->GetTrackedFeaturesCurr(),
                                      2,
                                      cv::Scalar(127, 127, 127),
                                      cv::Scalar(0, 0, 255),
                                      cv::Scalar(0, 0, 255));

          std::stringstream ss;
          ss << std::setprecision(5) << "Periodicity: " << _f;
          cv::putText(frame_debug_cvi_.image, ss.str(), cv::Point(40, 40), 1,
                      CV_FONT_HERSHEY_PLAIN, cv::Scalar(0, 0, 255));

          pub_debug_image_.publish(frame_debug_cvi_.toImageMsg());
        }

        if (
          enable_diff_image_ &&
          pub_diff_image_.getNumSubscribers() > 0 &&
          camera_tracker_->GetLatestDiff().data
        )
        {
          frame_diff_cvi_.image = camera_tracker_->GetLatestDiff();
          frame_diff_cvi_.header.frame_id = frame_input_cvptr_->header.frame_id;
          frame_diff_cvi_.header.stamp = frame_input_cvptr_->header.stamp;
          frame_diff_cvi_.encoding = "mono8";
          pub_diff_image_.publish(frame_diff_cvi_.toImageMsg());
        }

        if (
          enable_simmat_image_ &&
          pub_simmat_image_.getNumSubscribers() > 0 &&
          object_tracker_->IsTracking()
        )
        {
          frame_sim_cvi_.image = object_tracker_->GetObject().GetSelfSimilarity().GetSimMatrixRendered();
          frame_sim_cvi_.header.frame_id = frame_input_cvptr_->header.frame_id;
          frame_sim_cvi_.header.stamp = frame_input_cvptr_->header.stamp;
          frame_sim_cvi_.encoding = "mono8";
          pub_simmat_image_.publish(frame_sim_cvi_.toImageMsg());
        }

        ticker_.tick("Visualization");
        frame_counter_++;
        LOG(INFO) << ticker_.getstr();
      }
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
    frame_counter_(0),
    ticker_(StepBenchmarker::GetInstance())
  {
    UpdateParams();

    sub_image_ = ros_it_.subscribe(
                  "camera/image_raw",
                  queue_size,
                  &ObzerverROS::ImageCallback,
                  this);
    pub_object_ = ros_nh_.advertise<obzerver_ros::object>("obzerver/object", 20);

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

    obz_log_config("obzerver_ros_node", param_log_file_);

    feature_detector_ = new cv::FastFeatureDetector(param_ffd_threshold_, true);
    camera_tracker_ = new CameraTracker(param_hist_len_, feature_detector_, param_max_features_,
                                       param_pylk_winsize_, param_pylk_iters_, param_pylk_eps_);
    object_tracker_ = new ObjectTracker(param_num_particles_, param_hist_len_, param_fps_);

    do_downsample_ = param_downsample_factor_ < 1.0 && param_downsample_factor_ > 0.0;
  }

  virtual void spin()
  {
    // TODO(mani-monaj): Check if fixed freq. is better
    ROS_INFO("[OBR] Setting ROS Loop Rate to %f hz", param_fps_);
    ros::Rate rate(param_fps_);
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
  ObzerverROS obzerver_ros(param_queue_size, ros_nh);

  /* Main Loop */
  ROS_INFO("[OBR] Starting obzerver_ros");
  obzerver_ros.spin();
  return 0;
}
