#include "ros/ros.h"
#include "image_transport/image_transport.h"
#include "cv_bridge/cv_bridge.h"
#include "sensor_msgs/image_encodings.h"

#include "glog/logging.h"

#include "opencv2/features2d/features2d.hpp"

#include "obzerver/logger.hpp"
#include "obzerver/utility.hpp"
#include "obzerver/benchmarker.hpp"
#include "obzerver/camera_tracker.hpp"
#include "obzerver/object_tracker.hpp"

class ObzerverROS {
public:
  template<class T>
  static void get_param(const std::string& param_name, T& var, const T default_value) {
    ros::param::param(param_name, var, default_value);
    ROS_INFO_STREAM("Param " << param_name << " : " << var);
  }

private:
  ros::NodeHandle ros_nh;
  image_transport::ImageTransport ros_it;
  image_transport::Subscriber sub_image;

  image_transport::Publisher pub_debug_image;
  image_transport::Publisher pub_stablized_image;
  image_transport::Publisher pub_diff_image;
  image_transport::Publisher pub_simmat_image;


  std::string param_log_file;

  float param_downsample_factor;
  float param_fps;

  int param_max_features;
  int param_num_particles;
  int param_hist_len;
  int param_pylk_winsize;
  int param_pylk_iters;
  double param_pylk_eps;
  int param_ffd_threshold;

  bool enable_debug_image;
  bool enable_stablized_image;
  bool enable_diff_image;
  bool enable_simmat_image;

  cv::Mat frame_input;
  cv::Mat frame_gray;
  bool do_convert;
  bool do_downsample;
  cv::Ptr<cv::FeatureDetector> feature_detector;
  cv::Ptr<CameraTracker> camera_tracker;
  cv::Ptr<ObjectTracker> object_tracker;
  std::size_t frame_counter;

  StepBenchmarker& ticker;

  void update_params()
  {
    get_param<bool>("enable_debug_image", enable_debug_image, false);
    get_param<bool>("enable_stablized_image", enable_stablized_image, false);
    get_param<bool>("enable_diff_image", enable_diff_image, false);
    get_param<bool>("enable_simmat_image", enable_simmat_image, false);

    get_param<std::string>("obz_logfile", param_log_file, std::string(""));
    get_param<float>("downsample_factor", param_downsample_factor, 1.0);
    get_param<float>("fps", param_fps, 30.0);
    get_param<int>("max_features", param_max_features, 300);
    get_param<int>("num_particles", param_num_particles, 1000);
    get_param<int>("history_length", param_hist_len, 90);
    get_param<int>("pylk_winsize", param_pylk_winsize, 30);
    get_param<int>("pylk_iterations", param_pylk_iters, 30);
    get_param<double>("pylk_eps", param_pylk_eps, 0.01);
    get_param<int>("ffd_threshold", param_ffd_threshold, 30);
  }

  void ImageCallback(const sensor_msgs::ImageConstPtr& msg) {
    ticker.reset();
    cv_bridge::CvImageConstPtr cv_ptr;
    try {
      if (sensor_msgs::image_encodings::isColor(msg->encoding)) {
        ROS_WARN_ONCE("Input image is BGR8");
        cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
        do_convert = true;
      } else {
        ROS_WARN_ONCE("Input image is MONO8");
        cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::MONO8);
        do_convert = false;
      }

      frame_input = cv_ptr->image; // No Copy For Now
      ticker.tick("Frame Copy");
      if (do_downsample) {
        cv::resize(frame_input, frame_input, cv::Size(0, 0), param_downsample_factor, param_downsample_factor, cv::INTER_CUBIC);
        ticker.tick("Downsampling.");
      }

      frame_gray = frame_input; // No Copy for now
      if (do_convert) {
        cv::cvtColor(frame_input, frame_gray, cv::COLOR_BGR2GRAY);
        ticker.tick("Frame 2 Gray");
      }
      LOG(INFO) << "Frame: " << frame_counter << " [" << frame_input.cols << " x " << frame_input.rows << "]";

      bool ct_success = camera_tracker->Update(frame_gray, frame_input);

      if (!ct_success) {
        LOG(WARNING) << "Camera Tracker Failed";
        // TODO
      } else {
        object_tracker->Update(camera_tracker->GetStablizedGray(),
                              camera_tracker->GetLatestDiff(),
                              camera_tracker->GetLatestSOF(),
                              camera_tracker->GetLatestCameraTransform());


        LOG(INFO) << "Tracking status: " << object_tracker->GetStatus();
        if (object_tracker->IsTracking()) {
          float _f = object_tracker->GetObject().GetPeriodicity().GetDominantFrequency(1); // TODO
          LOG(INFO) << "Object: "
                    << object_tracker->GetObjectBoundingBox()
                    << " Periodicity:"
                    << _f;
          //LOG(INFO) << "Spectrum: " << cv::Mat(object_tracker.GetObject().GetPeriodicity().GetSpectrum(), false);
        }
      }

    } catch (const cv_bridge::Exception& e) {
      ROS_ERROR("cv_bridge exception: %s", e.what());
    }

    ;
  }

public:
  ObzerverROS(const int queue_size):
    ros_it(ros_nh),
    frame_counter(0),
    ticker(StepBenchmarker::GetInstance())
  {
    update_params();
    sub_image = ros_it.subscribe("image", queue_size, &ObzerverROS::ImageCallback, this);

    if (enable_debug_image) {
      ROS_INFO("debug_image is enabled.");
      pub_debug_image = ros_it.advertise("debug_image", 1);
    }

    if (enable_diff_image) {
      ROS_INFO("diff_image is enabled.");
      pub_diff_image = ros_it.advertise("diff_image", 1);
    }

    if (enable_stablized_image) {
      ROS_INFO("stablized_image is enabled.");
      pub_stablized_image = ros_it.advertise("stablized_image", 1);
    }

    if (enable_simmat_image) {
      ROS_INFO("simmat_image is enabled.");
      pub_simmat_image = ros_it.advertise("simmat_image", 1);
    }

    obz_log_config("obzerver_ros_node", param_log_file);

    feature_detector = new cv::FastFeatureDetector(param_ffd_threshold, true);
    camera_tracker = new CameraTracker(param_hist_len, feature_detector, param_max_features, param_pylk_winsize, param_pylk_iters, param_pylk_eps);
    object_tracker = new ObjectTracker(param_num_particles, param_hist_len, param_fps);

    do_downsample = param_downsample_factor < 1.0 && param_downsample_factor > 0.0;
  }

};

int main(int argc, char* argv[]) {

  /* ROS Stuff */
  ros::init(argc, argv, "obzerver_ros");
  ros::NodeHandle ros_nh;
  int param_queue_size;
  ObzerverROS::get_param<int>("queue_size", param_queue_size, 1);
  ObzerverROS obzerver_ros(param_queue_size);

  /* Main Loop */
  ROS_INFO("Starting obzerver_ros");
  ros::spin();
  return 0;
}
