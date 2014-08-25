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
  ros::Publisher pub_object;

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
  int param_skip_dc_term_count;

  bool enable_debug_image;
  bool enable_stablized_image;
  bool enable_diff_image;
  bool enable_simmat_image;

  cv::Mat frame_input;
  cv::Mat frame_gray;

  cv_bridge::CvImage frame_debug_cvi;
  cv_bridge::CvImage frame_sim_cvi;
  cv_bridge::CvImage frame_stab_cvi;
  cv_bridge::CvImage frame_diff_cvi;
  obzerver_ros::object out_object;

  bool do_convert;
  bool do_downsample;
  cv::Ptr<cv::FeatureDetector> feature_detector;
  cv::Ptr<CameraTracker> camera_tracker;
  cv::Ptr<ObjectTracker> object_tracker;
  std::size_t frame_counter;

  StepBenchmarker& ticker;

  void update_params()
  {
    get_param<bool>("~enable_debug_image", enable_debug_image, false);
    get_param<bool>("~enable_stablized_image", enable_stablized_image, false);
    get_param<bool>("~enable_diff_image", enable_diff_image, false);
    get_param<bool>("~enable_simmat_image", enable_simmat_image, false);

    get_param<std::string>("~obz_logfile", param_log_file, std::string(""));
    get_param<float>("~downsample_factor", param_downsample_factor, 1.0);
    get_param<float>("~fps", param_fps, 30.0);
    get_param<int>("~max_features", param_max_features, 300);
    get_param<int>("~num_particles", param_num_particles, 1000);
    get_param<int>("~history_length", param_hist_len, 90);
    get_param<int>("~pylk_winsize", param_pylk_winsize, 30);
    get_param<int>("~pylk_iterations", param_pylk_iters, 30);
    get_param<double>("~pylk_eps", param_pylk_eps, 0.01);
    get_param<int>("~ffd_threshold", param_ffd_threshold, 30);
    get_param<int>("~skip_dc_term_count", param_skip_dc_term_count, 1);
  }

  void ImageCallback(const sensor_msgs::ImageConstPtr& msg) {
    ticker.reset();
    float _f = -1.0;
    cv_bridge::CvImageConstPtr frame_input_cvptr;
    try {
      if (sensor_msgs::image_encodings::isColor(msg->encoding)) {
        ROS_WARN_ONCE("Input image is BGR8");
        frame_input_cvptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
        do_convert = true;
      } else {
        ROS_WARN_ONCE("Input image is MONO8");
        frame_input_cvptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::MONO8);
        do_convert = false;
      }

      frame_input = frame_input_cvptr->image; // No Copy For Now
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

      const bool ct_success = camera_tracker->Update(frame_gray, frame_input);

      out_object.header.stamp = ros::Time::now();
      out_object.header.frame_id = frame_input_cvptr->header.frame_id;
      out_object.status = 0;
      if (!ct_success) {
        LOG(WARNING) << "Camera Tracker Failed";
        // TODO
      } else {
        object_tracker->Update(camera_tracker->GetStablizedGray(),
                               camera_tracker->GetLatestDiff(),
                               camera_tracker->GetLatestSOF(),
                               camera_tracker->GetLatestCameraTransform());

        out_object.status = object_tracker->GetStatus();

        LOG(INFO) << "Tracking status: " << object_tracker->GetStatus();
        if (object_tracker->IsTracking()) {
          _f = object_tracker->GetObject().GetPeriodicity().GetDominantFrequency(param_skip_dc_term_count); // TODO
          LOG(INFO) << "Object: "
                    << object_tracker->GetObjectBoundingBox()
                    << " Periodicity:"
                    << _f;
          out_object.roi.x_offset = object_tracker->GetObjectBoundingBox().tl().x;
          out_object.roi.y_offset = object_tracker->GetObjectBoundingBox().tl().y;
          out_object.roi.width = object_tracker->GetObjectBoundingBox().width;
          out_object.roi.height = object_tracker->GetObjectBoundingBox().height;
          out_object.spectrum = object_tracker->GetObject().GetPeriodicity().GetSpectrum();
          out_object.dominant_freq = _f;
        }

        // Publish

        if (pub_object.getNumSubscribers() > 0) {
          pub_object.publish(out_object);
        }

        if (
            enable_stablized_image &&
            pub_stablized_image.getNumSubscribers() > 0 &&
            camera_tracker->GetStablizedGray().data
            )
        {
          frame_stab_cvi.image = camera_tracker->GetStablizedGray();
          frame_stab_cvi.header.frame_id = frame_input_cvptr->header.frame_id;
          frame_stab_cvi.header.stamp = frame_input_cvptr->header.stamp;
          frame_stab_cvi.encoding = "mono8";
          object_tracker->DrawParticles(frame_stab_cvi.image);
          pub_stablized_image.publish(frame_stab_cvi.toImageMsg());
        }

        if (
            enable_debug_image &&
            pub_debug_image.getNumSubscribers() > 0 &&
            frame_input.data
            )
        {
          frame_debug_cvi.image = frame_input;
          frame_debug_cvi.header.frame_id = frame_input_cvptr->header.frame_id;
          frame_debug_cvi.header.stamp = frame_input_cvptr->header.stamp;
          frame_debug_cvi.encoding = do_convert ? "bgr8" : "mono8";

          drawFeaturePointsTrajectory(frame_debug_cvi.image,
                                      camera_tracker->GetHomographyOutliers(),
                                      camera_tracker->GetTrackedFeaturesPrev(),
                                      camera_tracker->GetTrackedFeaturesCurr(),
                                      2,
                                      cv::Scalar(127,127,127),
                                      cv::Scalar(0, 0, 255),
                                      cv::Scalar(0, 0, 255));

          std::stringstream ss;
          ss << std::setprecision(5) << "Periodicity: " << _f;
          cv::putText(frame_debug_cvi.image, ss.str(), cv::Point(40,40), 1, CV_FONT_HERSHEY_PLAIN, cv::Scalar(0, 0, 255));

          pub_debug_image.publish(frame_debug_cvi.toImageMsg());
        }

        if (
            enable_diff_image &&
            pub_diff_image.getNumSubscribers() > 0 &&
            camera_tracker->GetLatestDiff().data
            )
        {
          frame_diff_cvi.image = camera_tracker->GetLatestDiff();
          frame_diff_cvi.header.frame_id = frame_input_cvptr->header.frame_id;
          frame_diff_cvi.header.stamp = frame_input_cvptr->header.stamp;
          frame_diff_cvi.encoding = "mono8";
          pub_diff_image.publish(frame_diff_cvi.toImageMsg());
        }

        if (
            enable_simmat_image &&
            pub_simmat_image.getNumSubscribers() > 0 &&
            object_tracker->IsTracking()
            )
        {
          frame_sim_cvi.image = object_tracker->GetObject().GetSelfSimilarity().GetSimMatrixRendered();
          frame_sim_cvi.header.frame_id = frame_input_cvptr->header.frame_id;
          frame_sim_cvi.header.stamp = frame_input_cvptr->header.stamp;
          frame_sim_cvi.encoding = "mono8";
          pub_simmat_image.publish(frame_sim_cvi.toImageMsg());
        }

        ticker.tick("Visualization");
        frame_counter++;
        LOG(INFO) << ticker.getstr();
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
    sub_image = ros_it.subscribe("obzerver/image", queue_size, &ObzerverROS::ImageCallback, this);
    pub_object = ros_nh.advertise<obzerver_ros::object>("obzerver/object", 20);

    if (enable_debug_image) {
      ROS_INFO("debug_image is enabled.");
      pub_debug_image = ros_it.advertise("obzerver/debug_image", 1);
    }

    if (enable_diff_image) {
      ROS_INFO("diff_image is enabled.");
      pub_diff_image = ros_it.advertise("obzerver/diff_image", 1);
    }

    if (enable_stablized_image) {
      ROS_INFO("stablized_image is enabled.");
      pub_stablized_image = ros_it.advertise("obzerver/stablized_image", 1);
    }

    if (enable_simmat_image) {
      ROS_INFO("simmat_image is enabled.");
      pub_simmat_image = ros_it.advertise("obzerver/simmat_image", 1);
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
  ObzerverROS::get_param<int>("~queue_size", param_queue_size, 1);
  ObzerverROS obzerver_ros(param_queue_size);

  /* Main Loop */
  ROS_INFO("Starting obzerver_ros");
  ros::spin();
  return 0;
}
