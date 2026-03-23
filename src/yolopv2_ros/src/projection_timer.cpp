#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_eigen/tf2_eigen.h>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <chrono> // 시간 측정용 헤더 추가

class ProposedROILogicTimer {
private:
    ros::NodeHandle nh;
    ros::Publisher road_pub, obs_pub;
    tf2_ros::Buffer tf_buffer;
    tf2_ros::TransformListener tf_listener;

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::PointCloud2> MySyncPolicy;
    std::shared_ptr<message_filters::Synchronizer<MySyncPolicy>> sync;
    message_filters::Subscriber<sensor_msgs::Image> mask_sub;
    message_filters::Subscriber<sensor_msgs::PointCloud2> lidar_sub;

public:
    ProposedROILogicTimer() : tf_listener(tf_buffer) {
        mask_sub.subscribe(nh, "/yolop_fusion_masks", 10);
        lidar_sub.subscribe(nh, "/lidar3D", 10);

        sync = std::make_shared<message_filters::Synchronizer<MySyncPolicy>>(MySyncPolicy(100), mask_sub, lidar_sub);
        sync->registerCallback(boost::bind(&ProposedROILogicTimer::callback, this, _1, _2));

        road_pub = nh.advertise<sensor_msgs::PointCloud2>("/lidar_road", 1);
        obs_pub = nh.advertise<sensor_msgs::PointCloud2>("/lidar_obstacle", 1);
        
        ROS_INFO("Algorithm Performance Measurement Node Started");
    }

    void callback(const sensor_msgs::ImageConstPtr& mask_msg, const sensor_msgs::PointCloud2ConstPtr& lidar_msg) {
        // --- [1. 측정 시작] ---
        auto start_time = std::chrono::high_resolution_clock::now();

        // 데이터 변환 (이 과정은 알고리즘의 전처리로 포함됩니다)
        cv::Mat fusion_mask;
        try {
            fusion_mask = cv_bridge::toCvShare(mask_msg, "bgr8")->image;
        } catch (cv_bridge::Exception& e) { return; }

        geometry_msgs::TransformStamped lidar_to_cam;
        try {
            lidar_to_cam = tf_buffer.lookupTransform("Camera-3", "lidar_link", ros::Time(0));
        } catch (tf2::TransformException& ex) { return; }

        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::fromROSMsg(*lidar_msg, *cloud);

        pcl::PointCloud<pcl::PointXYZ> road_out, obs_out;
        Eigen::Affine3d T = tf2::transformToEigen(lidar_to_cam);

        // --- [2. 핵심 로직: Projection & Semantic Filtering] ---
        // 논문 식 (6), (8), (9)에 해당하는 부분입니다 [cite: 82, 93, 98]
        for (const auto& pt : cloud->points) {
            Eigen::Vector3d p_l(pt.x, pt.y, pt.z);
            Eigen::Vector3d p_c = T * p_l;
            double xc = -p_c.y(), yc = -p_c.z(), zc = p_c.x();

            if (zc > 0.1) {
                // 영상 평면 투영 (Coordinate Transformation) [cite: 84]
                int u = static_cast<int>((320.0 * xc / zc) + 320.0);
                int v = static_cast<int>((320.0 * yc / zc) + 240.0);

                if (u >= 0 && u < 640 && v >= 0 && v < 480) {
                    cv::Vec3b pixel = fusion_mask.at<cv::Vec3b>(v, u);
                    
                    // Semantic Mask 기반 필터링 [cite: 96]
                    if (pixel[1] > 127) { // Obstacle Mask (Green-ish in BGR)
                        obs_out.push_back(pcl::PointXYZ(pt.x, pt.y, pt.z));
                    } else if (pixel[0] > 127) { // Road Mask (Blue-ish in BGR)
                        road_out.push_back(pcl::PointXYZ(pt.x, pt.y, pt.z));
                    }
                }
            }
        }

        // --- [3. 측정 종료] ---
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end_time - start_time;

        // 결과 출력 (이 로그 출력 시간은 측정에 포함되지 않습니다)
        ROS_INFO("Logic Processing Time: %.4f ms | Points: %zu", elapsed.count(), road_out.size() + obs_out.size());

        // 결과 퍼블리시 (시각화용)
        sensor_msgs::PointCloud2 r_msg, o_msg;
        pcl::toROSMsg(road_out, r_msg); r_msg.header = lidar_msg->header;
        pcl::toROSMsg(obs_out, o_msg); o_msg.header = lidar_msg->header;
        road_pub.publish(r_msg); obs_pub.publish(o_msg);
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "projection_timer");
    ProposedROILogicTimer prp;
    ros::spin();
    return 0;
}