#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <cv_bridge/cv_bridge.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_eigen/tf2_eigen.h>
#include <Eigen/Dense>
#include <chrono> // 시간 측정용 헤더 추가

class FixedROILogicTimer {
private:
    ros::NodeHandle nh;
    ros::Subscriber lidar_sub;
    ros::Publisher fixed_roi_pub;
    
    tf2_ros::Buffer tf_buffer;
    tf2_ros::TransformListener tf_listener;

public:
    FixedROILogicTimer() : tf_listener(tf_buffer) {
        // Fixed ROI는 마스크가 필요 없으므로 LiDAR 토픽만 구독해도 충분해요!
        lidar_sub = nh.subscribe("/lidar3D", 10, &FixedROILogicTimer::lidarCallback, this);
        fixed_roi_pub = nh.advertise<sensor_msgs::PointCloud2>("/lidar_fixed_roi", 1);
        
        ROS_INFO("Fixed ROI Performance Measurement Node Started");
    }

    void lidarCallback(const sensor_msgs::PointCloud2ConstPtr& lidar_msg) {
        // --- [1. 측정 시작] ---
        auto start_time = std::chrono::high_resolution_clock::now();

        // 데이터 변환 및 TF 조회 (전처리 단계)
        geometry_msgs::TransformStamped lidar_to_cam;
        try {
            // 카메라와 LiDAR 사이의 변환 정보를 가져와요 [cite: 68]
            lidar_to_cam = tf_buffer.lookupTransform("Camera-3", "lidar_link", ros::Time(0));
        } catch (tf2::TransformException& ex) { return; }

        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::fromROSMsg(*lidar_msg, *cloud);

        pcl::PointCloud<pcl::PointXYZ> fixed_roi_out;
        Eigen::Affine3d T = tf2::transformToEigen(lidar_to_cam);

        // --- [2. 핵심 로직: Fixed ROI Projection & Filtering] ---
        // 논문 식 (6)에 따른 투영 과정입니다 [cite: 82]
        for (const auto& pt : cloud->points) {
            Eigen::Vector3d p_l(pt.x, pt.y, pt.z);
            Eigen::Vector3d p_c = T * p_l; // LiDAR 점을 카메라 좌표계로 변환 [cite: 72]
            double xc = -p_c.y(), yc = -p_c.z(), zc = p_c.x();

            if (zc > 0.1) {
                // 영상 평면 좌표로 투영합니다 [cite: 84]
                int u = static_cast<int>((320.0 * xc / zc) + 320.0);
                int v = static_cast<int>((320.0 * yc / zc) + 240.0);

                // 마스크를 확인하지 않고, 단순히 이미지 범위(640x480) 내에 있는지 확인해요 [cite: 32, 34]
                if (u >= 0 && u < 640 && v >= 0 && v < 480) {
                    fixed_roi_out.push_back(pcl::PointXYZ(pt.x, pt.y, pt.z));
                }
            }
        }

        // --- [3. 측정 종료] ---
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end_time - start_time;

        // 결과 출력 (이 로그값을 모아서 평균과 표준편차를 구하시면 돼요!)
        ROS_INFO("Fixed ROI Processing Time: %.4f ms | Points: %zu", elapsed.count(), fixed_roi_out.size());

        // 결과 퍼블리시
        sensor_msgs::PointCloud2 roi_msg;
        pcl::toROSMsg(fixed_roi_out, roi_msg);
        roi_msg.header = lidar_msg->header;
        fixed_roi_pub.publish(roi_msg);
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "fixed_timer");
    FixedROILogicTimer frt;
    ros::spin();
    return 0;
}