#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Path.h>
#include <morai_msgs/EgoVehicleStatus.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <chrono> // 시간 측정용

class WaypointROILogicTimer {
private:
    ros::NodeHandle nh;
    ros::Publisher roi_pub;
    ros::Subscriber path_sub;
    
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, morai_msgs::EgoVehicleStatus> MySyncPolicy;
    std::shared_ptr<message_filters::Synchronizer<MySyncPolicy>> sync;
    message_filters::Subscriber<sensor_msgs::PointCloud2> lidar_sub;
    message_filters::Subscriber<morai_msgs::EgoVehicleStatus> ego_sub;

    nav_msgs::Path global_path;
    double R_y = 2.2; // 차선 폭 기준 (논문 설정값) 

public:
    WaypointROILogicTimer() {
        roi_pub = nh.advertise<sensor_msgs::PointCloud2>("/lidar_waypoint_roi", 1);
        path_sub = nh.subscribe("/global_path", 1, &WaypointROILogicTimer::pathCallback, this);

        lidar_sub.subscribe(nh, "/lidar3D", 10);
        ego_sub.subscribe(nh, "/Ego_topic", 10);

        sync = std::make_shared<message_filters::Synchronizer<MySyncPolicy>>(MySyncPolicy(100), lidar_sub, ego_sub);
        sync->registerCallback(boost::bind(&WaypointROILogicTimer::callback, this, _1, _2));
        
        ROS_INFO("Waypoint ROI Performance Measurement Node Started");
    }

    void pathCallback(const nav_msgs::Path::ConstPtr& msg) { global_path = *msg; }

    void callback(const sensor_msgs::PointCloud2ConstPtr& lidar_msg, const morai_msgs::EgoVehicleStatus::ConstPtr& ego_msg) {
        // 경로가 아직 안 들어왔다면 계산하지 않아요!
        if (global_path.poses.empty()) return;

        // --- [1. 측정 시작] ---
        auto start_time = std::chrono::high_resolution_clock::now();

        // 전방 인지 거리 계산 (논문 식 2 관련) [cite: 41, 43]
        double R_x_front = 41.0 / 3.6 * 2.35 + (std::pow(41.0/3.6, 2)/(2*1.5)) + 10.0;
        double heading_rad = ego_msg->heading * M_PI / 180.0;

        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::fromROSMsg(*lidar_msg, *cloud);
        pcl::PointCloud<pcl::PointXYZI> filtered_cloud;

        // --- [2. 핵심 로직: Waypoint-based ROI Filtering] ---
        for (const auto& pt : cloud->points) {
            // 기본 거리 필터링
            if (pt.x <= 0.0 || pt.x > R_x_front) continue;

            // LiDAR 좌표를 지도로 변환 (경로와 비교하기 위해)
            double map_x = ego_msg->position.x + (pt.x * std::cos(heading_rad) - pt.y * std::sin(heading_rad));
            double map_y = ego_msg->position.y + (pt.x * std::sin(heading_rad) + pt.y * std::cos(heading_rad));

            bool is_in_waypoint_roi = false;
            // 각 점마다 경로의 모든 점들과 거리를 비교해요 (연산량이 꽤 많은 부분!) [cite: 39, 44]
            for (const auto& pose : global_path.poses) {
                double dx = map_x - pose.pose.position.x;
                double dy = map_y - pose.pose.position.y;
                if (std::sqrt(dx*dx + dy*dy) < R_y) {
                    is_in_waypoint_roi = true;
                    break;
                }
            }

            if (is_in_waypoint_roi) {
                filtered_cloud.push_back(pt);
            }
        }

        // --- [3. 측정 종료] ---
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end_time - start_time;

        // 결과 출력 (이 값을 논문 <표 1>의 Waypoint ROI Mean에 넣으세요!) [cite: 120]
        ROS_INFO("Waypoint ROI Processing Time: %.4f ms | Points: %zu", elapsed.count(), filtered_cloud.size());

        sensor_msgs::PointCloud2 output;
        pcl::toROSMsg(filtered_cloud, output);
        output.header = lidar_msg->header;
        roi_pub.publish(output);
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "waypoint_timer");
    WaypointROILogicTimer wrt;
    ros::spin();
    return 0;
}