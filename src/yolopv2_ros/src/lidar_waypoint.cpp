#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Path.h>
#include <morai_msgs/EgoVehicleStatus.h>
#include <morai_msgs/ObjectStatusList.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_eigen/tf2_eigen.h>
#include <Eigen/Dense>
#include <cmath>
#include <memory>
#include <vector>

struct TargetVehicle {
    int unique_id;
    float l, w, h;               
    double rel_x, rel_y, rel_z;  
    double rel_yaw;              
    
    int n_obj_full = 0;          
    int n_obj_roi = 0;           
    
    bool is_updated = false;

    void reset() {
        n_obj_full = 0;
        n_obj_roi = 0;
    }

    bool isInside(double px, double py, double pz) {
        if (!is_updated) return false;
        double dx = px - rel_x;
        double dy = py - rel_y;
        double dz = pz - rel_z;

        double cos_y = std::cos(-rel_yaw);
        double sin_y = std::sin(-rel_yaw);
        double rx = dx * cos_y - dy * sin_y;
        double ry = dx * sin_y + dy * cos_y;

        return (std::abs(rx) <= l/2.0 && std::abs(ry) <= w/2.0 && std::abs(dz) <= h/2.0);
    }
};

class PerfectWaypointROIProjector {
private:
    ros::NodeHandle nh;
    ros::Publisher roi_pub;
    ros::Subscriber obj_sub, path_sub;
    
    tf2_ros::Buffer tf_buffer;
    tf2_ros::TransformListener tf_listener;

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, morai_msgs::EgoVehicleStatus> MySyncPolicy;
    std::shared_ptr<message_filters::Synchronizer<MySyncPolicy>> sync;
    message_filters::Subscriber<sensor_msgs::PointCloud2> lidar_sub;
    message_filters::Subscriber<morai_msgs::EgoVehicleStatus> ego_sub;

    nav_msgs::Path global_path;
    std::vector<TargetVehicle> targets; 
    double R_y = 2.2; 

public:
    PerfectWaypointROIProjector() : tf_listener(tf_buffer) {
        ros::NodeHandle pnh("~");
        int t_id;
        pnh.param<int>("target_id_1", t_id, 2); 
        targets.push_back({t_id});

        roi_pub = nh.advertise<sensor_msgs::PointCloud2>("/lidar_waypoint_roi", 1);
        path_sub = nh.subscribe("/global_path", 1, &PerfectWaypointROIProjector::pathCallback, this);
        obj_sub = nh.subscribe("/Object_topic", 10, &PerfectWaypointROIProjector::objectCallback, this);

        lidar_sub.subscribe(nh, "/lidar3D", 10);
        ego_sub.subscribe(nh, "/Ego_topic", 10);

        sync = std::make_shared<message_filters::Synchronizer<MySyncPolicy>>(MySyncPolicy(100), lidar_sub, ego_sub);
        sync->registerCallback(boost::bind(&PerfectWaypointROIProjector::callback, this, _1, _2));
        
        ROS_INFO("Final Waypoint-ROI Analysis Node Started");
    }

    ~PerfectWaypointROIProjector() {
        if (sync) sync.reset();
        ROS_INFO("the node has safely shut down");
    }

    void pathCallback(const nav_msgs::Path::ConstPtr& msg) { global_path = *msg; }

    void objectCallback(const morai_msgs::ObjectStatusListConstPtr& msg) {
        geometry_msgs::TransformStamped map_to_lidar;
        try {
            map_to_lidar = tf_buffer.lookupTransform("lidar_link", "map", ros::Time(0));
        } catch (tf2::TransformException& ex) { return; }

        for (auto& target : targets) {
            bool found = false;
            morai_msgs::ObjectStatus target_obj;

            // 1. NPC 리스트 수색
            for (const auto& obj : msg->npc_list) {
                if (obj.unique_id == target.unique_id) { target_obj = obj; found = true; break; }
            }
            // 2. 보행자 리스트 수색
            if (!found) {
                for (const auto& obj : msg->pedestrian_list) {
                    if (obj.unique_id == target.unique_id) { target_obj = obj; found = true; break; }
                }
            }
            // 3. 장애물 리스트 수색
            if (!found) {
                for (const auto& obj : msg->obstacle_list) {
                    if (obj.unique_id == target.unique_id) { target_obj = obj; found = true; break; }
                }
            }

            if (found) {
                geometry_msgs::PoseStamped world_pose, lidar_pose;
                world_pose.header.frame_id = "map";
                // 에러 방지를 위한 개별 좌표 대입
                world_pose.pose.position.x = target_obj.position.x;
                world_pose.pose.position.y = target_obj.position.y;
                world_pose.pose.position.z = target_obj.position.z;
                
                tf2::Quaternion q;
                q.setRPY(0, 0, target_obj.heading * M_PI / 180.0);
                world_pose.pose.orientation = tf2::toMsg(q);

                tf2::doTransform(world_pose, lidar_pose, map_to_lidar);

                target.rel_x = lidar_pose.pose.position.x;
                target.rel_y = lidar_pose.pose.position.y;
                target.rel_z = lidar_pose.pose.position.z;
                
                tf2::Quaternion rel_q;
                tf2::fromMsg(lidar_pose.pose.orientation, rel_q);
                double r, p, y;
                tf2::Matrix3x3(rel_q).getRPY(r, p, y);
                target.rel_yaw = y;

                target.l = target_obj.size.x; target.w = target_obj.size.y; target.h = target_obj.size.z;
                target.is_updated = true;
            }
        }
    }

    void callback(const sensor_msgs::PointCloud2ConstPtr& lidar_msg, const morai_msgs::EgoVehicleStatus::ConstPtr& ego_msg) {
        if (global_path.poses.empty()) return;
        
        bool all_updated = true;
        for(const auto& t : targets) if(!t.is_updated) all_updated = false;
        if (!all_updated) return;

        double R_x_front = 41.0 / 3.6 * 2.35 + (std::pow(41.0/3.6, 2)/(2*1.5)) + 10.0;
        double heading_rad = ego_msg->heading * M_PI / 180.0;

        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::fromROSMsg(*lidar_msg, *cloud);
        pcl::PointCloud<pcl::PointXYZI> filtered_cloud;

        for(auto& t : targets) t.reset();
        int n_roi_total = 0;

        for (const auto& pt : cloud->points) {
            // 1. N_obj_Full 카운트
            for(auto& t : targets) if (t.isInside(pt.x, pt.y, pt.z)) t.n_obj_full++;

            // 2. Waypoint ROI 필터링 로직
            if (pt.x <= 0.0 || pt.x > R_x_front) continue;

            double map_x = ego_msg->position.x + (pt.x * std::cos(heading_rad) - pt.y * std::sin(heading_rad));
            double map_y = ego_msg->position.y + (pt.x * std::sin(heading_rad) + pt.y * std::cos(heading_rad));

            bool is_in_waypoint_roi = false;
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
                n_roi_total++;
                // 3. N_obj_ROI 카운트
                for(auto& t : targets) if (t.isInside(pt.x, pt.y, pt.z)) t.n_obj_roi++;
            }
        }

        // --- [출력 로그 - 왕자님 요청대로 100% 동일 포맷] ---
        for(const auto& t : targets) {
            double opr = (n_roi_total > 0) ? (double)t.n_obj_roi / n_roi_total : 0.0;
            double oppr = (t.n_obj_full > 0) ? (double)t.n_obj_roi / t.n_obj_full : 0.0;
            double dist = std::sqrt(t.rel_x*t.rel_x + t.rel_y*t.rel_y);

            ROS_INFO("=========================================");
            ROS_INFO("Vehicle ID %d", t.unique_id);
            ROS_INFO(">> N_ROI: %d | N_obj_Full: %d | N_obj_ROI: %d", n_roi_total, t.n_obj_full, t.n_obj_roi);
            ROS_INFO(">> OPR (Purity): %.4f", opr);
            ROS_INFO(">> OPPR (Preservation): %.4f", oppr);
        }

        sensor_msgs::PointCloud2 output;
        pcl::toROSMsg(filtered_cloud, output);
        output.header = lidar_msg->header;
        roi_pub.publish(output);
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "waypoint_roi_analyzer");
    PerfectWaypointROIProjector pwrp;
    ros::spin();
    return 0;
}

/*
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
#include <cmath>
#include <memory>
#include <vector>

// [차량별 데이터를 개별 관리하기 위한 구조체]
struct TargetVehicle {
    float id;           // Intensity 기반 ID
    int p_actual = 0;   // 전체 점 개수
    int p_preserved = 0;// ROI 내 보존된 점 개수
    double dist_sum = 0.0; // 탐지 거리 합계

    bool is_recorded = false; 
    double first_detection_dist = 0.0;

    void reset() {
        p_actual = 0;
        p_preserved = 0;
        dist_sum = 0.0;
    }
};

class WaypointROIProjector {
private:
    ros::NodeHandle nh;
    ros::Publisher roi_pub;
    
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, morai_msgs::EgoVehicleStatus> MySyncPolicy;
    std::shared_ptr<message_filters::Synchronizer<MySyncPolicy>> sync;
    message_filters::Subscriber<sensor_msgs::PointCloud2> lidar_sub;
    message_filters::Subscriber<morai_msgs::EgoVehicleStatus> ego_sub;
    ros::Subscriber path_sub;

    nav_msgs::Path global_path;
    std::vector<TargetVehicle> targets; // 추적 대상 리스트
    double R_y = 2.2; 
    const int N = 5;

public:
    WaypointROIProjector() {
        ros::NodeHandle pnh("~");
        
        // 다중 차량 ID 설정 (타 노드들과 파라미터 이름 통일)
        float id1, id2;
        pnh.param<float>("target_id_1", id1, 1.0f);
        pnh.param<float>("target_id_2", id2, 2.0f);
        
        targets.push_back({id1});
        targets.push_back({id2});

        roi_pub = nh.advertise<sensor_msgs::PointCloud2>("/lidar_waypoint_roi", 1);
        path_sub = nh.subscribe("/global_path", 1, &WaypointROIProjector::pathCallback, this);

        lidar_sub.subscribe(nh, "/lidar3D", 10);
        ego_sub.subscribe(nh, "/Ego_topic", 10);

        sync = std::make_shared<message_filters::Synchronizer<MySyncPolicy>>(MySyncPolicy(100), lidar_sub, ego_sub);
        sync->getPolicy()->setMaxIntervalDuration(ros::Duration(0.2));
        sync->registerCallback(boost::bind(&WaypointROIProjector::callback, this, _1, _2));
        
        ROS_INFO("Multi-Target Waypoint ROI Analysis Node Started");
    }

    void printFinalReport() {
        std::cout << "\n=========================================" << std::endl;
        std::cout << "       [ WAYPOINT ROI FINAL REPORT ]     " << std::endl;
        std::cout << "=========================================" << std::endl;
        for (size_t i = 0; i < targets.size(); ++i) {
            if (targets[i].is_recorded) {
                // 이 거리가 바로 왕자님이 원하시던 '가장 처음 찍힌 거리'예요!
                printf("Vehicle %zu (ID: %.0f) First Detected at: %.2f m", 
                    i + 1, targets[i].id, targets[i].first_detection_dist);
            } else {
                printf("Vehicle %zu (ID: %.0f) was NEVER Detected", i + 1, targets[i].id);
            }
        }
        std::cout << "=========================================\n" << std::endl;
    }

    void pathCallback(const nav_msgs::Path::ConstPtr& msg) {
        global_path = *msg;
    }

    double calculateRxFront(double v_kph) {
        //double v = v_kph / 3.6;
        //double t_total = 2.35;
        //double a_brake = 1.5;
        //double C_s = 10.0;

        double fixed_v_kph = 41.0; 
        double v = fixed_v_kph / 3.6;     
        double t_total = 2.35;      
        double a_brake = 1.5;       
        double C_s = 10.0;

        return v * t_total + (v * v) / (2.0 * a_brake) + C_s;
    }

    void callback(const sensor_msgs::PointCloud2ConstPtr& lidar_msg, const morai_msgs::EgoVehicleStatus::ConstPtr& ego_msg) {
        if (global_path.poses.empty()) return;

        double R_x_front = calculateRxFront(ego_msg->velocity.x);
        double heading_rad = ego_msg->heading * M_PI / 180.0;

        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::fromROSMsg(*lidar_msg, *cloud);
        pcl::PointCloud<pcl::PointXYZI> filtered_cloud;

        // 매 프레임 타겟 데이터 초기화
        for(auto& t : targets) t.reset();

        for (const auto& pt : cloud->points) {
            // 1. 전체 포인트 중 타겟 ID 확인 (P_actual 카운트)
            for(auto& t : targets) {
                if (std::abs(pt.intensity - (t.id / 255.0f)) < 0.001f) {
                    t.p_actual++;
                    break;
                }
            }

            // 2. 경로 기반 필터링 로직 (Forward & Waypoint Distance)
            if (pt.x <= 0.0 || pt.x > R_x_front) continue;

            double map_x = ego_msg->position.x + (pt.x * std::cos(heading_rad) - pt.y * std::sin(heading_rad));
            double map_y = ego_msg->position.y + (pt.x * std::sin(heading_rad) + pt.y * std::cos(heading_rad));

            bool is_inside = false;
            for (const auto& pose : global_path.poses) {
                double dx = map_x - pose.pose.position.x;
                double dy = map_y - pose.pose.position.y;
                if (std::sqrt(dx*dx + dy*dy) < R_y) {
                    is_inside = true;
                    break;
                }
            }

            if (is_inside) {
                filtered_cloud.push_back(pt);
                // 3. ROI 내 생존한 점이 타겟인지 확인 (P_preserved 카운트)
                for(auto& t : targets) {
                    if (std::abs(pt.intensity - (t.id / 255.0f)) < 0.001f) {
                        t.p_preserved++;
                        t.dist_sum += std::sqrt(pt.x * pt.x + pt.y * pt.y);
                        break;
                    }
                }
            }
        }

        // --- [최종 로그 출력부] ---
        ROS_INFO("=========================================");
        size_t total_points = cloud->size();
        size_t projected_points = filtered_cloud.size();
        double data_efficiency = (total_points > 0) ? (double)projected_points / total_points * 100.0 : 0.0;
        
        //ROS_INFO("0) Raw LiDAR Total Points: %zu", total_points);
        //ROS_INFO("1) LiDAR ROI Points: %zu", projected_points);
        //ROS_INFO("2) Data Efficiency        : %.2f%% of Original", data_efficiency);
        //ROS_INFO("-----------------------------------------");

        // 차량별 개별 로그 출력 (동적/고정 ROI와 로그 양식 100% 일치)
        for(size_t i = 0; i < targets.size(); ++i) {
            auto& t = targets[i];
            double opr = (t.p_actual > 0) ? (double)t.p_preserved / t.p_actual * 100.0 : 0.0;
            double reaction_dist = (t.p_preserved >= N) ? t.dist_sum / t.p_preserved : 0.0;

            if (reaction_dist > 0) {
                // [핵심 로직] 아직 기록되지 않은 경우에만 최초 거리를 기록!
                if (!t.is_recorded) {
                    t.first_detection_dist = reaction_dist;
                    t.is_recorded = true;
                    ROS_WARN("!!! Vehicle %zu (ID: %.0f) FIRST DETECTION: %.2f m !!!", i + 1, t.id, reaction_dist);
                }
                ROS_INFO("Vehicle %zu - Detection Dist: %.2f m", i + 1, reaction_dist);
            } else {
                ROS_INFO("Vehicle %zu - Detection Dist: Not Detected", i + 1);
            }
        }
        ROS_INFO("=========================================");

        sensor_msgs::PointCloud2 output;
        pcl::toROSMsg(filtered_cloud, output);
        output.header = lidar_msg->header;
        roi_pub.publish(output);
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "waypoint_roi_analysis");
    WaypointROIProjector wrp;
    ros::spin();
    wrp.printFinalReport();
    return 0;
}
*/

/*
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
#include <cmath>
#include <memory>

class WaypointROIProjector {
private:
    ros::NodeHandle nh;
    ros::Publisher roi_pub;
    
    // 동기화: LiDAR와 차량 상태(Ego)의 시간을 일치시킴
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, morai_msgs::EgoVehicleStatus> MySyncPolicy;
    std::shared_ptr<message_filters::Synchronizer<MySyncPolicy>> sync;
    message_filters::Subscriber<sensor_msgs::PointCloud2> lidar_sub;
    message_filters::Subscriber<morai_msgs::EgoVehicleStatus> ego_sub;
    ros::Subscriber path_sub;

    nav_msgs::Path global_path;
    float target_actor_id;
    double R_y = 2.2; 
    const int N = 5;

public:
    WaypointROIProjector() {
        ros::NodeHandle pnh("~");
        pnh.param<float>("target_id", target_actor_id, 1.0f);

        roi_pub = nh.advertise<sensor_msgs::PointCloud2>("/lidar_waypoint_roi", 1);
        path_sub = nh.subscribe("/global_path", 1, &WaypointROIProjector::pathCallback, this);

        // [동기화 구독]
        lidar_sub.subscribe(nh, "/lidar3D", 10);
        ego_sub.subscribe(nh, "/Ego_topic", 10);

        sync = std::make_shared<message_filters::Synchronizer<MySyncPolicy>>(MySyncPolicy(100), lidar_sub, ego_sub);
        sync->getPolicy()->setMaxIntervalDuration(ros::Duration(0.2)); // 최대 0.2초 차이까지 허용
        sync->registerCallback(boost::bind(&WaypointROIProjector::callback, this, _1, _2));
        
        ROS_INFO("🚀 Waypoint ROI Analysis Node Started (Logging Enhanced)");
    }

    void pathCallback(const nav_msgs::Path::ConstPtr& msg) {
        global_path = *msg;
    }

    double calculateRxFront(double v_kph) {
        double v = v_kph / 3.6;
        double t_total = 2.35;
        double a_brake = 1.5;
        double C_s = 10.0;
        return v * t_total + (v * v) / (2.0 * a_brake) + C_s;
    }

    void callback(const sensor_msgs::PointCloud2ConstPtr& lidar_msg, const morai_msgs::EgoVehicleStatus::ConstPtr& ego_msg) {
        if (global_path.poses.empty()) return;

        double R_x_front = calculateRxFront(ego_msg->velocity.x);
        double heading_rad = ego_msg->heading * M_PI / 180.0;

        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::fromROSMsg(*lidar_msg, *cloud);
        pcl::PointCloud<pcl::PointXYZI> filtered_cloud;

        int P_actual = 0;
        int P_preserved = 0;
        double dist_sum = 0.0;

        for (const auto& pt : cloud->points) {
            bool is_target = (std::abs(pt.intensity - (target_actor_id / 255.0f)) < 0.001f);
            if (is_target) P_actual++;

            if (pt.x <= 0.0 || pt.x > R_x_front) continue;

            double map_x = ego_msg->position.x + (pt.x * std::cos(heading_rad) - pt.y * std::sin(heading_rad));
            double map_y = ego_msg->position.y + (pt.x * std::sin(heading_rad) + pt.y * std::cos(heading_rad));

            bool is_inside = false;
            for (const auto& pose : global_path.poses) {
                double dx = map_x - pose.pose.position.x;
                double dy = map_y - pose.pose.position.y;
                if (std::sqrt(dx*dx + dy*dy) < R_y) {
                    is_inside = true;
                    break;
                }
            }

            if (is_inside) {
                filtered_cloud.push_back(pt);
                if (is_target) {
                    P_preserved++;
                    dist_sum += std::sqrt(pt.x * pt.x + pt.y * pt.y);
                }
            }
        }

        // --- 지표 및 로그 계산 시작 ---
        size_t total_points = cloud->size();
        size_t projected_points = filtered_cloud.size();
        double data_efficiency = (total_points > 0) ? (double)projected_points / total_points * 100.0 : 0.0;
        double opr = (P_actual > 0) ? (double)P_preserved / P_actual * 100.0 : 0.0;
        double reaction_dist = (P_preserved >= N) ? dist_sum / P_preserved : 0.0;

        // [왕자님이 요청하신 로그 형식 완전 복구]
        ROS_INFO("=========================================");
        //ROS_INFO("0) Raw LiDAR Total Points: %zu", total_points);
        ROS_INFO("1) LiDAR ROI Points     : %zu", projected_points); // Waypoint 내부 점
        ROS_INFO("2) Data Efficiency    : %.2f%% of Original", data_efficiency);
        ROS_INFO("-----------------------------------------");
        ROS_INFO("P_actual (Total)   : %d", P_actual);
        ROS_INFO("P_preserved (ROI) : %d", P_preserved);
        ROS_INFO("OPR (Safety)     : %.2f%%", opr);
        
        if (reaction_dist > 0) {
            ROS_INFO("Detection Dist   : %.2f m", reaction_dist);
        } else {
            ROS_INFO("Detection Dist   : Not Detected");
        }
        ROS_INFO("=========================================");

        sensor_msgs::PointCloud2 output;
        pcl::toROSMsg(filtered_cloud, output);
        output.header = lidar_msg->header;
        roi_pub.publish(output);
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "waypoint_roi_analysis");
    WaypointROIProjector wrp;
    ros::spin();
    return 0;
}
*/