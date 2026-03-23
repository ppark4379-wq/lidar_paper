#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
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
#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>

// [차량 데이터를 관리하는 구조체]
struct TargetVehicle {
    int unique_id;
    float l, w, h;               // GT Size
    double rel_x, rel_y, rel_z;  // Lidar 기준 상대 좌표
    double rel_yaw;              // Lidar 기준 상대 헤딩
    
    int n_obj_full = 0;          // N_obj_Full (원본 전체 객체 점)
    int n_obj_roi = 0;           // N_obj_ROI (ROI 내 객체 점)
    
    bool is_updated = false;

    void reset() {
        n_obj_full = 0;
        n_obj_roi = 0;
    }

    // [기하학적 판정 로직]
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

class PerfectFixedROIProjector {
private:
    ros::NodeHandle nh;
    ros::Subscriber obj_sub;
    ros::Publisher fixed_roi_pub;
    
    tf2_ros::Buffer tf_buffer;
    tf2_ros::TransformListener tf_listener;

    cv::Mat K;
    std::vector<TargetVehicle> targets; // [멀티 타겟 리스트]

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::PointCloud2> MySyncPolicy;
    std::shared_ptr<message_filters::Synchronizer<MySyncPolicy>> sync;
    message_filters::Subscriber<sensor_msgs::Image> mask_sub;
    message_filters::Subscriber<sensor_msgs::PointCloud2> lidar_sub;

public:
    PerfectFixedROIProjector() : tf_listener(tf_buffer) {
        ros::NodeHandle pnh("~"); 
        
        // 1. 파라미터로 여러 대의 타겟 ID를 받습니다.
        int t_id;
        pnh.param<int>("target_id_1", t_id, 2); 
        targets.push_back({t_id});

        //K = (cv::Mat_<double>(3, 3) << 320.0, 0.0, 320.0, 0.0, 320.0, 240.0, 0.0, 0.0, 1.0);

        obj_sub = nh.subscribe("/Object_topic", 10, &PerfectFixedROIProjector::objectCallback, this);
        mask_sub.subscribe(nh, "/yolop_fusion_masks", 10);
        lidar_sub.subscribe(nh, "/lidar3D", 10);

        sync = std::make_shared<message_filters::Synchronizer<MySyncPolicy>>(MySyncPolicy(100), mask_sub, lidar_sub);
        sync->registerCallback(boost::bind(&PerfectFixedROIProjector::lidarCallback, this, _1, _2));

        fixed_roi_pub = nh.advertise<sensor_msgs::PointCloud2>("/lidar_fixed_roi", 1);
        ROS_INFO("Final Multi-Target Fixed-ROI Analysis Node Started");
    }

    ~PerfectFixedROIProjector() {
        if (sync) sync.reset();
        ROS_INFO("the node has safely shut down");
    }

    // [NPC/보행자/장애물 전체 리스트 수색 로직]
    void objectCallback(const morai_msgs::ObjectStatusListConstPtr& msg) {
        geometry_msgs::TransformStamped map_to_lidar;
        try {
            map_to_lidar = tf_buffer.lookupTransform("lidar_link", "map", ros::Time(0));
        } catch (tf2::TransformException& ex) { return; }

        for (auto& target : targets) {
            bool found = false;
            morai_msgs::ObjectStatus target_obj;

            // 1. NPC -> 2. Pedestrian -> 3. Obstacle 순으로 검색
            for (const auto& obj : msg->npc_list) 
                if (obj.unique_id == target.unique_id) { target_obj = obj; found = true; break; }
            if (!found) for (const auto& obj : msg->pedestrian_list) 
                if (obj.unique_id == target.unique_id) { target_obj = obj; found = true; break; }
            if (!found) for (const auto& obj : msg->obstacle_list) 
                if (obj.unique_id == target.unique_id) { target_obj = obj; found = true; break; }

            if (found) {
                geometry_msgs::PoseStamped world_pose, lidar_pose;
                world_pose.header.frame_id = "map";
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

    void lidarCallback(const sensor_msgs::ImageConstPtr& mask_msg, const sensor_msgs::PointCloud2ConstPtr& lidar_msg) {
        // 모든 타겟이 한 번이라도 업데이트 되었는지 확인
        bool all_updated = true;
        for(const auto& t : targets) if(!t.is_updated) all_updated = false;
        if (!all_updated) return;

        geometry_msgs::TransformStamped lidar_to_cam;
        try {
            lidar_to_cam = tf_buffer.lookupTransform("Camera-3", "lidar_link", ros::Time(0));
        } catch (tf2::TransformException& ex) { return; }

        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::fromROSMsg(*lidar_msg, *cloud);

        int n_roi_total = 0;   
        for(auto& t : targets) t.reset();

        Eigen::Affine3d T = tf2::transformToEigen(lidar_to_cam);

        for (const auto& pt : cloud->points) {
            // 1. N_obj_Full 카운트 (멀티 타겟 각각 확인)
            for(auto& t : targets) if (t.isInside(pt.x, pt.y, pt.z)) t.n_obj_full++;

            // 2. Fixed ROI(640x480) 투영 및 필터링
            Eigen::Vector3d p_l(pt.x, pt.y, pt.z);
            Eigen::Vector3d p_c = T * p_l;
            double xc = -p_c.y(), yc = -p_c.z(), zc = p_c.x();

            if (zc > 0.1) {
                int u = static_cast<int>((320.0 * xc / zc) + 320.0);
                int v = static_cast<int>((320.0 * yc / zc) + 240.0);

                if (u >= 0 && u < 640 && v >= 0 && v < 480) {
                    n_roi_total++; // N_ROI 증가

                    // 3. N_obj_ROI 카운트 (멀티 타겟 각각 확인)
                    for(auto& t : targets) if (t.isInside(pt.x, pt.y, pt.z)) t.n_obj_roi++;
                }
            }
        }

        // --- 출력 로그 (다른 노드들과 완벽하게 동일한 포맷) ---
        for(const auto& t : targets) {
            double opr = (n_roi_total > 0) ? (double)t.n_obj_roi / n_roi_total : 0.0;
            double oppr = (t.n_obj_full > 0) ? (double)t.n_obj_roi / t.n_obj_full : 0.0;
            double dist = std::sqrt(t.rel_x*t.rel_x + t.rel_y*t.rel_y);

            ROS_INFO("=========================================");
            //ROS_INFO("Vehicle ID %d | Dist: %.2f m", t.unique_id, dist);
            ROS_INFO("Vehicle ID %d", t.unique_id);
            ROS_INFO(">> N_ROI: %d | N_obj_Full: %d | N_obj_ROI: %d", n_roi_total, t.n_obj_full, t.n_obj_roi);
            ROS_INFO(">> OPR (Purity): %.4f", opr);
            ROS_INFO(">> OPPR (Preservation): %.4f", oppr);
        }
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "fixed_roi_analyzer");
    PerfectFixedROIProjector pfrp;
    ros::spin();
    return 0;
}

/*
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <message_filters/subscriber.h> 
#include <message_filters/synchronizer.h> 
#include <message_filters/sync_policies/approximate_time.h> 
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#include <tf2_eigen/tf2_eigen.h>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
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

class FixedROIProjector {
private:
    ros::NodeHandle nh;
    ros::Publisher fixed_roi_pub;
    
    tf2_ros::Buffer tf_buffer;
    tf2_ros::TransformListener tf_listener;
    
    cv::Mat K;
    const int img_w = 640;
    const int img_h = 480;

    // 다중 추적 대상 차량 리스트
    std::vector<TargetVehicle> targets;
    const int N = 5;

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::PointCloud2> MySyncPolicy;
    std::shared_ptr<message_filters::Synchronizer<MySyncPolicy>> sync;
    message_filters::Subscriber<sensor_msgs::Image> mask_sub;
    message_filters::Subscriber<sensor_msgs::PointCloud2> lidar_sub;

public:
    FixedROIProjector() : tf_listener(tf_buffer) {
        ros::NodeHandle pnh("~");
        
        // 동적 ROI와 동일한 파라미터 구조로 설정
        float id1, id2;
        pnh.param<float>("target_id_1", id1, 1.0f);
        pnh.param<float>("target_id_2", id2, 3.0f); // 왕자님의 로그를 참고하여 기본값을 3.0으로 설정해봤어요!
        
        targets.push_back({id1});
        targets.push_back({id2});

        K = (cv::Mat_<double>(3, 3) << 320.0, 0.0, 320.0, 0.0, 320.0, 240.0, 0.0, 0.0, 1.0);

        mask_sub.subscribe(nh, "/yolop_fusion_masks", 10);
        lidar_sub.subscribe(nh, "/lidar3D", 10);

        sync = std::make_shared<message_filters::Synchronizer<MySyncPolicy>>(MySyncPolicy(100), mask_sub, lidar_sub);
        sync->getPolicy()->setMaxIntervalDuration(ros::Duration(0.2));
        sync->registerCallback(boost::bind(&FixedROIProjector::lidarCallback, this, _1, _2));

        fixed_roi_pub = nh.advertise<sensor_msgs::PointCloud2>("/lidar_fixed_roi", 1);
        
        ROS_INFO("Multi-Target Fixed ROI Analysis Node Started");
        for(const auto& t : targets) ROS_INFO("Tracking Target ID: %.0f", t.id);
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

    void lidarCallback(const sensor_msgs::ImageConstPtr& mask_msg, const sensor_msgs::PointCloud2ConstPtr& lidar_msg) {
        geometry_msgs::TransformStamped transform;
        try {
            transform = tf_buffer.lookupTransform("Camera-3", "lidar_link", ros::Time(0));
        } catch (tf2::TransformException& ex) { return; }

        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::fromROSMsg(*lidar_msg, *cloud);

        pcl::PointCloud<pcl::PointXYZI> fixed_roi_cloud;
        Eigen::Affine3d T = tf2::transformToEigen(transform);

        // 매 프레임마다 모든 차량 카운트 초기화
        for(auto& t : targets) t.reset();

        for (const auto& pt : cloud->points) {
            if (pt.z < -2.0) continue;

            // 해당 포인트가 추적 대상 리스트에 있는지 확인
            for(auto& t : targets) {
                if (std::abs(pt.intensity - (t.id / 255.0f)) < 0.001f) {
                    t.p_actual++;
                    break;
                }
            }

            Eigen::Vector3d p_l(pt.x, pt.y, pt.z);
            Eigen::Vector3d p_c = T * p_l;
            double x = -p_c.y(), y = -p_c.z(), z = p_c.x();

            //if (z > 0.1 && z < 50.0) {
            if (z > 0.1) {
                int u = static_cast<int>((K.at<double>(0,0) * x / z) + K.at<double>(0,2));
                int v = static_cast<int>((K.at<double>(1,1) * y / z) + K.at<double>(1,2));

                // 고정된 ROI 영역(640x480) 필터링
                if (u >= 0 && u < img_w && v >= 0 && v < img_h) {
                    fixed_roi_cloud.push_back(pt);
                    
                    // ROI 내에 들어온 타겟 점 카운트
                    for(auto& t : targets) {
                        if (std::abs(pt.intensity - (t.id / 255.0f)) < 0.001f) {
                            t.p_preserved++;
                            t.dist_sum += std::sqrt(pt.x * pt.x + pt.y * pt.y);
                            break;
                        }
                    }
                }
            }
        }

        // --- [최종 로그 출력부] ---
        ROS_INFO("=========================================");
        size_t total_points = cloud->size();
        size_t projected_points = fixed_roi_cloud.size();
        double data_efficiency = (total_points > 0) ? (double)projected_points / total_points * 100.0 : 0.0;
        
        // 고정 ROI 방식의 전체 효율 로그 (왕자님이 주신 포맷 유지)
        //ROS_INFO("0) Raw LiDAR Total Points: %zu", total_points);
        //ROS_INFO("1) LiDAR ROI Points: %zu", projected_points);
        //ROS_INFO("2) Data Efficiency        : %.2f%% of Original", data_efficiency);
        // ROS_INFO("-----------------------------------------");

        // 각 차량별 루프 돌며 로그 출력 (동적 ROI와 완벽 일치)
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

        sensor_msgs::PointCloud2 ros_cloud;
        pcl::toROSMsg(fixed_roi_cloud, ros_cloud);
        ros_cloud.header = lidar_msg->header;
        fixed_roi_pub.publish(ros_cloud);
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "lidar_fixed_roi_node");
    FixedROIProjector frp;
    ros::spin();
    frp.printFinalReport();
    return 0;
}
*/

/*
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <message_filters/subscriber.h> 
#include <message_filters/synchronizer.h> 
#include <message_filters/sync_policies/approximate_time.h> 
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#include <tf2_eigen/tf2_eigen.h>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <memory>

class FixedROIProjector {
private:
    ros::NodeHandle nh;
    // ros::Subscriber lidar_sub; 
    ros::Publisher fixed_roi_pub;
    
    tf2_ros::Buffer tf_buffer;
    tf2_ros::TransformListener tf_listener;
    
    cv::Mat K;
    const int img_w = 640;
    const int img_h = 480;

    // 동적 ROI와 동일한 타겟 ID 설정
    float target_actor_id;
    const int N = 5; // 탐지 인정 최소 점 개수

    // [연속성 카운터]
    int f_total = 0;    
    int f_detected = 0; 

    // [동기화, 구독자 정의] 
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::PointCloud2> MySyncPolicy;
    std::shared_ptr<message_filters::Synchronizer<MySyncPolicy>> sync;
    message_filters::Subscriber<sensor_msgs::Image> mask_sub;
    message_filters::Subscriber<sensor_msgs::PointCloud2> lidar_sub;

public:
    FixedROIProjector() : tf_listener(tf_buffer) {
        ros::NodeHandle pnh("~"); // "~"를 넣어야 안방(Private)을 뒤집니다!
        pnh.param<float>("target_id", target_actor_id, 1.0f);   
        K = (cv::Mat_<double>(3, 3) << 320.0, 0.0, 320.0, 0.0, 320.0, 240.0, 0.0, 0.0, 1.0);

        // [동기화를 위해 동적 ROI 노드와 동일한 방식으로 토픽을 구독]
        mask_sub.subscribe(nh, "/yolop_fusion_masks", 10);
        lidar_sub.subscribe(nh, "/lidar3D", 10);

        // [시간 동기화 설정 (MaxInterval 0.2초로 동일하게 설정)]
        sync = std::make_shared<message_filters::Synchronizer<MySyncPolicy>>(MySyncPolicy(100), mask_sub, lidar_sub);
        sync->getPolicy()->setMaxIntervalDuration(ros::Duration(0.2));
        sync->registerCallback(boost::bind(&FixedROIProjector::lidarCallback, this, _1, _2));

        fixed_roi_pub = nh.advertise<sensor_msgs::PointCloud2>("/lidar_fixed_roi", 1);
        
        ROS_INFO("Target Actor ID is set to: %.0f", target_actor_id);
        ROS_INFO("FOV-based Fixed ROI Node Started!");
    }

    // [콜백 함수의 인자에 이미지 메시지를 추가(사용은 안 하지만 동기화를 위해 필요)]
    void lidarCallback(const sensor_msgs::ImageConstPtr& mask_msg, const sensor_msgs::PointCloud2ConstPtr& lidar_msg) {
        geometry_msgs::TransformStamped transform;
        try {
            transform = tf_buffer.lookupTransform("Camera-3", "lidar_link", ros::Time(0));
        } catch (tf2::TransformException& ex) {
            return;
        }

        // Actor ID를 읽기 위해 PointXYZI로 변환
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::fromROSMsg(*lidar_msg, *cloud);

        pcl::PointCloud<pcl::PointXYZI> fixed_roi_cloud;
        Eigen::Affine3d T = tf2::transformToEigen(transform);

        int P_actual = 0;    // 전체 타겟 점 개수
        int P_preserved = 0; // 고정 ROI(박스) 내 생존 점 개수
        double dist_sum = 0.0;  

        for (const auto& pt : cloud->points) {
            if (pt.z < -2.0) continue;

            bool is_target = (std::abs(pt.intensity - (target_actor_id / 255.0f)) < 0.001f);

            if (is_target) {
                P_actual++; 
            }

            Eigen::Vector3d p_l(pt.x, pt.y, pt.z);
            Eigen::Vector3d p_c = T * p_l;
            double x = -p_c.y(); double y = -p_c.z(); double z = p_c.x();

            if (z > 0.1 && z < 50.0) {
                int u = static_cast<int>((K.at<double>(0,0) * x / z) + K.at<double>(0,2));
                int v = static_cast<int>((K.at<double>(1,1) * y / z) + K.at<double>(1,2));

                // 마스킹 대신 '고정된 이미지 픽셀 범위'를 필터로 사용
                if (u >= 0 && u < img_w && v >= 0 && v < img_h) {
                    
                    fixed_roi_cloud.push_back(pt);
                    
                    if (is_target) {
                        P_preserved++;
                        dist_sum += std::sqrt(std::pow(pt.x, 2) + std::pow(pt.y, 2));
                    }
                }
            }
        }

        // [지표 계산]
        double opr = (P_actual > 0) ? (double)P_preserved / P_actual * 100.0 : 0.0;
        
        double reaction_dist = 0.0;
        if (P_preserved >= N) { 
            reaction_dist = dist_sum / P_preserved; 
        }

        if (P_actual > 0) f_total++; 
        if (P_preserved >= N) f_detected++; 
        double continuity = (f_total > 0) ? (double)f_detected / f_total * 100.0 : 0.0;

        size_t total_points = cloud->size();
        size_t projected_points = fixed_roi_cloud.size();
        double data_efficiency = (total_points > 0) ? (double)projected_points / total_points * 100.0 : 0.0;

        ROS_INFO("=========================================");
        //ROS_INFO("0) Raw LiDAR Total Points: %zu", total_points);
        //ROS_INFO("1) Total Projected Points: %zu", projected_points);
        //ROS_INFO("2) Data Efficiency    : %.2f%% of Original", data_efficiency);
        // Data Efficiency를 계산할 때 분모가 되는 Total Original Points는 "센서가 360도 전체에서 획득하려고 시도한 모든 점"을 의미 
        // -> 뒤쪽이 차체에 가려지더라도, 라이다 센서 자체는 일단 360도 전 구간에 대해 데이터를 생성하려고 시도
        // -> 360도 전체 스캔 데이터 중에서 전방 60도~ 70도 정도의 영역 => 전체의 약 16% ~ 20% => 물체가 온전히 다 보였을때 18.86% 수치는 적절
        ROS_INFO("-----------------------------------------");
        //ROS_INFO("Target Actor ID  : %.0f (Sur-2)", target_actor_id);
        ROS_INFO("P_actual (Total)   : %d", P_actual);
        ROS_INFO("P_preserved (ROI) : %d", P_preserved);
        ROS_INFO("OPR (Safety)     : %.2f%%", opr);
        
        if (reaction_dist > 0) {
            ROS_INFO("Detection Dist   : %.2f m", reaction_dist);
        } else {
            ROS_INFO("Detection Dist   : Not Detected");
        }

        //ROS_INFO("-----------------------------------------");
        //ROS_INFO("F_total : %d | F_detected : %d", f_total, f_detected);
        //ROS_INFO("Continuity Rate  : %.2f%%", continuity);
        //ROS_INFO("=========================================");

        sensor_msgs::PointCloud2 ros_cloud;
        pcl::toROSMsg(fixed_roi_cloud, ros_cloud);
        ros_cloud.header = lidar_msg->header;
        fixed_roi_pub.publish(ros_cloud);
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "lidar_fixed_roi_node");
    FixedROIProjector frp;
    ros::spin();
    return 0;
}
*/

/*
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <message_filters/subscriber.h> 
#include <message_filters/synchronizer.h> 
#include <message_filters/sync_policies/approximate_time.h> 
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#include <tf2_eigen/tf2_eigen.h>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <memory>

class FixedROIProjector {
private:
    ros::NodeHandle nh;
    // ros::Subscriber lidar_sub; 
    ros::Publisher fixed_roi_pub;
    
    tf2_ros::Buffer tf_buffer;
    tf2_ros::TransformListener tf_listener;
    
    cv::Mat K;
    const int img_w = 640;
    const int img_h = 480;

    // 동적 ROI와 동일한 타겟 ID 설정
    float target_actor_id;
    const int N = 5; // 탐지 인정 최소 점 개수

    // [연속성 카운터]
    int f_total = 0;    
    int f_detected = 0; 

    // [동기화, 구독자 정의] 
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::PointCloud2> MySyncPolicy;
    std::shared_ptr<message_filters::Synchronizer<MySyncPolicy>> sync;
    message_filters::Subscriber<sensor_msgs::Image> mask_sub;
    message_filters::Subscriber<sensor_msgs::PointCloud2> lidar_sub;

public:
    FixedROIProjector() : tf_listener(tf_buffer) {
        ros::NodeHandle pnh("~"); // "~"를 넣어야 안방(Private)을 뒤집니다!
        pnh.param<float>("target_id", target_actor_id, 1.0f);   
        K = (cv::Mat_<double>(3, 3) << 320.0, 0.0, 320.0, 0.0, 320.0, 240.0, 0.0, 0.0, 1.0);

        // [동기화를 위해 동적 ROI 노드와 동일한 방식으로 토픽을 구독]
        mask_sub.subscribe(nh, "/yolop_fusion_masks", 10);
        lidar_sub.subscribe(nh, "/lidar3D", 10);

        // [시간 동기화 설정 (MaxInterval 0.2초로 동일하게 설정)]
        sync = std::make_shared<message_filters::Synchronizer<MySyncPolicy>>(MySyncPolicy(100), mask_sub, lidar_sub);
        sync->getPolicy()->setMaxIntervalDuration(ros::Duration(0.2));
        sync->registerCallback(boost::bind(&FixedROIProjector::lidarCallback, this, _1, _2));

        fixed_roi_pub = nh.advertise<sensor_msgs::PointCloud2>("/lidar_fixed_roi", 1);
        
        ROS_INFO("Target Actor ID is set to: %.0f", target_actor_id);
        ROS_INFO("FOV-based Fixed ROI Node Started!");
    }

    // [콜백 함수의 인자에 이미지 메시지를 추가(사용은 안 하지만 동기화를 위해 필요)]
    void lidarCallback(const sensor_msgs::ImageConstPtr& mask_msg, const sensor_msgs::PointCloud2ConstPtr& lidar_msg) {
        geometry_msgs::TransformStamped transform;
        try {
            transform = tf_buffer.lookupTransform("Camera-3", "lidar_link", ros::Time(0));
        } catch (tf2::TransformException& ex) {
            return;
        }

        // Actor ID를 읽기 위해 PointXYZI로 변환
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::fromROSMsg(*lidar_msg, *cloud);

        pcl::PointCloud<pcl::PointXYZI> fixed_roi_cloud;
        Eigen::Affine3d T = tf2::transformToEigen(transform);

        int P_actual = 0;    // 전체 타겟 점 개수
        int P_preserved = 0; // 고정 ROI(박스) 내 생존 점 개수
        double dist_sum = 0.0;  

        for (const auto& pt : cloud->points) {
            if (pt.z < -2.0) continue;

            bool is_target = (std::abs(pt.intensity - (target_actor_id / 255.0f)) < 0.001f);

            Eigen::Vector3d p_l(pt.x, pt.y, pt.z);
            Eigen::Vector3d p_c = T * p_l;
            double x = -p_c.y(); double y = -p_c.z(); double z = p_c.x();

            if (is_target && z > 0.1 && z < 50.0) {
                P_actual++; // 카메라가 물리적으로 '바라보는 방향'에 있는 모든 타겟 점
            }

            if (z > 0.1 && z < 50.0) {
                int u = static_cast<int>((K.at<double>(0,0) * x / z) + K.at<double>(0,2));
                int v = static_cast<int>((K.at<double>(1,1) * y / z) + K.at<double>(1,2));

                // 마스킹 대신 '고정된 이미지 픽셀 범위'를 필터로 사용
                if (u >= 0 && u < img_w && v >= 0 && v < img_h) {
                    
                    fixed_roi_cloud.push_back(pt);
                    
                    if (is_target) {
                        P_preserved++;
                        dist_sum += std::sqrt(std::pow(pt.x, 2) + std::pow(pt.y, 2));
                    }
                }
            }
        }

        // [지표 계산]
        double opr = (P_actual > 0) ? (double)P_preserved / P_actual * 100.0 : 0.0;
        
        double reaction_dist = 0.0;
        if (P_preserved >= N) { 
            reaction_dist = dist_sum / P_preserved; 
        }

        if (P_actual > 0) f_total++; 
        if (P_preserved >= N) f_detected++; 
        double continuity = (f_total > 0) ? (double)f_detected / f_total * 100.0 : 0.0;

        size_t total_points = cloud->size();
        size_t projected_points = fixed_roi_cloud.size();
        double data_efficiency = (total_points > 0) ? (double)projected_points / total_points * 100.0 : 0.0;

        ROS_INFO("=========================================");
        ROS_INFO("0) Raw LiDAR Total Points: %zu", total_points);
        ROS_INFO("1) Total Projected Points: %zu", projected_points);
        ROS_INFO("2) Data Efficiency    : %.2f%% of Original", data_efficiency);
        // Data Efficiency를 계산할 때 분모가 되는 Total Original Points는 "센서가 360도 전체에서 획득하려고 시도한 모든 점"을 의미 
        // -> 뒤쪽이 차체에 가려지더라도, 라이다 센서 자체는 일단 360도 전 구간에 대해 데이터를 생성하려고 시도
        // -> 360도 전체 스캔 데이터 중에서 전방 60도~ 70도 정도의 영역 => 전체의 약 16% ~ 20% => 물체가 온전히 다 보였을때 18.86% 수치는 적절
        ROS_INFO("-----------------------------------------");
        //ROS_INFO("Target Actor ID  : %.0f (Sur-2)", target_actor_id);
        ROS_INFO("P_actual (Total)   : %d", P_actual);
        ROS_INFO("P_preserved (ROI) : %d", P_preserved);
        ROS_INFO("OPR (Safety)     : %.2f%%", opr);
        
        if (reaction_dist > 0) {
            ROS_INFO("Detection Dist   : %.2f m", reaction_dist);
        } else {
            ROS_INFO("Detection Dist   : Not Detected");
        }

        //ROS_INFO("-----------------------------------------");
        //ROS_INFO("F_total : %d | F_detected : %d", f_total, f_detected);
        //ROS_INFO("Continuity Rate  : %.2f%%", continuity);
        //ROS_INFO("=========================================");

        sensor_msgs::PointCloud2 ros_cloud;
        pcl::toROSMsg(fixed_roi_cloud, ros_cloud);
        ros_cloud.header = lidar_msg->header;
        fixed_roi_pub.publish(ros_cloud);
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "lidar_fixed_roi_node");
    FixedROIProjector frp;
    ros::spin();
    return 0;
}
*/

/*
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#include <tf2_eigen/tf2_eigen.h>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <memory>

class FixedROIProjector {
private:
    ros::NodeHandle nh;
    ros::Subscriber lidar_sub;
    ros::Publisher fixed_roi_pub;
    
    tf2_ros::Buffer tf_buffer;
    tf2_ros::TransformListener tf_listener;
    
    cv::Mat K;
    const int img_w = 640;
    const int img_h = 480;

    // 동적 ROI와 동일한 타겟 ID 설정
    //const float target_actor_id = 2.0f; // Sur-3 차량
    float target_actor_id;
    const int N = 5; // 탐지 인정 최소 점 개수

    // [연속성 카운터]
    int f_total = 0;    
    int f_detected = 0; 

public:
    FixedROIProjector() : tf_listener(tf_buffer) {
        //nh.param<float>("target_id", target_actor_id, 1.0f);
        ros::NodeHandle pnh("~"); // "~"를 넣어야 안방(Private)을 뒤집니다!
        pnh.param<float>("target_id", target_actor_id, 1.0f);   
        K = (cv::Mat_<double>(3, 3) << 320.0, 0.0, 320.0, 0.0, 320.0, 240.0, 0.0, 0.0, 1.0);

        lidar_sub = nh.subscribe("/lidar3D", 10, &FixedROIProjector::lidarCallback, this);
        fixed_roi_pub = nh.advertise<sensor_msgs::PointCloud2>("/lidar_fixed_roi", 1);
        
        ROS_INFO("Target Actor ID is set to: %.0f", target_actor_id);
        ROS_INFO("FOV-based Fixed ROI Node Started!");
    }

    void lidarCallback(const sensor_msgs::PointCloud2ConstPtr& lidar_msg) {
        geometry_msgs::TransformStamped transform;
        try {
            transform = tf_buffer.lookupTransform("Camera-3", "lidar_link", ros::Time(0));
        } catch (tf2::TransformException& ex) {
            return;
        }

        // Actor ID를 읽기 위해 PointXYZI로 변환
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::fromROSMsg(*lidar_msg, *cloud);

        pcl::PointCloud<pcl::PointXYZI> fixed_roi_cloud;
        Eigen::Affine3d T = tf2::transformToEigen(transform);

        int P_actual = 0;    // 전체 타겟 점 개수
        int P_preserved = 0; // 고정 ROI(박스) 내 생존 점 개수
        double dist_sum = 0.0;  

        for (const auto& pt : cloud->points) {
            if (pt.z < -2.0) continue;

            bool is_target = (std::abs(pt.intensity - (target_actor_id / 255.0f)) < 0.001f);

            Eigen::Vector3d p_l(pt.x, pt.y, pt.z);
            Eigen::Vector3d p_c = T * p_l;
            double x = -p_c.y(); double y = -p_c.z(); double z = p_c.x();

            if (is_target && z > 0.1 && z < 50.0) {
                P_actual++; // 카메라가 물리적으로 '바라보는 방향'에 있는 모든 타겟 점
            }

            if (z > 0.1 && z < 50.0) {
                int u = static_cast<int>((K.at<double>(0,0) * x / z) + K.at<double>(0,2));
                int v = static_cast<int>((K.at<double>(1,1) * y / z) + K.at<double>(1,2));

                // 마스킹 대신 '고정된 이미지 픽셀 범위'를 필터로 사용
                if (u >= 0 && u < img_w && v >= 0 && v < img_h) {
                    
                    fixed_roi_cloud.push_back(pt);
                    
                    if (is_target) {
                        P_preserved++;
                        dist_sum += std::sqrt(std::pow(pt.x, 2) + std::pow(pt.y, 2));
                    }
                }
            }
        }

        // [지표 계산]
        double opr = (P_actual > 0) ? (double)P_preserved / P_actual * 100.0 : 0.0;
        
        double reaction_dist = 0.0;
        if (P_preserved >= N) { 
            reaction_dist = dist_sum / P_preserved; 
        }

        if (P_actual > 0) f_total++; 
        if (P_preserved >= N) f_detected++; 
        double continuity = (f_total > 0) ? (double)f_detected / f_total * 100.0 : 0.0;

        size_t total_points = cloud->size();
        size_t projected_points = fixed_roi_cloud.size();
        double data_efficiency = (total_points > 0) ? (double)projected_points / total_points * 100.0 : 0.0;

        ROS_INFO("=========================================");
        //ROS_INFO("0) Raw LiDAR Total Points: %zu", total_points);
        //ROS_INFO("1) Total Projected Points: %zu", projected_points);
        //ROS_INFO("2) Data Efficiency    : %.2f%% of Original", data_efficiency);
        // Data Efficiency를 계산할 때 분모가 되는 Total Original Points는 "센서가 360도 전체에서 획득하려고 시도한 모든 점"을 의미 
        // -> 뒤쪽이 차체에 가려지더라도, 라이다 센서 자체는 일단 360도 전 구간에 대해 데이터를 생성하려고 시도
        // -> 360도 전체 스캔 데이터 중에서 전방 60도~ 70도 정도의 영역 => 전체의 약 16% ~ 20% => 물체가 온전히 다 보였을때 18.86% 수치는 적절
        //ROS_INFO("-----------------------------------------");
        //ROS_INFO("Target Actor ID  : %.0f (Sur-2)", target_actor_id);
        ROS_INFO("P_actual (Total)   : %d", P_actual);
        ROS_INFO("P_preserved (ROI) : %d", P_preserved);
        ROS_INFO("OPR (Safety)     : %.2f%%", opr);
        
        if (reaction_dist > 0) {
            ROS_INFO("Detection Dist   : %.2f m", reaction_dist);
        } else {
            ROS_INFO("Detection Dist   : Not Detected");
        }

        //ROS_INFO("-----------------------------------------");
        //ROS_INFO("F_total : %d | F_detected : %d", f_total, f_detected);
        //ROS_INFO("Continuity Rate  : %.2f%%", continuity);
        //ROS_INFO("=========================================");

        sensor_msgs::PointCloud2 ros_cloud;
        pcl::toROSMsg(fixed_roi_cloud, ros_cloud);
        ros_cloud.header = lidar_msg->header;
        fixed_roi_pub.publish(ros_cloud);
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "lidar_fixed_roi_node");
    FixedROIProjector frp;
    ros::spin();
    return 0;
}
*/

//카메라 화각 범위 고정 roi
/*
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#include <tf2_eigen/tf2_eigen.h>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

class FixedROIProjector {
private:
    ros::NodeHandle nh;
    ros::Subscriber lidar_sub;
    ros::Publisher fixed_roi_pub;
    
    tf2_ros::Buffer tf_buffer;
    tf2_ros::TransformListener tf_listener;
    
    cv::Mat K; // 카메라 내판 행렬 (Intrinsic Matrix)
    const int img_w = 640; // 모라이 카메라 가로 해상도
    const int img_h = 480; // 모라이 카메라 세로 해상도

public:
    FixedROIProjector() : tf_listener(tf_buffer) {
        // 1. 카메라 매트릭스 설정 
        // K = [fx 0 cx; 0 fy cy; 0 0 1]
        K = (cv::Mat_<double>(3, 3) << 320.0, 0.0, 320.0, 0.0, 320.0, 240.0, 0.0, 0.0, 1.0);

        // 2. 토픽 구독 및 발행 설정
        lidar_sub = nh.subscribe("/lidar3D", 10, &FixedROIProjector::lidarCallback, this);
        fixed_roi_pub = nh.advertise<sensor_msgs::PointCloud2>("/lidar_fixed_roi", 1);
        
        ROS_INFO("FOV-based Fixed ROI Node Started!");
    }

    void lidarCallback(const sensor_msgs::PointCloud2ConstPtr& lidar_msg) {
        // 3. 카메라와 라이다 간의 상대 위치(TF) 조회
        geometry_msgs::TransformStamped transform;
        try {
            // 모라이 시뮬레이터의 Frame ID를 확인하세요!
            transform = tf_buffer.lookupTransform("Camera-3", "lidar_link", ros::Time(0));
        } catch (tf2::TransformException& ex) {
            ROS_WARN("TF Lookup Failed: %s", ex.what());
            return;
        }

        // 4. ROS 메시지를 PCL 데이터로 변환
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*lidar_msg, *cloud);

        pcl::PointCloud<pcl::PointXYZ> fixed_roi_cloud;
        Eigen::Affine3d T = tf2::transformToEigen(transform);
        int count_fixed = 0;

        for (const auto& pt : cloud->points) {
            // 5. 라이다 좌표(Lidar Frame) -> 카메라 좌표(Camera Frame) 변환
            Eigen::Vector3d p_l(pt.x, pt.y, pt.z);
            Eigen::Vector3d p_c = T * p_l;

            // 6. OpenCV 좌표계로 축 변환 (ROS x-forward -> OpenCV z-forward)
            double x = -p_c.y();
            double y = -p_c.z();
            double z = p_c.x();

            // 7. 전방 영역 및 거리 필터링 (0.1m ~ 50m)
            if (z > 0.1 && z < 50.0) {
                // 8. 핀홀 카메라 모델 투영 수식 적용
                // $u = \frac{f_x \cdot x}{z} + c_x$
                // $v = \frac{f_y \cdot y}{z} + c_y$
                int u = static_cast<int>((K.at<double>(0,0) * x / z) + K.at<double>(0,2));
                int v = static_cast<int>((K.at<double>(1,1) * y / z) + K.at<double>(1,2));

                // 9. [핵심] 이미지 픽셀 범위(FOV) 내부에 있는지 확인
                if (u >= 0 && u < img_w && v >= 0 && v < img_h) {
                    fixed_roi_cloud.push_back(pt);
                    count_fixed++;
                }
            }
        }

        // 결과 출력
        ROS_INFO("[FOV ROI] Total Points in View: %d", count_fixed);

        // 10. RViz 시각화를 위해 필터링된 점들을 다시 발행
        sensor_msgs::PointCloud2 ros_cloud;
        pcl::toROSMsg(fixed_roi_cloud, ros_cloud);
        ros_cloud.header = lidar_msg->header;
        fixed_roi_pub.publish(ros_cloud);
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "lidar_fixed_roi_node");
    FixedROIProjector frp;
    ros::spin();
    return 0;
}



//가로세로높이 고정 범위 roi
/*
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

class FixedROIProjector {
private:
    ros::NodeHandle nh;
    ros::Subscriber lidar_sub;
    ros::Publisher fixed_roi_pub;

    //[고정 ROI 범위]
    double min_x = 0.0,  max_x = 30.0;
    double min_y = -4.0, max_y = 4.0;
    //double min_z = -1.5, max_z = 1.0;

public:
    FixedROIProjector() {
        // 1. 마스크 없이 라이다 토픽만 직접 구독합니다.
        lidar_sub = nh.subscribe("/lidar3D", 10, &FixedROIProjector::lidarCallback, this);
        fixed_roi_pub = nh.advertise<sensor_msgs::PointCloud2>("/lidar_fixed_roi", 1);
        
        ROS_INFO("🚀 Independent Fixed ROI Node Started! (No Mask Needed)");
    }

    void lidarCallback(const sensor_msgs::PointCloud2ConstPtr& lidar_msg) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*lidar_msg, *cloud);

        pcl::PointCloud<pcl::PointXYZ> fixed_roi_cloud;
        int count_fixed = 0;

        for (const auto& pt : cloud->points) {
            // 2. 오로지 가로, 세로, 높이 범위만 체크합니다.
            if (pt.x >= min_x && pt.x <= max_x && 
                pt.y >= min_y && pt.y <= max_y && 
                pt.z >= min_z && pt.z <= max_z) {
                
                fixed_roi_cloud.push_back(pt);
                count_fixed++;
            }
        }

        // 점 개수 출력
        ROS_INFO("[Fixed ROI] Point Count: %d", count_fixed);

        // RViz 가시화를 위한 토픽 발행
        sensor_msgs::PointCloud2 ros_cloud;
        pcl::toROSMsg(fixed_roi_cloud, ros_cloud);
        ros_cloud.header = lidar_msg->header;
        fixed_roi_pub.publish(ros_cloud);
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "lidar_fixed_roi_node");
    FixedROIProjector frp;
    ros::spin();
    return 0;
}
*/