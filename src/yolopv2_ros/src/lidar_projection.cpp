//OPR, OPPR 계산 제일 최근 코드
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <morai_msgs/ObjectStatusList.h>
#include <cv_bridge/cv_bridge.h>
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
#include <memory>
#include <vector>
#include <cmath>

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

class PerfectProposedROIProjector {
private:
    ros::NodeHandle nh;
    ros::Subscriber obj_sub;
    ros::Publisher road_pub, obs_pub;
    tf2_ros::Buffer tf_buffer;
    tf2_ros::TransformListener tf_listener;

    std::vector<TargetVehicle> targets;

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::PointCloud2> MySyncPolicy;
    std::shared_ptr<message_filters::Synchronizer<MySyncPolicy>> sync;
    message_filters::Subscriber<sensor_msgs::Image> mask_sub;
    message_filters::Subscriber<sensor_msgs::PointCloud2> lidar_sub;

public:
    PerfectProposedROIProjector() : tf_listener(tf_buffer) {
        ros::NodeHandle pnh("~");
        int t_id;
        pnh.param<int>("target_id_1", t_id, 2); 
        targets.push_back({t_id});

        obj_sub = nh.subscribe("/Object_topic", 10, &PerfectProposedROIProjector::objectCallback, this);
        mask_sub.subscribe(nh, "/yolop_fusion_masks", 10);
        lidar_sub.subscribe(nh, "/lidar3D", 10);

        sync = std::make_shared<message_filters::Synchronizer<MySyncPolicy>>(MySyncPolicy(100), mask_sub, lidar_sub);
        sync->registerCallback(boost::bind(&PerfectProposedROIProjector::callback, this, _1, _2));

        road_pub = nh.advertise<sensor_msgs::PointCloud2>("/lidar_road", 1);
        obs_pub = nh.advertise<sensor_msgs::PointCloud2>("/lidar_obstacle", 1);
        
        ROS_INFO("Final Proposed-ROI Analysis Node Started");
    }

    ~PerfectProposedROIProjector() {
        if (sync) sync.reset();
        ROS_INFO("the node has safely shut down");
    }

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

    void callback(const sensor_msgs::ImageConstPtr& mask_msg, const sensor_msgs::PointCloud2ConstPtr& lidar_msg) {
        bool all_updated = true;
        for(const auto& t : targets) if(!t.is_updated) all_updated = false;
        if (!all_updated) return;

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

        for(auto& t : targets) t.reset();
        int n_roi_total = 0;

        for (const auto& pt : cloud->points) {
            for(auto& t : targets) if (t.isInside(pt.x, pt.y, pt.z)) t.n_obj_full++;

            Eigen::Vector3d p_l(pt.x, pt.y, pt.z);
            Eigen::Vector3d p_c = T * p_l;
            double xc = -p_c.y(), yc = -p_c.z(), zc = p_c.x();

            if (zc > 0.1) {
                int u = static_cast<int>((320.0 * xc / zc) + 320.0);
                int v = static_cast<int>((320.0 * yc / zc) + 240.0);

                if (u >= 0 && u < 640 && v >= 0 && v < 480) {
                    cv::Vec3b pixel = fusion_mask.at<cv::Vec3b>(v, u);
                    bool is_in_roi = false;

                    if (pixel[1] > 127) { // Obstacle
                        obs_out.push_back(pcl::PointXYZ(pt.x, pt.y, pt.z));
                        is_in_roi = true;
                    } else if (pixel[0] > 127) { // Road
                        road_out.push_back(pcl::PointXYZ(pt.x, pt.y, pt.z));
                        is_in_roi = true;
                    }

                    if (is_in_roi) {
                        n_roi_total++;
                        for(auto& t : targets) if (t.isInside(pt.x, pt.y, pt.z)) t.n_obj_roi++;
                    }
                }
            }
        }

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

        sensor_msgs::PointCloud2 r_msg, o_msg;
        pcl::toROSMsg(road_out, r_msg); r_msg.header = lidar_msg->header;
        pcl::toROSMsg(obs_out, o_msg); o_msg.header = lidar_msg->header;
        road_pub.publish(r_msg); obs_pub.publish(o_msg);
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "proposed_roi_analyzer");
    PerfectProposedROIProjector prp;
    ros::spin();
    return 0;
}


/*
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
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#include <tf2_eigen/tf2_eigen.h>
#include <Eigen/Dense>
#include <memory>
#include <vector>
#include <cmath>

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

class LidarProjector {
private:
    ros::NodeHandle nh;
    ros::Publisher road_pub, obs_pub;
    tf2_ros::Buffer tf_buffer;
    tf2_ros::TransformListener tf_listener;
    cv::Mat K;

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::PointCloud2> MySyncPolicy;
    std::shared_ptr<message_filters::Synchronizer<MySyncPolicy>> sync;
    message_filters::Subscriber<sensor_msgs::Image> mask_sub;
    message_filters::Subscriber<sensor_msgs::PointCloud2> lidar_sub;

    // 추적 대상 차량 리스트
    std::vector<TargetVehicle> targets;
    const int N = 5;

public:
    LidarProjector() : tf_listener(tf_buffer) {
        ros::NodeHandle pnh("~"); 
        
        // 두 대의 차량 ID를 파라미터로 설정 (기본값 1.0, 2.0)
        float id1, id2;
        pnh.param<float>("target_id_1", id1, 1.0f);
        pnh.param<float>("target_id_2", id2, 2.0f);
        
        targets.push_back({id1});
        targets.push_back({id2});

        K = (cv::Mat_<double>(3, 3) << 320.0, 0.0, 320.0, 0.0, 320.0, 240.0, 0.0, 0.0, 1.0);

        road_pub = nh.advertise<sensor_msgs::PointCloud2>("/lidar_road", 1);
        obs_pub = nh.advertise<sensor_msgs::PointCloud2>("/lidar_obstacle", 1);

        mask_sub.subscribe(nh, "/yolop_fusion_masks", 10);
        lidar_sub.subscribe(nh, "/lidar3D", 10);

        sync = std::make_shared<message_filters::Synchronizer<MySyncPolicy>>(MySyncPolicy(100), mask_sub, lidar_sub);
        sync->getPolicy()->setMaxIntervalDuration(ros::Duration(0.2));
        sync->registerCallback(boost::bind(&LidarProjector::callback, this, _1, _2));
        
        ROS_INFO("Multi-Target Dynamic ROI Analysis Node Started");
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

    void publishCloud(ros::Publisher& pub, const pcl::PointCloud<pcl::PointXYZ>& pcl_cloud, const std_msgs::Header& header) {
        if (pcl_cloud.empty()) return;
        sensor_msgs::PointCloud2 ros_cloud;
        pcl::toROSMsg(pcl_cloud, ros_cloud);
        ros_cloud.header = header;
        pub.publish(ros_cloud);
    }

    void callback(const sensor_msgs::ImageConstPtr& mask_msg, const sensor_msgs::PointCloud2ConstPtr& lidar_msg) {
        cv::Mat fusion_mask;
        try {
            fusion_mask = cv_bridge::toCvShare(mask_msg, "bgr8")->image;
        } catch (cv_bridge::Exception& e) { return; }

        geometry_msgs::TransformStamped transform;
        try {
            transform = tf_buffer.lookupTransform("Camera-3", "lidar_link", ros::Time(0));
        } catch (tf2::TransformException& ex) { return; }

        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::fromROSMsg(*lidar_msg, *cloud);

        pcl::PointCloud<pcl::PointXYZ> road_cloud, obs_cloud;
        Eigen::Affine3d T = tf2::transformToEigen(transform);

        // [중요] 매 프레임마다 모든 차량의 카운트 초기화
        for(auto& t : targets) t.reset();

        for (const auto& pt : cloud->points) {
            if (pt.z < -2.0) continue;

            // 해당 포인트가 추적 대상 중 누구인지 확인
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

                if (u >= 0 && u < fusion_mask.cols && v >= 0 && v < fusion_mask.rows) {
                    cv::Vec3b pixel = fusion_mask.at<cv::Vec3b>(v, u);
                    pcl::PointXYZ out_pt(pt.x, pt.y, pt.z);

                    if (pixel[1] > 127) { // Obstacle
                        obs_cloud.push_back(out_pt);
                        // ROI 내부 점이 추적 대상인지 확인하여 p_preserved 업데이트
                        for(auto& t : targets) {
                            if (std::abs(pt.intensity - (t.id / 255.0f)) < 0.001f) {
                                t.p_preserved++;
                                t.dist_sum += std::sqrt(pt.x * pt.x + pt.y * pt.y);
                                break;
                            }
                        }
                    } else if (pixel[0] > 127) { // Road
                        road_cloud.push_back(out_pt);
                    }
                }
            }
        }

        // --- [최종 로그 출력부] ---
        ROS_INFO("=========================================");
        size_t total_points = cloud->size();
        size_t projected_points = obs_cloud.size() + road_cloud.size();
        double data_efficiency = (total_points > 0) ? (double)projected_points / total_points * 100.0 : 0.0;
        
        //ROS_INFO("0) Raw LiDAR Total Points: %zu", total_points);
        //ROS_INFO("1) LiDAR ROI Points: %zu", projected_points);
        //ROS_INFO("2) Data Efficiency        : %.2f%% of Original", data_efficiency);
        //ROS_INFO("-----------------------------------------");

        // 등록된 모든 차량에 대해 루프를 돌며 로그 출력
        for(size_t i = 0; i < targets.size(); ++i) {
            auto& t = targets[i];
            double opr = (t.p_actual > 0) ? (double)t.p_preserved / t.p_actual * 100.0 : 0.0;
            double reaction_dist = (t.p_preserved >= N) ? t.dist_sum / t.p_preserved : 0.0;

            //ROS_INFO("[Vehicle %zu - Target ID: %.0f]", i + 1, t.id);
            //ROS_INFO(" P_actual (Total)   : %d", t.p_actual);
            //ROS_INFO(" P_preserved (ROI) : %d", t.p_preserved);
            //ROS_INFO(" OPR (Safety)     : %.2f%%", opr);

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

        publishCloud(road_pub, road_cloud, lidar_msg->header);
        publishCloud(obs_pub, obs_cloud, lidar_msg->header);
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "lidar_projection");
    LidarProjector lp;
    ros::spin();
    lp.printFinalReport();
    return 0;
}
*/

//여러대 차량 추적 가능
/*
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
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#include <tf2_eigen/tf2_eigen.h>
#include <Eigen/Dense>
#include <memory>
#include <vector>
#include <cmath>

// [차량별 데이터를 개별 관리하기 위한 구조체]
struct TargetVehicle {
    float id;           // Intensity 기반 ID
    int p_actual = 0;   // 전체 점 개수
    int p_preserved = 0;// ROI 내 보존된 점 개수
    double dist_sum = 0.0; // 탐지 거리 합계

    void reset() {
        p_actual = 0;
        p_preserved = 0;
        dist_sum = 0.0;
    }
};

class LidarProjector {
private:
    ros::NodeHandle nh;
    ros::Publisher road_pub, obs_pub;
    tf2_ros::Buffer tf_buffer;
    tf2_ros::TransformListener tf_listener;
    cv::Mat K;

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::PointCloud2> MySyncPolicy;
    std::shared_ptr<message_filters::Synchronizer<MySyncPolicy>> sync;
    message_filters::Subscriber<sensor_msgs::Image> mask_sub;
    message_filters::Subscriber<sensor_msgs::PointCloud2> lidar_sub;

    // 추적 대상 차량 리스트
    std::vector<TargetVehicle> targets;
    const int N = 5; 

public:
    LidarProjector() : tf_listener(tf_buffer) {
        ros::NodeHandle pnh("~"); 
        
        // 두 대의 차량 ID를 파라미터로 설정 (기본값 1.0, 2.0)
        float id1, id2;
        pnh.param<float>("target_id_1", id1, 1.0f);
        pnh.param<float>("target_id_2", id2, 2.0f);
        
        targets.push_back({id1});
        targets.push_back({id2});

        K = (cv::Mat_<double>(3, 3) << 320.0, 0.0, 320.0, 0.0, 320.0, 240.0, 0.0, 0.0, 1.0);

        road_pub = nh.advertise<sensor_msgs::PointCloud2>("/lidar_road", 1);
        obs_pub = nh.advertise<sensor_msgs::PointCloud2>("/lidar_obstacle", 1);

        mask_sub.subscribe(nh, "/yolop_fusion_masks", 10);
        lidar_sub.subscribe(nh, "/lidar3D", 10);

        sync = std::make_shared<message_filters::Synchronizer<MySyncPolicy>>(MySyncPolicy(100), mask_sub, lidar_sub);
        sync->getPolicy()->setMaxIntervalDuration(ros::Duration(0.2));
        sync->registerCallback(boost::bind(&LidarProjector::callback, this, _1, _2));
        
        ROS_INFO("Multi-Target Dynamic ROI Analysis Node Started");
    }

    void publishCloud(ros::Publisher& pub, const pcl::PointCloud<pcl::PointXYZ>& pcl_cloud, const std_msgs::Header& header) {
        if (pcl_cloud.empty()) return;
        sensor_msgs::PointCloud2 ros_cloud;
        pcl::toROSMsg(pcl_cloud, ros_cloud);
        ros_cloud.header = header;
        pub.publish(ros_cloud);
    }

    void callback(const sensor_msgs::ImageConstPtr& mask_msg, const sensor_msgs::PointCloud2ConstPtr& lidar_msg) {
        cv::Mat fusion_mask;
        try {
            fusion_mask = cv_bridge::toCvShare(mask_msg, "bgr8")->image;
        } catch (cv_bridge::Exception& e) { return; }

        geometry_msgs::TransformStamped transform;
        try {
            transform = tf_buffer.lookupTransform("Camera-3", "lidar_link", ros::Time(0));
        } catch (tf2::TransformException& ex) { return; }

        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::fromROSMsg(*lidar_msg, *cloud);

        pcl::PointCloud<pcl::PointXYZ> road_cloud, obs_cloud;
        Eigen::Affine3d T = tf2::transformToEigen(transform);

        // [중요] 매 프레임마다 모든 차량의 카운트 초기화
        for(auto& t : targets) t.reset();

        for (const auto& pt : cloud->points) {
            if (pt.z < -2.0) continue;

            // 해당 포인트가 추적 대상 중 누구인지 확인
            for(auto& t : targets) {
                if (std::abs(pt.intensity - (t.id / 255.0f)) < 0.001f) {
                    t.p_actual++;
                    break; 
                }
            }

            Eigen::Vector3d p_l(pt.x, pt.y, pt.z);
            Eigen::Vector3d p_c = T * p_l;
            double x = -p_c.y(), y = -p_c.z(), z = p_c.x();

            if (z > 0.1 && z < 50.0) {
                int u = static_cast<int>((K.at<double>(0,0) * x / z) + K.at<double>(0,2));
                int v = static_cast<int>((K.at<double>(1,1) * y / z) + K.at<double>(1,2));

                if (u >= 0 && u < fusion_mask.cols && v >= 0 && v < fusion_mask.rows) {
                    cv::Vec3b pixel = fusion_mask.at<cv::Vec3b>(v, u);
                    pcl::PointXYZ out_pt(pt.x, pt.y, pt.z);

                    if (pixel[1] > 127) { // Obstacle
                        obs_cloud.push_back(out_pt);
                        // ROI 내부 점이 추적 대상인지 확인하여 p_preserved 업데이트
                        for(auto& t : targets) {
                            if (std::abs(pt.intensity - (t.id / 255.0f)) < 0.001f) {
                                t.p_preserved++;
                                t.dist_sum += std::sqrt(pt.x * pt.x + pt.y * pt.y);
                                break;
                            }
                        }
                    } else if (pixel[0] > 127) { // Road
                        road_cloud.push_back(out_pt);
                    }
                }
            }
        }

        // --- [최종 로그 출력부] ---
        ROS_INFO("=========================================");
        size_t total_points = cloud->size();
        size_t projected_points = obs_cloud.size() + road_cloud.size();
        double data_efficiency = (total_points > 0) ? (double)projected_points / total_points * 100.0 : 0.0;
        
        //ROS_INFO("0) Raw LiDAR Total Points: %zu", total_points);
        //ROS_INFO("1) LiDAR ROI Points: %zu", projected_points);
        //ROS_INFO("2) Data Efficiency        : %.2f%% of Original", data_efficiency);
        //ROS_INFO("-----------------------------------------");

        // 등록된 모든 차량에 대해 루프를 돌며 로그 출력
        for(size_t i = 0; i < targets.size(); ++i) {
            auto& t = targets[i];
            double opr = (t.p_actual > 0) ? (double)t.p_preserved / t.p_actual * 100.0 : 0.0;
            double reaction_dist = (t.p_preserved >= N) ? t.dist_sum / t.p_preserved : 0.0;

            //ROS_INFO("[Vehicle %zu - Target ID: %.0f]", i + 1, t.id);
            //ROS_INFO(" P_actual (Total)   : %d", t.p_actual);
            //ROS_INFO(" P_preserved (ROI) : %d", t.p_preserved);
            //ROS_INFO(" OPR (Safety)     : %.2f%%", opr);
            
            if (reaction_dist > 0) ROS_INFO(" Detection Dist   : %.2f m", reaction_dist);
            else ROS_INFO(" Detection Dist   : Not Detected");

            //if (i < targets.size() - 1) ROS_INFO("-----------------------------------------");
        }
        ROS_INFO("=========================================");

        publishCloud(road_pub, road_cloud, lidar_msg->header);
        publishCloud(obs_pub, obs_cloud, lidar_msg->header);
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "lidar_projection");
    LidarProjector lp;
    ros::spin();
    return 0;
}
*/
/*
//점개수+OPR(객체 보존율)+객체탐지거리
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
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#include <tf2_eigen/tf2_eigen.h>
#include <Eigen/Dense>
#include <memory>
#include <cmath> 

class LidarProjector {
private:
    ros::NodeHandle nh;
    ros::Publisher road_pub, obs_pub;
    tf2_ros::Buffer tf_buffer;
    tf2_ros::TransformListener tf_listener;
    cv::Mat K;

    // [동기화, 구독자 정의] 
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::PointCloud2> MySyncPolicy;
    std::shared_ptr<message_filters::Synchronizer<MySyncPolicy>> sync;
    message_filters::Subscriber<sensor_msgs::Image> mask_sub;
    message_filters::Subscriber<sensor_msgs::PointCloud2> lidar_sub;

    float target_actor_id;
    const int N = 5; // 탐지 인정을 위한 최소 점 개수
    int f_total = 0;    
    int f_detected = 0;

public:
        LidarProjector() : tf_listener(tf_buffer) {
        ros::NodeHandle pnh("~"); 
        pnh.param<float>("target_id", target_actor_id, 1.0f);   

        K = (cv::Mat_<double>(3, 3) << 320.0, 0.0, 320.0, 0.0, 320.0, 240.0, 0.0, 0.0, 1.0);

        road_pub = nh.advertise<sensor_msgs::PointCloud2>("/lidar_road", 1);
        obs_pub = nh.advertise<sensor_msgs::PointCloud2>("/lidar_obstacle", 1);

        mask_sub.subscribe(nh, "/yolop_fusion_masks", 10);
        lidar_sub.subscribe(nh, "/lidar3D", 10);

        sync = std::make_shared<message_filters::Synchronizer<MySyncPolicy>>(MySyncPolicy(100), mask_sub, lidar_sub);
        sync->getPolicy()->setMaxIntervalDuration(ros::Duration(0.2));
        sync->registerCallback(boost::bind(&LidarProjector::callback, this, _1, _2));
        
        ROS_INFO("Integrated OPR & Distance Node");
        ROS_INFO("Target Actor ID is set to: %.0f", target_actor_id);
    }

    void publishCloud(ros::Publisher& pub, const pcl::PointCloud<pcl::PointXYZ>& pcl_cloud, const std_msgs::Header& header) {
        if (pcl_cloud.empty()) return;
        sensor_msgs::PointCloud2 ros_cloud;
        pcl::toROSMsg(pcl_cloud, ros_cloud);
        ros_cloud.header = header;
        pub.publish(ros_cloud);
    }

    void callback(const sensor_msgs::ImageConstPtr& mask_msg, const sensor_msgs::PointCloud2ConstPtr& lidar_msg) {
        cv::Mat fusion_mask;
        try {
            fusion_mask = cv_bridge::toCvShare(mask_msg, "bgr8")->image;
        } catch (cv_bridge::Exception& e) { return; }

        geometry_msgs::TransformStamped transform;
        try {
            transform = tf_buffer.lookupTransform("Camera-3", "lidar_link", ros::Time(0));
        } catch (tf2::TransformException& ex) { return; }

        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::fromROSMsg(*lidar_msg, *cloud);

        pcl::PointCloud<pcl::PointXYZ> road_cloud, obs_cloud;
        Eigen::Affine3d T = tf2::transformToEigen(transform);

        int P_actual = 0;
        int P_preserved = 0;
        double dist_sum = 0.0; 

        for (const auto& pt : cloud->points) {
            if (pt.z < -2.0) continue;

            bool is_target = (std::abs(pt.intensity - (target_actor_id / 255.0f)) < 0.001f);

            if (is_target) {
                P_actual++; 
            }

            Eigen::Vector3d p_l(pt.x, pt.y, pt.z);
            Eigen::Vector3d p_c = T * p_l;

            double x = -p_c.y();
            double y = -p_c.z();
            double z = p_c.x();

            if (z > 0.1 && z < 50.0) {
                int u = static_cast<int>((K.at<double>(0,0) * x / z) + K.at<double>(0,2));
                int v = static_cast<int>((K.at<double>(1,1) * y / z) + K.at<double>(1,2));

                if (u >= 0 && u < fusion_mask.cols && v >= 0 && v < fusion_mask.rows) {
                    cv::Vec3b pixel = fusion_mask.at<cv::Vec3b>(v, u);
                
                    pcl::PointXYZ out_pt; 
                    out_pt.x = pt.x; out_pt.y = pt.y; out_pt.z = pt.z;

                    if (pixel[1] > 127) { 
                        obs_cloud.push_back(out_pt);

                        if (is_target) {
                            P_preserved++;
                            dist_sum += std::sqrt(std::pow(pt.x, 2) + std::pow(pt.y, 2)); // d = sqrt((x2-x1)^2 + (y2-y1)^2) 수식 적용
                        }
                    } 
                    else if (pixel[0] > 127) { 
                        road_cloud.push_back(out_pt);
                    }
                }
            }
        }

        double opr = (P_actual > 0) ? (double)P_preserved / P_actual * 100.0 : 0.0;
        double reaction_dist = (P_preserved >= N) ? dist_sum / P_preserved : 0.0;

        // 연속성 계산
        if (P_actual > 0) f_total++; 
        if (P_preserved >= N) f_detected++; 
        double continuity = (f_total > 0) ? (double)f_detected / f_total * 100.0 : 0.0;

        // 데이터 효율성 계산
        size_t total_points = cloud->size();
        size_t projected_points = obs_cloud.size() + road_cloud.size();
        double data_efficiency = (total_points > 0) ? (double)projected_points / total_points * 100.0 : 0.0;

        // [최종 로그 출력]
        ROS_INFO("=========================================");
        //ROS_INFO("0) Raw LiDAR Total Points: %zu", total_points);
        //ROS_INFO("1) LiDAR Obstacle Points: %zu", obs_cloud.size());
        //ROS_INFO("2) LiDAR Road Points    : %zu", road_cloud.size());
        ROS_INFO("3) Total Projected Points : %zu", projected_points);
        ROS_INFO("4) Data Efficiency    : %.2f%% of Original", data_efficiency);
        ROS_INFO("-----------------------------------------");
        //ROS_INFO("Target Actor ID  : %.0f (Sur-3)", target_actor_id);
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

        publishCloud(road_pub, road_cloud, lidar_msg->header);
        publishCloud(obs_pub, obs_cloud, lidar_msg->header);
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "lidar_projection");
    LidarProjector lp;
    ros::spin();
    return 0;
}
*/

/*
//점개수+OPR(객체 보존율)+객체탐지거리
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
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#include <tf2_eigen/tf2_eigen.h>
#include <Eigen/Dense>
#include <memory>
#include <cmath> 

class LidarProjector {
private:
    ros::NodeHandle nh;
    ros::Publisher road_pub, obs_pub;
    tf2_ros::Buffer tf_buffer;
    tf2_ros::TransformListener tf_listener;
    cv::Mat K;

    // [동기화, 구독자 정의] 
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::PointCloud2> MySyncPolicy;
    std::shared_ptr<message_filters::Synchronizer<MySyncPolicy>> sync;
    message_filters::Subscriber<sensor_msgs::Image> mask_sub;
    message_filters::Subscriber<sensor_msgs::PointCloud2> lidar_sub;

    float target_actor_id;
    const int N = 5; // 탐지 인정을 위한 최소 점 개수
    int f_total = 0;    
    int f_detected = 0;

public:
        LidarProjector() : tf_listener(tf_buffer) {
        ros::NodeHandle pnh("~"); 
        pnh.param<float>("target_id", target_actor_id, 1.0f);   

        K = (cv::Mat_<double>(3, 3) << 320.0, 0.0, 320.0, 0.0, 320.0, 240.0, 0.0, 0.0, 1.0);

        road_pub = nh.advertise<sensor_msgs::PointCloud2>("/lidar_road", 1);
        obs_pub = nh.advertise<sensor_msgs::PointCloud2>("/lidar_obstacle", 1);

        mask_sub.subscribe(nh, "/yolop_fusion_masks", 10);
        lidar_sub.subscribe(nh, "/lidar3D", 10);

        sync = std::make_shared<message_filters::Synchronizer<MySyncPolicy>>(MySyncPolicy(100), mask_sub, lidar_sub);
        sync->getPolicy()->setMaxIntervalDuration(ros::Duration(0.2));
        sync->registerCallback(boost::bind(&LidarProjector::callback, this, _1, _2));
        
        ROS_INFO("Integrated OPR & Distance Node");
        ROS_INFO("Target Actor ID is set to: %.0f", target_actor_id);
    }

    void publishCloud(ros::Publisher& pub, const pcl::PointCloud<pcl::PointXYZ>& pcl_cloud, const std_msgs::Header& header) {
        if (pcl_cloud.empty()) return;
        sensor_msgs::PointCloud2 ros_cloud;
        pcl::toROSMsg(pcl_cloud, ros_cloud);
        ros_cloud.header = header;
        pub.publish(ros_cloud);
    }

    void callback(const sensor_msgs::ImageConstPtr& mask_msg, const sensor_msgs::PointCloud2ConstPtr& lidar_msg) {
        cv::Mat fusion_mask;
        try {
            fusion_mask = cv_bridge::toCvShare(mask_msg, "bgr8")->image;
        } catch (cv_bridge::Exception& e) { return; }

        geometry_msgs::TransformStamped transform;
        try {
            transform = tf_buffer.lookupTransform("Camera-3", "lidar_link", ros::Time(0));
        } catch (tf2::TransformException& ex) { return; }

        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::fromROSMsg(*lidar_msg, *cloud);

        pcl::PointCloud<pcl::PointXYZ> road_cloud, obs_cloud;
        Eigen::Affine3d T = tf2::transformToEigen(transform);

        int P_actual = 0;
        int P_preserved = 0;
        double dist_sum = 0.0; 

        for (const auto& pt : cloud->points) {
            if (pt.z < -2.0) continue;

            Eigen::Vector3d p_l(pt.x, pt.y, pt.z);
            Eigen::Vector3d p_c = T * p_l;

            double x = -p_c.y();
            double y = -p_c.z();
            double z = p_c.x();

            if (z < 0.1 || z > 50.0) continue;

            bool is_target = (std::abs(pt.intensity - (target_actor_id / 255.0f)) < 0.001f);

            if (is_target) {
                    P_actual++; 
                }
                
            int u = static_cast<int>((K.at<double>(0,0) * x / z) + K.at<double>(0,2));
            int v = static_cast<int>((K.at<double>(1,1) * y / z) + K.at<double>(1,2));

            if (u >= 0 && u < fusion_mask.cols && v >= 0 && v < fusion_mask.rows) {
                cv::Vec3b pixel = fusion_mask.at<cv::Vec3b>(v, u);
                
                pcl::PointXYZ out_pt; 
                out_pt.x = pt.x; out_pt.y = pt.y; out_pt.z = pt.z;

                if (pixel[1] > 127) { 
                    obs_cloud.push_back(out_pt);

                    if (is_target) {
                        P_preserved++;
                        dist_sum += std::sqrt(std::pow(pt.x, 2) + std::pow(pt.y, 2)); // d = sqrt((x2-x1)^2 + (y2-y1)^2) 수식 적용
                    }
                } 
                else if (pixel[0] > 127) { 
                    road_cloud.push_back(out_pt);
                }
            }
        }

        double opr = (P_actual > 0) ? (double)P_preserved / P_actual * 100.0 : 0.0;
        double reaction_dist = (P_preserved >= N) ? dist_sum / P_preserved : 0.0;

        // 연속성 계산
        if (P_actual > 0) f_total++; 
        if (P_preserved >= N) f_detected++; 
        double continuity = (f_total > 0) ? (double)f_detected / f_total * 100.0 : 0.0;

        // 데이터 효율성 계산
        size_t total_points = cloud->size();
        size_t projected_points = obs_cloud.size() + road_cloud.size();
        double data_efficiency = (total_points > 0) ? (double)projected_points / total_points * 100.0 : 0.0;

        // [최종 로그 출력]
        ROS_INFO("=========================================");
        ROS_INFO("0) Raw LiDAR Total Points: %zu", total_points);
        ROS_INFO("1) LiDAR Obstacle Points: %zu", obs_cloud.size());
        ROS_INFO("2) LiDAR Road Points    : %zu", road_cloud.size());
        ROS_INFO("3) Total Projected Points : %zu", projected_points);
        ROS_INFO("4) Data Efficiency    : %.2f%% of Original", data_efficiency);
        //ROS_INFO("-----------------------------------------");
        //ROS_INFO("Target Actor ID  : %.0f (Sur-3)", target_actor_id);
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

        publishCloud(road_pub, road_cloud, lidar_msg->header);
        publishCloud(obs_pub, obs_cloud, lidar_msg->header);
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "lidar_projection");
    LidarProjector lp;
    ros::spin();
    return 0;
}
*/

//점개수+OPR(객체 보존율)
/*
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
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#include <tf2_eigen/tf2_eigen.h>
#include <Eigen/Dense>
#include <memory>

class LidarProjector {
private:
    ros::NodeHandle nh;
    ros::Publisher road_pub, obs_pub;
    tf2_ros::Buffer tf_buffer;
    tf2_ros::TransformListener tf_listener;
    cv::Mat K;

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::PointCloud2> MySyncPolicy;
    std::shared_ptr<message_filters::Synchronizer<MySyncPolicy>> sync;
    message_filters::Subscriber<sensor_msgs::Image> mask_sub;
    message_filters::Subscriber<sensor_msgs::PointCloud2> lidar_sub;

    // 추적할 차량의 Actor ID
    const float target_actor_id = 2.0f;

public:
    LidarProjector() : tf_listener(tf_buffer) {
        K = (cv::Mat_<double>(3, 3) << 320.0, 0.0, 320.0, 0.0, 320.0, 240.0, 0.0, 0.0, 1.0);

        road_pub = nh.advertise<sensor_msgs::PointCloud2>("/lidar_road", 1);
        obs_pub = nh.advertise<sensor_msgs::PointCloud2>("/lidar_obstacle", 1);

        mask_sub.subscribe(nh, "/yolop_fusion_masks", 10);
        lidar_sub.subscribe(nh, "/lidar3D", 10);

        sync = std::make_shared<message_filters::Synchronizer<MySyncPolicy>>(MySyncPolicy(50), mask_sub, lidar_sub);
        sync->getPolicy()->setMaxIntervalDuration(ros::Duration(0.1));
        sync->registerCallback(boost::bind(&LidarProjector::callback, this, _1, _2));
        
        ROS_INFO("Tracking Actor ID: %.0f", target_actor_id);
    }

    // 포인트 클라우드 발행 함수 (기존 기능 유지)
    void publishCloud(ros::Publisher& pub, const pcl::PointCloud<pcl::PointXYZ>& pcl_cloud, const std_msgs::Header& header) {
        if (pcl_cloud.empty()) return;
        sensor_msgs::PointCloud2 ros_cloud;
        pcl::toROSMsg(pcl_cloud, ros_cloud);
        ros_cloud.header = header;
        pub.publish(ros_cloud);
    }

    void callback(const sensor_msgs::ImageConstPtr& mask_msg, const sensor_msgs::PointCloud2ConstPtr& lidar_msg) {
        cv::Mat fusion_mask;
        try {
            fusion_mask = cv_bridge::toCvShare(mask_msg, "bgr8")->image;
        } catch (cv_bridge::Exception& e) { return; }

        geometry_msgs::TransformStamped transform;
        try {
            transform = tf_buffer.lookupTransform("Camera-3", "lidar_link", ros::Time(0));
        } catch (tf2::TransformException& ex) { return; }

        // 🆔 Actor ID를 읽기 위해 원본 데이터를 PointXYZI로 변환
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::fromROSMsg(*lidar_msg, *cloud);

        pcl::PointCloud<pcl::PointXYZ> road_cloud, obs_cloud;
        Eigen::Affine3d T = tf2::transformToEigen(transform);

        int P_actual = 0;    // 특정 장애물의 실제 전체 점 개수
        int P_preserved = 0; // ROI 내에서 살아남은 특정 장애물의 점 개수

        for (const auto& pt : cloud->points) {
             if (pt.z < -2.0) continue; // 바닥 필터링

            bool is_target = (std::abs(pt.intensity - (2.0f / 255.0f)) < 0.001f);
    
            if (is_target) P_actual++; // 분모: 특정 장애물의 전체 점 개수

            Eigen::Vector3d p_l(pt.x, pt.y, pt.z);
            Eigen::Vector3d p_c = T * p_l;
            double x = -p_c.y(); double y = -p_c.z(); double z = p_c.x();

            if (z < 0.1 || z > 50.0) continue;

            int u = static_cast<int>((K.at<double>(0,0) * x / z) + K.at<double>(0,2));
            int v = static_cast<int>((K.at<double>(1,1) * y / z) + K.at<double>(1,2));

            if (u >= 0 && u < fusion_mask.cols && v >= 0 && v < fusion_mask.rows) {
                cv::Vec3b pixel = fusion_mask.at<cv::Vec3b>(v, u);
                pcl::PointXYZ out_pt; out_pt.x = pt.x; out_pt.y = pt.y; out_pt.z = pt.z;

                if (pixel[1] > 127) { // 장애물 영역
                    obs_cloud.push_back(out_pt);
                    // 타겟 점이 장애물 영역 내에 투영되면 보존된 것으로 간주
                    if (is_target) P_preserved++; 
                } else if (pixel[0] > 127) { // 도로 영역
                    road_cloud.push_back(out_pt);
                }
            }
        }

        // 결과 데이터 계산 및 출력 
        double opr = (P_actual > 0) ? (double)P_preserved / P_actual * 100.0 : 0.0;

        ROS_INFO("=========================================");
        ROS_INFO("1) LiDAR Obstacle Points: %zu", obs_cloud.size());
        ROS_INFO("2) LiDAR Road Points    : %zu", road_cloud.size());
        ROS_INFO("3) Total Projected Points : %zu", obs_cloud.size() + road_cloud.size());
        ROS_INFO("-----------------------------------------");
        ROS_INFO("Target Actor ID  : %.0f", target_actor_id);
        ROS_INFO("P_actual (Total)   : %d", P_actual);
        ROS_INFO("P_preserved (ROI) : %d", P_preserved);
        ROS_INFO("OPR (Safety)     : %.2f%%", opr); // 95% 이상 목표!
        ROS_INFO("=========================================");

        publishCloud(road_pub, road_cloud, lidar_msg->header);
        publishCloud(obs_pub, obs_cloud, lidar_msg->header);
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "lidar_projection");
    LidarProjector lp;
    ros::spin();
    return 0;
}
*/


//점개수
/*
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
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#include <tf2_eigen/tf2_eigen.h>
#include <Eigen/Dense>
#include <memory>

class LidarProjector {
private:
    ros::NodeHandle nh;
    ros::Publisher road_pub, obs_pub;
    tf2_ros::Buffer tf_buffer;
    tf2_ros::TransformListener tf_listener;
    cv::Mat K;

    // 동기화 관련 멤버 변수
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::PointCloud2> MySyncPolicy;
    std::shared_ptr<message_filters::Synchronizer<MySyncPolicy>> sync;
    message_filters::Subscriber<sensor_msgs::Image> mask_sub;
    message_filters::Subscriber<sensor_msgs::PointCloud2> lidar_sub;

public:
    LidarProjector() : tf_listener(tf_buffer) {
        // 카메라 매트릭스 (재원님 설정값)
        K = (cv::Mat_<double>(3, 3) << 320.0, 0.0, 320.0, 0.0, 320.0, 240.0, 0.0, 0.0, 1.0);

        road_pub = nh.advertise<sensor_msgs::PointCloud2>("/lidar_road", 1);
        obs_pub = nh.advertise<sensor_msgs::PointCloud2>("/lidar_obstacle", 1);

        // 1. 구독자 설정 (nh를 통해 토픽 연결)
        mask_sub.subscribe(nh, "/yolop_fusion_masks", 10);
        lidar_sub.subscribe(nh, "/lidar3D", 10);

        // 2. 동기화 정책 생성 (큐 사이즈 50)
        // 컴파일 에러 해결: 정책 객체를 생성자에 명시적으로 전달
        sync = std::make_shared<message_filters::Synchronizer<MySyncPolicy>>(MySyncPolicy(50), mask_sub, lidar_sub);
        
        // 3. 허용 오차(slop) 설정: 0.1초 이내의 데이터는 동기화된 것으로 간주
        sync->getPolicy()->setMaxIntervalDuration(ros::Duration(0.1));
        sync->registerCallback(boost::bind(&LidarProjector::callback, this, _1, _2));
        
        ROS_INFO("🚀 [C++] Lidar Projection Node Started (Ready for Masks)");
        ros::spin();
    }

    void publishCloud(ros::Publisher& pub, const pcl::PointCloud<pcl::PointXYZ>& pcl_cloud, const std_msgs::Header& header) {
        if (pcl_cloud.empty()) return;
        sensor_msgs::PointCloud2 ros_cloud;
        pcl::toROSMsg(pcl_cloud, ros_cloud);
        ros_cloud.header = header;
        pub.publish(ros_cloud);
    }

    void callback(const sensor_msgs::ImageConstPtr& mask_msg, const sensor_msgs::PointCloud2ConstPtr& lidar_msg) {
        cv::Mat fusion_mask;
        try {
            fusion_mask = cv_bridge::toCvShare(mask_msg, "bgr8")->image;
        } catch (cv_bridge::Exception& e) { return; }

        geometry_msgs::TransformStamped transform;
        try {
            // TF 조회 (Camera-2 <-> Lidar3D-3)
            transform = tf_buffer.lookupTransform("Camera-3", "lidar_link", ros::Time(0)); //morai에서 카메라, 라이다 frame-id확인해서 수정
        } catch (tf2::TransformException& ex) { return; }

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*lidar_msg, *cloud);

        pcl::PointCloud<pcl::PointXYZ> road_cloud, obs_cloud;
        Eigen::Affine3d T = tf2::transformToEigen(transform);

        for (const auto& pt : cloud->points) {
            if (pt.z < -2.0) continue; // 바닥 필터링

            Eigen::Vector3d p_l(pt.x, pt.y, pt.z);
            Eigen::Vector3d p_c = T * p_l;

            // OpenCV 좌표계 투영 수식
            double x = -p_c.y();
            double y = -p_c.z();
            double z = p_c.x();

            if (z < 0.1 || z > 50.0) continue;

            int u = static_cast<int>((K.at<double>(0,0) * x / z) + K.at<double>(0,2));
            int v = static_cast<int>((K.at<double>(1,1) * y / z) + K.at<double>(1,2));

            if (u >= 0 && u < fusion_mask.cols && v >= 0 && v < fusion_mask.rows) {
                cv::Vec3b pixel = fusion_mask.at<cv::Vec3b>(v, u);
                // BGR 중 R(0)은 도로, G(1)은 장애물
                if (pixel[1] > 127) { 
                    obs_cloud.push_back(pt);
                } else if (pixel[0] > 127) { 
                    road_cloud.push_back(pt);
                }
            }
        }

        size_t obs_count = obs_cloud.size();   // 1) 장애물 포인트 개수
        size_t road_count = road_cloud.size(); // 2) 도로 포인트 개수
        size_t total_count = obs_count + road_count; // 3) 합계

        ROS_INFO("-----------------------------------------");
        ROS_INFO("1) LiDAR Obstacle Points: %zu", obs_count);
        ROS_INFO("2) LiDAR Road Points    : %zu", road_count);
        ROS_INFO("3) Total Projected Points : %zu", total_count);
        ROS_INFO("-----------------------------------------");

        publishCloud(road_pub, road_cloud, lidar_msg->header);
        publishCloud(obs_pub, obs_cloud, lidar_msg->header);
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "lidar_projection");
    LidarProjector lp;
    return 0;
}
    */