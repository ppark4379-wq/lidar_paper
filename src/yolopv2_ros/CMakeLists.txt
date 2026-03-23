cmake_minimum_required(VERSION 3.0.2)
project(yolopv2_ros)

add_compile_options(-std=c++14)

## 1. 필수 ROS 패키지 로드 (공통 사용)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  cv_bridge
  image_transport
  pcl_ros
  pcl_conversions
  message_filters
  tf2_ros
  tf2_sensor_msgs
  tf2_eigen
  morai_msgs
  nav_msgs
)

## 2. 외부 라이브러리(PCL, Eigen) 로드 (공통 사용)
find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)

## 3. 캐킨 패키지 설정
catkin_package(
  CATKIN_DEPENDS roscpp rospy std_msgs sensor_msgs cv_bridge pcl_ros message_filters tf2_ros
)

## 4. 헤더 파일 경로 포함 (공통 적용)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

###########
## Build ##
###########

## 5-A. 첫 번째 노드: Projection ROI 비교 및 가시화 노드
add_executable(lidar_projection_node src/lidar_projection.cpp)
target_link_libraries(lidar_projection_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)

## 5-B. 두 번째 노드: 고정 ROI 비교 및 가시화 노드
add_executable(lidar_fixed_node src/lidar_fixed_roi.cpp)
target_link_libraries(lidar_fixed_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)

## 5-C. 세 번째 노드: 경로기반 ROI 비교 및 가시화 노드
add_executable(lidar_waypoint_node src/lidar_waypoint.cpp)
target_link_libraries(lidar_waypoint_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)

## Projection ROI 시간 측정
add_executable(projection_timer_node src/projection_timer.cpp)
target_link_libraries(projection_timer_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)

## Fixed ROI 시간 측정
add_executable(fixed_timer_node src/fixed_timer.cpp)
target_link_libraries(fixed_timer_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)

## Waypoint ROI 시간 측정
add_executable(waypoint_timer_node src/waypoint_timer.cpp)
target_link_libraries(waypoint_timer_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)