cmake_minimum_required(VERSION 3.0.2)
project(cable_perception)

find_package(catkin REQUIRED COMPONENTS
  cable_core
  cable_msgs
  cv_bridge
  geometry_msgs
  image_transport
  rospy
  sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS cable_core cable_msgs cv_bridge geometry_msgs image_transport rospy sensor_msgs
)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
  OPTIONAL
)
