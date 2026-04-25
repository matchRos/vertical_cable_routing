cmake_minimum_required(VERSION 3.0.2)
project(cable_motion)

find_package(catkin REQUIRED COMPONENTS
  actionlib
  cable_msgs
  geometry_msgs
  rospy
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS actionlib cable_msgs geometry_msgs rospy std_msgs
)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
  OPTIONAL
)
