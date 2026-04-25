cmake_minimum_required(VERSION 3.0.2)
project(cable_planning)

find_package(catkin REQUIRED COMPONENTS
  cable_core
  cable_msgs
  geometry_msgs
  rospy
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS cable_core cable_msgs geometry_msgs rospy std_msgs
)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
  OPTIONAL
)
