cmake_minimum_required(VERSION 3.0.2)
project(cable_msgs)

find_package(catkin REQUIRED COMPONENTS
  actionlib_msgs
  geometry_msgs
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS actionlib_msgs geometry_msgs std_msgs
)
