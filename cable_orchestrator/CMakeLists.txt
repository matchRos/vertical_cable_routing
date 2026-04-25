cmake_minimum_required(VERSION 3.0.2)
project(cable_orchestrator)

find_package(catkin REQUIRED COMPONENTS
  actionlib
  cable_core
  cable_motion
  cable_msgs
  cable_perception
  cable_planning
  rospy
  std_msgs
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS actionlib cable_core cable_motion cable_msgs cable_perception cable_planning rospy std_msgs
)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
  OPTIONAL
)
