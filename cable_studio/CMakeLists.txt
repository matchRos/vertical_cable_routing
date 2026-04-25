cmake_minimum_required(VERSION 3.0.2)
project(cable_studio)

find_package(catkin REQUIRED COMPONENTS
  cable_msgs
  cable_orchestrator
  python_qt_binding
  rospy
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS cable_msgs cable_orchestrator python_qt_binding rospy std_msgs
)

install(DIRECTORY resource/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/resource
  OPTIONAL
)
