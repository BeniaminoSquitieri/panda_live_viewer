#!/usr/bin/env bash
# Shared ROS 2 + CycloneDDS environment for this project.
#
# Source this file (it is idempotent and safe to source multiple times) instead
# of exporting ROS_DOMAIN_ID / RMW_IMPLEMENTATION / CYCLONEDDS_URI by hand every
# time you open a terminal:
#
#     source "$(dirname "${BASH_SOURCE[0]}")/ros_env.sh"   # from a script
#     source ~/panda_live_viewer/ros_env.sh                # interactively
#
# It honours any value already set in the environment and only fills in the
# project defaults when a variable is missing, so it never clobbers a custom
# setup. It is machine-agnostic: CYCLONEDDS_URI points at
# $HOME/.ros/cyclonedds.xml, whose <Interfaces>/<Peers> are per-host, so the
# same file works on the robot laptop and on the GPU server after a git pull.

# DDS domain shared by every node in this project (robot, perception, VLM, BT).
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"

# CycloneDDS is the required RMW here (default Fast-DDS discovery is unreliable
# across the robot<->server link).
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"

# Per-machine CycloneDDS config (network interface + static peers). Only set the
# URI when the file actually exists, so a host without it does not end up with a
# broken file:// URI that makes every ros2 command fail.
if [ -z "${CYCLONEDDS_URI:-}" ] && [ -f "$HOME/.ros/cyclonedds.xml" ]; then
  export CYCLONEDDS_URI="file://$HOME/.ros/cyclonedds.xml"
fi

# Network discovery must be allowed: ROS_LOCALHOST_ONLY would isolate this node
# from the other machine, breaking the robot<->server topics/services.
unset ROS_LOCALHOST_ONLY
