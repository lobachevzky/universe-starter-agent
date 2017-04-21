#! /usr/bin/env bash

#echo ------------------------------------------------------------------
#echo Xvfb -shmem -screen 0 1280x1024x24
#Xvfb -shmem -screen 0 1280x1024x24 &
#echo ------------------------------------------------------------------
echo source /catkin/devel/setup.bash
source /catkin/devel/setup.bash
echo ------------------------------------------------------------------
echo roslaunch a3c train.launch log-dir:=$1 num_workers:=$2 i:=$3 remotes:=$4 gui:=$5
echo ------------------------------------------------------------------
roslaunch a3c train.launch log-dir:=$1 num_workers:=$2 i:=$3 remotes:=$4 gui:=$5
