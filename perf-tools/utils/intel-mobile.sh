#!/bin/bash
# for meteorlake, turn off the SMT
for i in {0..3}
do
  echo 4400000 > /sys/devices/system/cpu/cpufreq/policy$i/scaling_min_freq
  echo 4400000 > /sys/devices/system/cpu/cpufreq/policy$i/scaling_max_freq
  echo performance > /sys/devices/system/cpu/cpufreq/policy$i/scaling_governor
done
for i in {4..7}
do
  echo 3600000 > /sys/devices/system/cpu/cpufreq/policy$i/scaling_min_freq
  echo 3600000 > /sys/devices/system/cpu/cpufreq/policy$i/scaling_max_freq
  echo performance > /sys/devices/system/cpu/cpufreq/policy$i/scaling_governor
done