#!/bin/bash
# for meteorlake, turn off the SMT
for i in {0..3}
do
  echo 3200000 > /sys/devices/system/cpu/cpufreq/policy$i/scaling_min_freq
  echo 3200000 > /sys/devices/system/cpu/cpufreq/policy$i/scaling_max_freq
  echo performance > /sys/devices/system/cpu/cpufreq/policy$i/scaling_governor
done
for i in {4..8}
do
  echo 3000000 > /sys/devices/system/cpu/cpufreq/policy$i/scaling_min_freq
  echo 3000000 > /sys/devices/system/cpu/cpufreq/policy$i/scaling_max_freq
  echo performance > /sys/devices/system/cpu/cpufreq/policy$i/scaling_governor
done