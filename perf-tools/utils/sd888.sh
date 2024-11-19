echo performance > /sys/devices/system/cpu/cpufreq/policy4/scaling_governor
echo 2419200 > /sys/devices/system/cpu/cpufreq/policy4/scaling_max_freq
echo 2419200 > /sys/devices/system/cpu/cpufreq/policy4/scaling_min_freq
echo performance > /sys/devices/system/cpu/cpufreq/policy7/scaling_governor
echo 2150400 > /sys/devices/system/cpu/cpufreq/policy7/scaling_max_freq
echo 2150400 > /sys/devices/system/cpu/cpufreq/policy7/scaling_max_freq
# echo performance > /sys/class/kgsl/kgsl-3d0/devfreq/governor
echo 608000000 > /sys/class/kgsl/kgsl-3d0/devfreq/max_freq
echo 608000000 > /sys/class/kgsl/kgsl-3d0/devfreq/min_freq