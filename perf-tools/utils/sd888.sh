echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor
echo 1804800 > /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq
echo 1804800 > /sys/devices/system/cpu/cpufreq/policy0/scaling_min_freq
echo performance > /sys/devices/system/cpu/cpufreq/policy4/scaling_governor
echo 2112000 > /sys/devices/system/cpu/cpufreq/policy4/scaling_max_freq
echo 2112000 > /sys/devices/system/cpu/cpufreq/policy4/scaling_min_freq
echo performance > /sys/devices/system/cpu/cpufreq/policy7/scaling_governor
echo 2150400 > /sys/devices/system/cpu/cpufreq/policy7/scaling_max_freq
echo 2150400 > /sys/devices/system/cpu/cpufreq/policy7/scaling_max_freq
# echo performance > /sys/class/kgsl/kgsl-3d0/devfreq/governor
echo 608000000 > /sys/class/kgsl/kgsl-3d0/devfreq/max_freq
echo 608000000 > /sys/class/kgsl/kgsl-3d0/devfreq/min_freq
