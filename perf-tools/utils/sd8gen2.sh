echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor
echo 2016000 > /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq
echo 2016000 > /sys/devices/system/cpu/cpufreq/policy0/scaling_min_freq
echo performance > /sys/devices/system/cpu/cpufreq/policy3/scaling_governor
echo 2803200 > /sys/devices/system/cpu/cpufreq/policy3/scaling_max_freq
echo 2803200 > /sys/devices/system/cpu/cpufreq/policy3/scaling_min_freq
echo performance > /sys/devices/system/cpu/cpufreq/policy7/scaling_governor
echo 2841600 > /sys/devices/system/cpu/cpufreq/policy7/scaling_max_freq
echo 2841600 > /sys/devices/system/cpu/cpufreq/policy7/scaling_min_freq
echo 680000000 > /sys/class/kgsl/kgsl-3d0/devfreq/max_freq
echo 680000000 > /sys/class/kgsl/kgsl-3d0/devfreq/min_freq