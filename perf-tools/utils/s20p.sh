echo performance > /sys/devices/system/cpu/cpufreq/policy6/scaling_governor
echo 2841600 > /sys/devices/system/cpu/cpufreq/policy6/scaling_max_freq
echo 2841600 > /sys/devices/system/cpu/cpufreq/policy6/scaling_min_freq
echo 680000000 > /sys/class/kgsl/kgsl-3d0/devfreq/max_freq
echo 680000000 > /sys/class/kgsl/kgsl-3d0/devfreq/min_freq