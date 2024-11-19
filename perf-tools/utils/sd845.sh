echo performance > /sys/devices/system/cpu/cpufreq/policy4/scaling_governor
echo 2323200 > /sys/devices/system/cpu/cpufreq/policy4/scaling_max_freq
echo 2323200 > /sys/devices/system/cpu/cpufreq/policy4/scaling_min_freq
echo 710000000 > /sys/class/kgsl/kgsl-3d0/devfreq/max_freq
echo 710000000 > /sys/class/kgsl/kgsl-3d0/devfreq/min_freq