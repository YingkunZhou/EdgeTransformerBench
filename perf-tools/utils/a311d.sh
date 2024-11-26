echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor
echo 1800000 > /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq
echo 1800000 > /sys/devices/system/cpu/cpufreq/policy0/scaling_min_freq
echo performance > /sys/devices/system/cpu/cpufreq/policy2/scaling_governor
echo 2208000 > /sys/devices/system/cpu/cpufreq/policy2/scaling_max_freq
echo 2208000 > /sys/devices/system/cpu/cpufreq/policy2/scaling_min_freq
echo 4 > /sys/class/mpgpu/max_freq
echo 4 > /sys/class/mpgpu/min_freq