echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor
echo 1800000 > /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq
echo 1800000 > /sys/devices/system/cpu/cpufreq/policy0/scaling_min_freq
echo performance > /sys/devices/system/cpu/cpufreq/policy4/scaling_governor
echo 2256000 > /sys/devices/system/cpu/cpufreq/policy4/scaling_max_freq
echo 2256000 > /sys/devices/system/cpu/cpufreq/policy4/scaling_min_freq
echo performance > /sys/devices/system/cpu/cpufreq/policy6/scaling_governor
echo 2256000 > /sys/devices/system/cpu/cpufreq/policy6/scaling_max_freq
echo 2256000 > /sys/devices/system/cpu/cpufreq/policy6/scaling_min_freq
echo performance > /sys/devices/platform/fb000000.gpu/devfreq/fb000000.gpu/governor
echo 1000000000 > /sys/devices/platform/fb000000.gpu/devfreq/fb000000.gpu/max_freq
echo 1000000000 > /sys/devices/platform/fb000000.gpu/devfreq/fb000000.gpu/min_freq