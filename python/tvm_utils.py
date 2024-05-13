class remote_device():
    def __init__(self, name,
                cpu_target="llvm -mtriple=aarch64-linux-gnu",
                opencl_target=None,
                vulkan_target=None,):
        self.name = name
        self.cpu_target = cpu_target
        self.opencl_target = opencl_target
        self.vulkan_target = vulkan_target

remote_device_list =[]
remote_device_list.append(orpi5b := remote_device(
    name="orpi5b",
    cpu_target="llvm -mtriple=aarch64-linux-gnu -mattr=+v8.2a,+neon,+fullfp16",
    opencl_target="opencl -device=mali",
))
remote_device_list.append(vim3l := remote_device(
    name="vim3l",
    cpu_target="llvm -mtriple=aarch64-linux-gnu -mattr=+v8.2a,+neon,+fullfp16",
    opencl_target="opencl -device=mali",
))
remote_device_list.append(m1 := remote_device(
    name="m1",
    cpu_target="llvm -mtriple=aarch64-linux-gnu -mattr=+v8.2a,+neon,+fullfp16",
))

remote_device_list.append(andr888 := remote_device(
    name="andr888",
    cpu_target="llvm -mtriple=aarch64-unknown-linux-android24 -mattr=+v8.2a,+neon,+fullfp16",
    opencl_target="opencl -device=adreno",
))
remote_device_list.append(vim3 := remote_device(
    name="vim3",
    cpu_target="llvm -mtriple=aarch64-linux-gnu -mattr=+neon",
    opencl_target="opencl -device=mali",
))

def find_device_by_name(name,devices = remote_device_list):
    for device in devices:
        if device.name == name:
            return device
    return None