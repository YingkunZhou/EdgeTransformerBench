class remote_device():
    def __init__(self, name, host, port,
                cpu_target="llvm -mtriple=aarch64-linux-gnu",
                opencl_target=None,
                vulkan_target=None,):
        self.name = name
        self.host = host
        self.port = port
        self.cpu_target = cpu_target
        self.opencl_target = opencl_target
        self.vulkan_target = vulkan_target


orpi5b = remote_device(
    name="orpi5b",
    host="192.168.3.145",
    port=9090,
    cpu_target="llvm -mtriple=aarch64-linux-gnu -mattr=+v8.2a,+neon,+fullfp16",
    opencl_target="opencl -device=mali",
)

vim3 = remote_device(
    name="vim3",
    host="192.168.3.31",
    port=9190,
    cpu_target="llvm -mtriple=aarch64-linux-gnu -mattr=+neon",
    opencl_target="opencl -device=mali",
)

andr888 = remote_device(
    name="andr888",
    host="192.168.3.12",
    port=9090,
    cpu_target="llvm -mtriple=aarch64-unknown-linux-android24 -mattr=+v8.2a,+neon,+fullfp16",
    opencl_target="opencl -device=adreno",
)

remote_device_list =[]
remote_device_list.append(orpi5b)
remote_device_list.append(andr888)
remote_device_list.append(vim3)
def find_device_by_name(name,devices = remote_device_list):
    for device in devices:
        if device.name == name:
            return device
    return None
