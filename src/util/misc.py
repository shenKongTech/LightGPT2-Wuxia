import torch


def get_device() -> torch.device:
    """获取可用计算设备并打印信息"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"运行设备: {device}")
    if device.type == "cuda":
        print(
            f"显卡型号: {torch.cuda.get_device_name()}, "
            f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f}GB"
        )
    return device

