import torch

print("使用gpu数量为：", torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
try:
    print(torch.cuda.get_device_name(1))
except:
    print("并未使用其他显卡")