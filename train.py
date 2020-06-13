#!/usr/bin/env python
from onmt.bin.train import main
import torch


if __name__ == "__main__":
    if torch.cuda.is_available():
        print("cuda is available")

    gpucnt = torch.cuda.device_count()
    print("gpu device count is ",gpucnt)

    if gpucnt > 0:
        for i in range(gpucnt):
            gpuname = torch.cuda.get_device_name(i)

    current_device = torch.cuda.current_device()
    print("current_device is ",current_device)
    main()
