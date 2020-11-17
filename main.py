#!/usr/bin/env python
# encoding: utf-8

import torch
import torchvision

model = torchvision.models.resnet101(pretrained=True)
# MUST set to eval mode
model.eval()
sample = torch.randn(1, 3, 224, 224)

model_path = "./data/resnet101.zip"
trace_model = torch.jit.trace(model, sample)
torch.jit.save(trace_model, model_path)
