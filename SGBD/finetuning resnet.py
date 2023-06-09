"""This file has been used to run tests for the finetuing of resnet18"""
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

import main

DS = "CIFAR10"
E = 15

if __name__ == '__main__':
    # Those functions are usde to instantiate the models, load the weights and set the last layer to the correct size
    def std_model(use_cifar=True):
        mod = resnet18(num_classes=10)
        mod.__class__.__name__ = "Resnet non pretrained"
        return mod


    def pre_trained_model(*args, **kwargs):
        mod = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        mod.fc = nn.Linear(512, 10, bias=True)
        mod.__class__.__name__ = "Resnet pretrained"
        return mod


    main.main(True, pre_trained_model, dataset=DS, global_stepsize=1, write_logs=True, epochs=E, alfa_target=1 / 10)
    main.main(True, pre_trained_model, dataset=DS, global_stepsize=.1, write_logs=True, epochs=E, alfa_target=1 / 10)
    main.main(True, pre_trained_model, dataset=DS, global_stepsize=.01, write_logs=True, epochs=E, alfa_target=1 / 10)
    main.main(True, pre_trained_model, dataset=DS, global_stepsize=.001, write_logs=True, epochs=E, alfa_target=1 / 10)
    main.main(True, pre_trained_model, dataset=DS, global_stepsize=.0001, write_logs=True, epochs=E, alfa_target=1 / 10)
    main.main(False, pre_trained_model, dataset=DS, global_stepsize=0, write_logs=True, epochs=E, alfa_target=1 / 10)

    main.main(True, std_model, dataset=DS, global_stepsize=1, write_logs=True, epochs=E, alfa_target=1 / 10)
    main.main(True, std_model, dataset=DS, global_stepsize=.1, write_logs=True, epochs=E, alfa_target=1 / 10)
    main.main(True, std_model, dataset=DS, global_stepsize=.01, write_logs=True, epochs=E, alfa_target=1 / 10)
    main.main(True, std_model, dataset=DS, global_stepsize=.001, write_logs=True, epochs=E, alfa_target=1 / 10)
    main.main(True, std_model, dataset=DS, global_stepsize=.0001, write_logs=True, epochs=E, alfa_target=1 / 10)
    main.main(False, std_model, dataset=DS, global_stepsize=0, write_logs=True, epochs=E, alfa_target=1 / 10)
