from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

import main

DS = "CIFAR10"
EPOCHS = 10

if __name__ == '__main__':
    std_model = lambda x: resnet18(num_classes=10)


    def pre_trained_model(use_cifar=True):
        mod = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        mod.fc = nn.Linear(512, 10, bias=True)
        return mod


    # train_mod, opt = main.main(True, pre_trained_model, dataset=DS, write_logs=True, epochs=3, alfa_target=1 / 4)
    # gg = lambda x: train_mod
    main.main(True, std_model, dataset=DS, global_stepsize=1, write_logs=True, epochs=EPOCHS, alfa_target=1 / 10)
    main.main(True, std_model, dataset=DS, global_stepsize=.1, write_logs=True, epochs=EPOCHS, alfa_target=1 / 10)
    main.main(True, std_model, dataset=DS, global_stepsize=.01, write_logs=True, epochs=EPOCHS, alfa_target=1 / 10)
    main.main(False, std_model, dataset=DS, write_logs=True, epochs=EPOCHS, alfa_target=1 / 10)
    # main.main(False, pre_trained_model, dataset=DS, write_logs=True, epochs=EPOCHS, alfa_target=1 / 10)

    # train_mod_nopre, opt = main.main(std_model, gg, dataset=DS, write_logs=True, epochs=5, alfa_target=1 / 10)
    # gg2 = lambda x: train_mod_nopre
    # main.main(False, std_model, dataset=DS, write_logs=True, epochs=15, alfa_target=1 / 10)
