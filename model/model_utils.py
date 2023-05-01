from torch import nn


def get_resnet_encoder(in_channel, net):
    encoder = []

    # iterate over the off-the-shelf ResNet architecture and grab which layer is needed
    for name, module in net.named_children():
        # reset the first layer to match the input channels
        if name == "conv1":
            module = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # toss max pooling layers
        if isinstance(module, nn.MaxPool2d):
            continue
        # toss linear (classification) layers
        if isinstance(module, nn.Linear):
            continue
        if name in ["layer4"]:
            continue
        # toss adaptive pooling layers
        # if isinstance(module, nn.AdaptiveAvgPool2d):
        #     continue

        # put rest of layers of ResNet in a list
        encoder.append(module)
    # combine the ResNet layers from the encoder list
    return nn.Sequential(*encoder)
