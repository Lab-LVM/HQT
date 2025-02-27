import torch
import timm
import torchvision
import importlib
from torch import nn
from torch.autograd import Function
from torch.optim import SGD, Adam

class BihalfHash(Function):
    @staticmethod
    def forward(ctx, U, gamma):
        # Yunqiang for half and half (optimal transport)
        _, index = U.sort(0, descending=True)
        N, D = U.shape
        B_creat = torch.cat((torch.ones([int(N / 2), D]), -torch.ones([N - int(N / 2), D]))).cuda()
        B = torch.zeros(U.shape).cuda().scatter_(0, index, B_creat)

        ctx.save_for_backward(U, B)
        ctx.gamma = gamma
        return B

    @staticmethod
    def backward(ctx, g):
        U, B = ctx.saved_tensors
        add_g = (U - B) / (B.numel())

        grad = g + ctx.gamma * add_g

        return grad, None

class Hash(Function):
    @staticmethod
    def forward(ctx, input):
        # ctx.save_for_backward(input)
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        # input,  = ctx.saved_tensors
        # grad_output = grad_output.data

        return grad_output, None

def load_backbone(arch, use_timm=False):
    if use_timm:
        if "vgg" in arch:
            model = timm.create_model('vgg16', pretrained=True)
            # model.head = nn.Sequential(*list(model.head.children())[:-2])
    else:
        if "vgg" in arch:
            model = torchvision.models.vgg16(weights="IMAGENET1K_V1")
            model.classifier = nn.Sequential(*list(model.classifier.children())[:6])
    for param in model.parameters():
        param.requires_grad = False
    return model

def get_optimizer(model, hash_model, optimizer, lr=1e-4, momentum=0.9, weight_decay=2e-5,):
    if optimizer.lower() == 'sgd':
        if hash_model == 'uhscm':
            optimizer = SGD([{'params': model.model.features.parameters(), 'lr': lr * 0.05},
                                    {'params': model.model.classifier.parameters(), 'lr': lr * 0.05},
                                    {'params': model.fc_encode.parameters(), 'lr': lr}],
                                    lr=lr, weight_decay=weight_decay)
        else:
            optimizer = SGD(model.fc_encode.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer.lower() == 'adam':
        # using only cibhash
        optimizer = Adam(model.fc_encode.parameters(), lr=lr)
    else:
        raise ValueError("Please check your optimizer setting: Options are: sgd, adam")
    return optimizer

def get_model(hash_model, class_name="Net", **kwargs):
    module = importlib.import_module(f"models.{hash_model}")
    model = getattr(module, hash_model.capitalize() + class_name)
    return model(**kwargs)

def get_loss(hash_model, class_name="Loss", **kwargs):
    module = importlib.import_module(f"models.{hash_model}")
    loss = getattr(module, hash_model.capitalize() + class_name)
    return loss(**kwargs)

def adjust_learning_rate(optimizer, epoch, lr, lr_decay):
    lr = lr * (0.1 ** (epoch // lr_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
