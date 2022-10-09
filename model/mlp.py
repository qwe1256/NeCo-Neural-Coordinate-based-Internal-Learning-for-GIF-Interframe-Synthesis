import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, numInput, numOutput, numNeurons, depth, skipLst, bodyActi, lastActi) -> None:
        super(MLP, self).__init__()
        lastNumOutput = numInput
        net = []
        for i in range(depth):
            if i == depth-1:
                nextNumOutput = numOutput
                nextActi = lastActi
            else:
                nextNumOutput = numNeurons
                nextActi = bodyActi
            if i in skipLst:
                lastNumOutput += numInput
            if nextActi is not None:
                net.append(nn.Sequential(nn.Linear(lastNumOutput,
                        nextNumOutput), getattr(nn, nextActi)()))
            else:
                net.append(nn.Linear(lastNumOutput,nextNumOutput))
            lastNumOutput = nextNumOutput
        self.net = nn.ModuleList(net)
        self.skipLst = skipLst

    def forward(self, X):
        curr = X
        for i, l in enumerate(self.net):
            if i in self.skipLst:
                curr = torch.cat((curr, X), dim=1)
            curr = l(curr)
        return curr
