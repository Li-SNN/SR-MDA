
import torch.nn as nn

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)


class netGenerator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(netGenerator, self).__init__()
        self.ReLU = nn.LeakyReLU(0.2, inplace=True)
        self.Tanh = nn.Tanh()
        self.conv1 = nn.ConvTranspose2d(nz, ngf*4, 3, 1, 0, bias=False)
        self.BatchNorm1 = nn.BatchNorm2d(ngf * 4)

        self.conv2 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, bias=False)
        self.BatchNorm2 = nn.BatchNorm2d(ngf * 2)
        self.Drop2 = nn.Dropout2d(p=0.5)
        # self.Drop2 = DropBlock2D()

        self.conv3 = nn.ConvTranspose2d(ngf * 2, ngf, 3, 2, 1, bias=False)
        self.BatchNorm3 = nn.BatchNorm2d(ngf)

        self.conv4 = nn.ConvTranspose2d(ngf, 3, 3, 2, 1, bias=False)
        self.BatchNorm4 = nn.BatchNorm2d(3)
        self.apply(weights_init)

    def forward(self, input):
        x1 = self.conv1(input)
        x1 = self.BatchNorm1(x1)
        x1 = self.ReLU(x1)

        x2 = self.conv2(x1)
        x2 = self.BatchNorm2(x2)
        x2 = self.ReLU(x2)
        x2 = self.Drop2(x2)

        x3 = self.conv3(x2)
        x3 = self.BatchNorm3(x3)
        x3 = self.ReLU(x3)

        x4 = self.conv4(x3)
        # x4 = self.BatchNorm4(x4)
        #
        # x5 = self.conv5(x4)
        # x5 = self.BatchNorm5(x5)
        # x6 = self.conv6(x5)
        # x4 = self.ReLU(x4)
        # x5 = self.conv6(x4)
        output = self.Tanh(x4)
        return output


class netDiscriminator(nn.Module):
    def __init__(self, ndf, nc):
        super(netDiscriminator, self).__init__()
        self.LeakyReLU = nn.LeakyReLU(0.2, inplace=True)

        self.conv1 = nn.Conv2d(nc, ndf, 3, 2, 1, bias=False)
        self.BatchNorm1 = nn.BatchNorm2d(ndf)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, 3, 2, 1, bias=False)
        self.BatchNorm2 = nn.BatchNorm2d(ndf * 2)
        self.Drop2 = nn.Dropout2d(p=0.5)

        self.conv3 = nn.Conv2d(ndf * 2, ndf*4, 3, 2, 1, bias=False)
        self.BatchNorm3 = nn.BatchNorm2d(ndf*4)
        self.conv4 = nn.Conv2d(ndf*4, ndf, 3, 1, 0, bias=False)
        # self.BatchNorm4 = nn.BatchNorm2d(ndf * 8)
        # self.conv5 = nn.Conv2d(ndf * 8, ndf * 4, 4, 1, 0, bias=False)
        # self.BatchNorm5 = nn.BatchNorm2d(ndf * 4)
        # self.conv6 = nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False)
        # self.conv5 = nn.Conv2d(ndf * 8, ndf * 2, 4, 1, 0, bias=False)
        #self.disc_linear = nn.Linear(ndf * 2, 1)
        # self.aux_linear = nn.Linear(ndf * 2, 1)
        self.aux_linear = nn.Linear(ndf, 17)
        # self.aux_linear = nn.Linear(ndf, 10)
        # self.softmax = nn.Sigmoid()
        self.softmax = nn.LogSoftmax(dim=-1)
        #self.sigmoid = nn.Sigmoid()
        self.ndf = ndf
        self.apply(weights_init)

    def forward(self, input):

        x = self.conv1(input)
        x = self.LeakyReLU(x)

        x1 = self.conv2(x)
        x1 = self.BatchNorm2(x1)
        x1 = self.LeakyReLU(x1)
        x1 = self.Drop2(x1)

        x2 = self.conv3(x1)
        x2 = self.BatchNorm3(x2)
        x2 = self.LeakyReLU(x2)
        # x = self.Drop2(x)

        x3 = self.conv4(x2)
        # x3 = self.BatchNorm4(x3)
        # x3 = self.LeakyReLU(x3)

        # x4 = self.conv5(x3)
        # x4 = self.BatchNorm5(x4)
        # x5 = self.conv6(x4)
        # x5 = x3.view(-1, self.ndf * 2)
        # x5 = self.softmax(x3)
        # c = x5.view(-1, 1).squeeze(0)
        x4 = x3.view(-1, self.ndf)

        c = self.aux_linear(x4)
        s = self.softmax(c)

        # c = self.softmax(c)
        #s = self.disc_linear(x).squeeze()
        #s = self.sigmoid(s)
        return s