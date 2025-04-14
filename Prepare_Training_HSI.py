# define device
import math
import os
from datetime import datetime
import time
import torch.nn.functional as F
from pandas.core.common import random_state
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt
import pdb

from util.evaluate import stats
from util.auxiliary import get_logger
import random

# torch imports
import torch
from sklearn.metrics import confusion_matrix
from torch.optim import optimizer
from torch.utils.data import DataLoader,Dataset
from torch import optim,nn
import torchvision
from model.GAN17 import netGenerator as netGenerator
# from model.GAN23 import netGenerator as netGenerator
# from model.GAN64 import netG as netGenerator
# from model.Ge32 import DCGAN_generator as netGenerator
# from model.Ge32 import DCGAN_generator as netGenerator
# from model.GAN32DC import DCGAN_discriminator as netDiscriminator
# from model.GAN17 import netDiscriminator as netDiscriminator
from model.GAN17DC import netDiscriminator as netDiscriminator
from model.GAN17DC1 import netDiscriminator as netDiscriminator1
from model.GAN17DC2 import netDiscriminator as netDiscriminator2
# from model.GAN17 import netDiscriminator as netDiscriminator
# from model.GAN23DC import netDiscriminator as netDiscriminator
# from model.GAN64 import netD as netDiscriminator
import scipy.io as sio
from sklearn.decomposition import PCA
from util.data_preprocess import loadData

from model.SNN_RepVGGstard import TGRS
# from model.SNN_RepVGGstard32 import TGRS
# from model.SNN_RepIdentiry import TGRS
# from model.SNN_Rep33 import TGRS
# from model.SNN_Rep33Identir import TGRS
# from model.TGRS import TGRS
# from model.SNN_VGG7_SE import SNNVGG7SE as TGRS
# from model.SNN_SWRB import TGRS
from util.data_preprocess import load_hyper

device = torch.device("cuda:0")

generatorLosses = []
discriminatorLosses = []
classifierLosses = []

#training starts0

# train_samples = 1000
# train_samples = [155,436,49,72,31,118,31,86,22] #PUshujuji
# train_samples = [84,20,8,155,10,29,164,17,13] #WHLK shujuji 500sample
train_samples = [5,139,81,23,47,71,3,46,2,95,240,58,20,123,38,9] #IPshujuji
# train_samples = [11,20,11,8,14,22,19,62,34,18,9,11,5,6,40,10] #SAshujuji
# train_samples = [174,88,40,21,5,18,23,70,37,41,66,14,35,72,4,292] #WHHCSAshujuji # 1000/257530
# train_samples = [348,176,80,42,10,36,46,140,74,82,132,28,70,144,8,584] #WHHCSAshujuji # 2000/257530
# train_samples = [522,264,120,63,15,54,69,210,111,123,198,42,105,216,12,876] #WHHCSAshujuji # 3000/257530

train_percent = 0.5

def applyPCA(X,numComponents):
    newX=np.reshape(X,(-1,X.shape[2]))
    pca=PCA(n_components=numComponents,whiten=True)
    newX=pca.fit_transform(newX)
    newX=np.reshape(newX,(X.shape[0],X.shape[1],numComponents))
    return newX
####加padding
def padWithZeros(X,margin=2):
    newX=np.zeros((X.shape[0]+2*margin,X.shape[1]+2*margin,X.shape[2]))
    newX[margin:X.shape[0]+margin,margin:X.shape[1]+margin,:] = X
    return newX

def flip(data):
    y_4 = np.zeros_like(data)
    y_1 = y_4
    y_2 = y_4
    first = np.concatenate((y_1, y_2, y_1), axis=1)
    second = np.concatenate((y_4, data, y_4), axis=1)
    third = first
    Data = np.concatenate((first, second, third), axis=0)
    return Data

data_path = './Data'
# name = 'PU'
# name = 'WHHC'
# name = 'WHLK'
# name = 'SV'
name = 'IP'
# matfn1 = './Data/PaviaU.mat'
# matfn1 = './Data/WHU_Hi_HanChuan.mat'
# matfn1 = './Data/WHU_Hi_LongKou.mat'
matfn1 = './Data/Indian_pines_corrected.mat'
# matfn1 = './Data/salinas_corrected.mat'
data1 = sio.loadmat(matfn1)
# X = data1['paviaU']
# X = data1['WHU_Hi_HanChuan']
# X = data1['WHU_Hi_LongKou']
# X = data1['salinas_corrected']
X = data1['indian_pines_corrected']
# matfn2='./Data/PaviaU_gt.mat'
# matfn2='./Data/WHU_Hi_HanChuan_gt.mat'
# matfn2='./Data/WHU_Hi_LongKou_gt.mat'
# matfn2='./Data/salinas_gt.mat'
matfn2='./Data/Indian_pines_gt.mat'
data2=sio.loadmat(matfn2)
# y = data2['paviaU_gt']
# y = data2['WHU_Hi_HanChuan_gt']
# y = data2['WHU_Hi_LongKou_gt']
y = data2['indian_pines_gt']
# y = data2['salinas_gt']
# test_ratio=0.023344
# test_ratio=0.90
pca_components = 3
spatial_size=17
batch_size = 100
X_pca=applyPCA(X,numComponents=pca_components)
print('Data shape after PCA :',X_pca.shape)
[nRow, nColumn, nBand] = X_pca.shape
pcdata = flip(X_pca)
groundtruth = flip(y)

epochs = 1000

""" Testing dataset"""

advWeight = 0.1 # adversarial weight

loss = nn.BCELoss()
criterion = nn.CrossEntropyLoss()

file = open("util/ExternalClassifier.txt", "w")

c_label = torch.LongTensor(batch_size)
c_label = c_label.cuda()
c_label = Variable(c_label)

f_label = torch.LongTensor(batch_size)
f_label = torch.LongTensor(batch_size)
f_label = f_label.cuda()
f_label = Variable(f_label)

def train(datasetLoader):
    best_acc = -1
    best_k = -1
    best_aa = -1
    best_each_acc = -1
    train_time = 0
    test_time = 0
    for epoch in range(epochs):
        flag = None
        for param in netD.parameters():
            param.requires_grad = True
        for param in netD1.parameters():
            param.requires_grad = True
        for param in netD2.parameters():
            param.requires_grad = True
        netC.train()
        netG.train()
        netD1.train()
        netD.train()
        netD2.train()
        train_start_time = time.time()
        # if ((epoch+1)%200==0):
        #     # optG.param_groups[0]['lr'] *= 0.9
        #     optG.param_groups[0]['lr'] *= 0.01
        #     optD.param_groups[0]['lr'] *= 0.01
        #     optD1.param_groups[0]['lr'] *= 0.01
        #     optD2.param_groups[0]['lr'] *= 0.01
        #     optD.param_groups[0]['lr'] *= 0.9
            # optC.param_groups[0]['lr'] *= 0.1

        # for i, data in enumerate(datasetLoader, 0):
        for batch_idx, (inputs, labels) in enumerate(train_loader):

            # for j in range(2):
            inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = torch.autograd.Variable(inputs), torch.autograd.Variable(labels)
            tmpBatchSize = len(labels)
            c_label.resize_(batch_size).copy_(labels)

            label = np.full(batch_size, num_classes)
            f_label.data.resize_(batch_size).copy_(torch.from_numpy(label))

            true_label = torch.ones(batch_size, device=device)
            # true_label = torch.ones(batch_size, 1,device=device)
            fake_label = torch.zeros(batch_size, device=device)
            # fake_label = torch.zeros(batch_size, 1,device=device)

            r = torch.randn(batch_size, batch_size, 1, 1, device=device)
            fakeImageBatch = netG(r)
            # print(flag)
            if(flag==0):
                optD1.zero_grad()
                predictionsReal1 = netD1(inputs)
                lossDiscriminator1 = criterion(predictionsReal1, labels)  # labels = 1
                lossDiscriminator1.backward(retain_graph=True)
                predictionsFake1 = netD1(fakeImageBatch)
                lossFake1 = criterion(predictionsFake1, f_label)  # labels = 0
                lossFake1.backward(retain_graph=True)
                optD1.step()

                optD2.zero_grad()
                predictionsReal2 = netD2(inputs)
                lossDiscriminator2 = criterion(predictionsReal2, labels)  # labels = 1
                lossDiscriminator2.backward(retain_graph=True)
                predictionsFake2 = netD2(fakeImageBatch)
                lossFake2 = criterion(predictionsFake2, f_label)  # labels = 0
                lossFake2.backward(retain_graph=True)
                optD2.step()
            # optD.zero_grad()
            # predictionsReal = netD(inputs)
            # lossDiscriminator = criterion(predictionsReal, labels)  # labels = 1
            # lossDiscriminator.backward(retain_graph=True)
            # predictionsFake = netD(fakeImageBatch)
            # lossFake = criterion(predictionsFake, f_label)  # labels = 0
            # lossFake.backward(retain_graph=True)
            # optD.step()

            if(flag == 1):
                optD.zero_grad()
                predictionsReal = netD(inputs)
                lossDiscriminator = criterion(predictionsReal, labels)  # labels = 1
                lossDiscriminator.backward(retain_graph=True)
                predictionsFake = netD(fakeImageBatch)
                lossFake = criterion(predictionsFake, f_label)  # labels = 0
                lossFake.backward(retain_graph=True)
                optD.step()

                optD2.zero_grad()
                predictionsReal2 = netD2(inputs)
                lossDiscriminator2 = criterion(predictionsReal2, labels)  # labels = 1
                lossDiscriminator2.backward(retain_graph=True)
                predictionsFake2 = netD2(fakeImageBatch)
                lossFake2 = criterion(predictionsFake2, f_label)  # labels = 0
                lossFake2.backward(retain_graph=True)
                optD2.step()
            # optD1.zero_grad()
            # predictionsReal1 = netD1(inputs)
            # lossDiscriminator1 = criterion(predictionsReal1, labels)  # labels = 1
            # lossDiscriminator1.backward(retain_graph=True)
            # predictionsFake1 = netD1(fakeImageBatch)
            # lossFake1 = criterion(predictionsFake1, f_label)  # labels = 0
            # lossFake1.backward(retain_graph=True)
            # optD1.step()

            if(flag==2):
                optD.zero_grad()
                predictionsReal = netD(inputs)
                # lossDiscriminator = loss(predictionsReal, true_label)  # labels = 1
                # lossDiscriminator = criterion(predictionsReal, minortlabel)  # labels = 1
                lossDiscriminator = criterion(predictionsReal, labels)  # labels = 1
                lossDiscriminator.backward(retain_graph=True)
                predictionsFake = netD(fakeImageBatch)
                lossFake = criterion(predictionsFake, f_label)  # labels = 0
                lossFake.backward(retain_graph=True)
                optD.step()

                optD1.zero_grad()
                predictionsReal1 = netD1(inputs)
                lossDiscriminator1 = criterion(predictionsReal1, labels)  # labels = 1
                lossDiscriminator1.backward(retain_graph=True)
                predictionsFake1 = netD1(fakeImageBatch)
                lossFake1 = criterion(predictionsFake1, f_label)  # labels = 0
                lossFake1.backward(retain_graph=True)
                optD1.step()
            # optD2.zero_grad()
            # predictionsReal2 = netD2(inputs)
            # lossDiscriminator2 = criterion(predictionsReal2, labels)  # labels = 1
            # lossDiscriminator2.backward(retain_graph=True)
            # predictionsFake2 = netD2(fakeImageBatch)
            # lossFake2 = criterion(predictionsFake2, f_label)  # labels = 0
            # lossFake2.backward(retain_graph=True)
            # optD2.step()

            if(flag==None):
                optD.zero_grad()
                predictionsReal = netD(inputs)
                lossDiscriminator = criterion(predictionsReal, labels)  # labels = 1
                lossDiscriminator.backward(retain_graph=True)
                predictionsFake = netD(fakeImageBatch)
                lossFake = criterion(predictionsFake, f_label)  # labels = 0
                lossFake.backward(retain_graph=True)
                optD.step()

                optD1.zero_grad()
                predictionsReal1 = netD1(inputs)
                lossDiscriminator1 = criterion(predictionsReal1, labels)  # labels = 1
                lossDiscriminator1.backward(retain_graph=True)
                predictionsFake1 = netD1(fakeImageBatch)
                lossFake1 = criterion(predictionsFake1, f_label)  # labels = 0
                lossFake1.backward(retain_graph=True)
                optD1.step()

                optD2.zero_grad()
                predictionsReal2 = netD2(inputs)
                lossDiscriminator2 = criterion(predictionsReal2, labels)  # labels = 1
                lossDiscriminator2.backward(retain_graph=True)
                predictionsFake2 = netD2(fakeImageBatch)
                lossFake2 = criterion(predictionsFake2, f_label)  # labels = 0
                lossFake2.backward(retain_graph=True)
                optD2.step()

            # #
            optC.zero_grad()
            predictions = netC(inputs)
            realClassifierLoss = criterion(predictions, labels)
            realClassifierLoss.backward(retain_graph=True)
            optC.step()

            optC.zero_grad()
            predictionsFake = netC(fakeImageBatch)
            fakeClassificationloss = criterion(predictionsFake,f_label)
            fakeClassificationloss.backward(retain_graph=True)
            optC.step()
            #
            optG.zero_grad()
            predictionsFakeG = netD(fakeImageBatch)
            predictionsFakeG1 = netD1(fakeImageBatch)
            predictionsFakeG2 = netD2(fakeImageBatch)
            # lossGenerator = loss(predictionsFakeG, true_label)  # labels = 1
            lossGenerator = criterion(predictionsFakeG, labels)  # labels = 1
            lossGenerator1 = criterion(predictionsFakeG1, labels)  # labels = 1
            lossGenerator2 = criterion(predictionsFakeG2, labels)  # labels = 1
            # max(lossGenerator,lossGenerator1,lossGenerator2).backward()
            if(max(lossGenerator,lossGenerator1,lossGenerator2)==lossGenerator):
                flag = 0
                # lossGenerator1.backward(retain_graph=True)
                lossGenerator.backward()
                for param in netD.parameters():
                    param.requires_grad=False
                for param in netD1.parameters():
                    param.requires_grad=True
                for param in netD2.parameters():
                    param.requires_grad=True
            if (max(lossGenerator, lossGenerator1, lossGenerator2) == lossGenerator1):
                flag = 1
                # lossGenerator1.backward(retain_graph=True)
                lossGenerator1.backward()
                for param in netD.parameters():
                    param.requires_grad=True
                for param in netD1.parameters():
                    param.requires_grad=False
                for param in netD2.parameters():
                    param.requires_grad=True
            if (max(lossGenerator, lossGenerator1, lossGenerator2) == lossGenerator2):
                flag=2
                for param in netD.parameters():
                    param.requires_grad=True
                for param in netD1.parameters():
                    param.requires_grad=True
                for param in netD2.parameters():
                    param.requires_grad=False
                # lossGenerator.backward(retain_graph=True)
                lossGenerator2.backward()
            optG.step()

                # optC.zero_grad()
                # predictionsFake = netC(fakeImageBatch)
                # # get a tensor of the labels that are most likely according to model
                # predictedLabels = torch.argmax(predictionsFake, 1)  # -> [0 , 5, 9, 3, ...]
                # confidenceThresh = .2
                # probs = F.softmax(predictionsFake, dim=1)
                # mostLikelyProbs = np.asarray([probs[i, predictedLabels[i]].item() for i in range(len(probs))])
                # toKeep = mostLikelyProbs > confidenceThresh
                # if sum(toKeep) != 0:
                #     fakeClassifierLoss = criterion(predictionsFake[toKeep], predictedLabels[toKeep]) * advWeight
                #     fakeClassifierLoss.backward()
                # optC.step()

        print("Epoch " + str(epoch) + "Complete")
        train_end_time = time.time()

        # save gan image
        gridOfFakeImages = torchvision.utils.make_grid(fakeImageBatch.cpu())
        gridOfRealImages = torchvision.utils.make_grid(inputs.cpu())
        torchvision.utils.save_image(gridOfFakeImages,
                                     "./content/gridOfFakeImages/" + str(epoch) + '_' + str(batch_idx) + '.png')
        torchvision.utils.save_image(gridOfRealImages,
                                     "./content/gridOfRealImages/" + str(epoch) + '_' + str(batch_idx) + '.png')
        # if(epoch%10==0 and epoch >1000):
        if((epoch+1)%20==0 and epoch>199):
            test_start_time = time.time()
            netC.eval()
            test_loss = 0
            right = 0
            all_Label = []
            all_target = []
            for data, target in test_loader:
                indx_target = target.clone()
                data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                output = netC(data.cuda())

                test_loss += criterion(output.cuda(), target.cuda()).item()
                pred = output.max(1)[1]  # get the index of the max log-probability
                all_Label.extend(pred)
                all_target.extend(target)
                right += pred.cpu().eq(indx_target).sum()

            test_loss = test_loss / len(test_loader)  # average over number of mini-batch
            acc = float(100. * float(right)) / float(len(test_loader.dataset))
            print('\tTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
                test_loss, right, len(test_loader.dataset), acc))
            # C = confusion_matrix(target.data.cpu().numpy(), pred.cpu().numpy())
            C = confusion_matrix(all_target, all_Label)
            C = C[:16, :16]
            # C = C[:9, :9]
            np.save('c.npy', C)
            k = kappa(C, np.shape(C)[0]) * 100
            AA_ACC = np.diag(C) / np.sum(C, 1)
            AA = np.mean(AA_ACC, 0) * 100
            print('OA= %.5f AA= %.5f k= %.5f' % (acc, AA, k))
            test_end_time = time.time()
            train_time = train_time + (train_end_time - train_start_time)
            test_time = test_time + (test_end_time - test_start_time)
            group_logger.info('AA: %f, OA: %f, kappa: %f\n each_acc: %s' % (AA, acc, k, AA_ACC))
            group_logger.info("Train time:%s , Test time:%s", train_time, test_time)
            if acc > best_acc:
                state1 = {
                    'epoch': epoch + 1,
                    'state_dict': netC.state_dict(),
                    'acc': acc,
                    'best_acc': best_acc,
                    'optimizer': optC.state_dict(),
                }
                torch.save(state1, group_log_dir + "/best_modelC.pth.tar")
                best_acc = acc
                best_k = k
                best_aa = AA
                best_each_acc = AA_ACC
        # scheduler.step()
        # scheduler1.step()
        # scheduler2.step()
        # scheduler3.step()
        # scheduler4.step()
    group_logger.info(
        'best_AA: %f, best_OA: %f, best_kappa: %f\n each_acc: %s' % (best_aa, best_acc, best_k, best_each_acc))
    group_logger.info("all Train time:%s , all Test time:%s", train_time, test_time)
    oa_list.append(best_acc)
    aa_list.append(best_aa)
    kappa_list.append(best_k)
    each_acc_list.append(best_each_acc)
    # checkpoint = torch.load(group_log_dir + "/best_modelC.pth.tar")
    train_time_list.append(train_time)
    test_time_list.append(test_time)
    return oa_list, aa_list, kappa_list, each_acc_list, train_time_list, test_time_list

def kappa(testData, k):
    dataMat = np.mat(testData)
    P0 = 0.000
    for i in range(k):
        P0 += dataMat[i, i] * 1.0
    xsum = np.sum(dataMat, axis=1)
    ysum = np.sum(dataMat, axis=0)
    Pe = float(ysum * xsum) / np.sum(dataMat) ** 2
    P0 = float(P0 / np.sum(dataMat) * 1.0)
    cohens_coefficient = float((P0 - Pe) / (1 - Pe))
    return cohens_coefficient


time_str = datetime.strftime(datetime.now(), '%m-%d_%H-%M-%S')
log_path = os.path.join(os.getcwd(), "logs")  # logs目录
log_dir = os.path.join(log_path, time_str)  # log组根目录
oa_list = []
aa_list = []
kappa_list = []
each_acc_list = []
train_time_list = []
test_time_list = []
num_repeat = 10

####Training
class TrainDS(torch.utils.data.Dataset):
    def __init__(self):
        self.len = Xtrain.shape[0]
        self.x_data = torch.FloatTensor(Xtrain)
        self.y_data = torch.LongTensor(ytrain)
    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]
    def __len__(self):
        # 返回文件数据的数目
        return self.len

""" Testing dataset"""
class TestDS(torch.utils.data.Dataset):
    def __init__(self):
        self.len = Xtest.shape[0]
        self.x_data = torch.FloatTensor(Xtest)
        self.y_data = torch.LongTensor(ytest)
    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]
    def __len__(self):
        # 返回文件数据的数目
        return self.len
#
# data, labels, num_classes, _ = loadData(data_path, name, pca_components)
# HalfWidth = 16
# Wid = 2 * HalfWidth
# G = groundtruth[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth]
# data = pcdata[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth, :]
# [row, col] = G.shape
#
# NotZeroMask = np.zeros([row, col])
# Wid = 2 * HalfWidth
# NotZeroMask[HalfWidth + 1: -1 - HalfWidth + 1, HalfWidth + 1: -1 - HalfWidth + 1] = 1
# G = G * NotZeroMask
#
# [Row, Column] = np.nonzero(G)
# nSample = np.size(Row)
# RandPerm = np.random.permutation(nSample)
# # nTrain = 1000
# nTrain = 1000
# nTest = nSample - nTrain
#
# imdb = {}
# imdb['datas'] = np.zeros([2 * HalfWidth, 2 * HalfWidth, nBand, nTrain + nTest], dtype=np.float32)
# imdb['Labels'] = np.zeros([nTrain + nTest], dtype=np.int64)
# imdb['set'] = np.zeros([nTrain + nTest], dtype=np.int64)
# for iSample in range(nTrain + nTest):
#     xxx = data[Row[RandPerm[iSample]] - HalfWidth]
#     yy = Row[RandPerm[iSample]] + HalfWidth
#     www = Column[RandPerm[iSample]] - HalfWidth
#     zzz = Column[RandPerm[iSample]] + HalfWidth
#     imdb['datas'][:, :, :, iSample] = data[Row[RandPerm[iSample]] - HalfWidth: Row[RandPerm[iSample]] + HalfWidth,
#                                       Column[RandPerm[iSample]] - HalfWidth: Column[RandPerm[iSample]] + HalfWidth,
#                                       :]
#     imdb['Labels'][iSample] = G[Row[RandPerm[iSample]],
#     Column[RandPerm[iSample]]].astype(np.int64)
# print('Data is OK.')
#
# imdb['Labels'] = imdb['Labels'] - 1
#
# imdb['set'] = np.hstack((np.ones([nTrain]), 3 * np.ones([nTest]))).astype(np.int64)
# Xtrain = imdb['datas'][:, :, :, :nTrain]
# ytrain = imdb['Labels'][:nTrain]
# Xtest = imdb['datas']
# ytest = imdb['Labels']
# Xtrain = Xtrain.transpose(3, 2, 0, 1)
# Xtest = Xtest.transpose(3, 2, 0, 1)
# trainset = TrainDS()
# testset = TestDS()
# train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=0)
# test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size, shuffle=False, num_workers=0)

for iters in range(5):
    random_state = 1014 + 10
    train_loader, test_loader, val_loader, num_classes, n_bands = load_hyper(data_path, name, 17,
                                                                             False, train_samples, train_percent,
                                                                             100, components=3,
                                                                             rand_state=random_state)
    data, labels, num_classes, _ = loadData(data_path, name, pca_components)
    # data, labels, num_classes, _ = loadData(data_path, name, 3)

    fake_path = "./content/gridOfFakeImages/" + time_str + "/" + str(iters) + "/"
    real_path = "./content/gridOfRealImages/" + time_str + "/" + str(iters) + "/"
    print("==============================第", iters, "代========================================")

    netG = netGenerator(batch_size,batch_size,3)
    # netG = netGenerator(1)
    # netG = netGenerator(1)
    # netD = netDiscriminator(batch_size,3,10)
    netD = netDiscriminator(100,3)
    netD1 = netDiscriminator1(100,3)
    netD2 = netDiscriminator2(100,3)

    # netC = TGRS(5, 0.7, 17, 10, 3)
    # netC = TGRS(100, 0.7, 13, 16, 3)
    # netC = TGRS(3)
    netC = TGRS(5, 0.7, 17, 17, 3)
    netG.to(device)
    netD.to(device)
    netD1.to(device)
    netD2.to(device)
    netC.to(device)
    optD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.9, 0.999), weight_decay=1e-3)
    optD1 = optim.Adam(netD1.parameters(), lr=0.0002, betas=(0.9, 0.999), weight_decay=1e-3)
    optD2 = optim.Adam(netD2.parameters(), lr=0.0002, betas=(0.9, 0.999), weight_decay=1e-3)
    # optD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.9, 0.999), weight_decay=0.005)
    optG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.9, 0.999))
    # optG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.9, 0.999),weight_decay=0.02)
    optC = torch.optim.SGD(netC.parameters(), 0.0085, momentum=0.9, weight_decay=0.001,
                                nesterov=True)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optC, mode='min', factor=0.2, patience=3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optC, T_max=10)
    scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optG, T_max=10)
    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optD, T_max=10)
    scheduler3 = optim.lr_scheduler.CosineAnnealingLR(optD1, T_max=10)
    scheduler4 = optim.lr_scheduler.CosineAnnealingLR(optD2, T_max=10)

    group_log_dir = os.path.join(log_dir, "Experiment_" + str(iters + 1))  # logs组目录
    if not os.path.exists(group_log_dir):
        os.makedirs(group_log_dir)
    group_logger = get_logger(str(iters + 1), group_log_dir)
    train(train_loader)
    file.close()

stats_oa, stats_aa, stats_kappa, stats_each_acc, stats_train_time, \
    stats_test_time = stats(oa_list, aa_list, kappa_list, each_acc_list, train_time_list, test_time_list)
stats_logger = get_logger('final', log_dir)
stats_logger.info(
    '------------------------------------本组实验结果---------------------------------------------------')
stats_logger.info("OA均值:%f   总体标准差:%f   样本标准差:%f" %
                  (stats_oa['av_oa'], stats_oa['ov_std_oa'], stats_oa['samp_std_oa']))
stats_logger.info("AA均值:%f   总体标准差:%f   样本标准差:%f " %
                  (stats_aa['av_aa'], stats_aa['ov_std_aa'], stats_aa['samp_std_aa']))
stats_logger.info("kappa均值:%f  总体标准差:%f   样本标准差:%f" %
                  (stats_kappa['av_kappa'], stats_kappa['ov_std_kappa'], stats_kappa['samp_std_kappa']))

stats_logger.info("每类地物分类均值:      %s" % (stats_each_acc['av_each_acc']))
stats_logger.info("每类地物分类总体标准差:%s" % (stats_each_acc['ov_std_each_acc']))
stats_logger.info("每类地物分类样本标准差:%s" % (stats_each_acc['samp_std_each_acc']))
stats_logger.info(
    "训练时间均值:%f  总体标准差:%f  样本标准差:%f;  测试时间均值:%f  总体标准差:%f     样本标准差:%f" % (
        stats_train_time['av_train_time'], stats_train_time['ov_std_train_time'],
        stats_train_time['samp_std_train_time']
        , stats_test_time['av_test_time'], stats_test_time['ov_std_test_time'],
        stats_test_time['samp_std_test_time']))