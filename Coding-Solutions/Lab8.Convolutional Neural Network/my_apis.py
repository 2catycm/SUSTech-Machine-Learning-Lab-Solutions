from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim

device_ids = list(range(torch.cuda.device_count()))
# device_ids
train_device = torch.device('cuda', device_ids[0]) # 改成2会报错，因为默认主卡是0
# test_device = torch.device('cuda', device_ids[1])
test_device = torch.device('cuda', device_ids[0]) #我服了，pytorch真实垃圾

history = {'Train Loss':[],'Test Loss':[],'Test Accuracy':[]}

def prepare(model):
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    model = model.to(train_device)
    return model
def test(test_loader, net, critiria, test_device=test_device):
    correct,totalLoss = 0,0
    totalSize = 0
    net.train(False)
    critiria = critiria.to(test_device)
    for testImgs,labels in test_loader:
        testImgs = testImgs.to(test_device)
        labels = labels.to(test_device)
        outputs = net(testImgs)
        loss = critiria(outputs,labels)
        predictions = torch.argmax(outputs,dim = 1)
        totalSize += labels.size(0)
        totalLoss += loss
        correct += torch.sum(predictions == labels)
    testAccuracy = correct/totalSize
    testLoss = totalLoss/len(test_loader)
    return testLoss,testAccuracy
def train(train_loader, net, critiria, optimizer, test_loader, n_epochs=5):
    for epoch in range(1, n_epochs + 1):
        try:
            #构建tqdm进度条
            processBar = tqdm(train_loader,unit = 'step')  
            #打开网络的训练模式
            net.train(True)
            critiria = critiria.to(train_device) # 这个critiria会多管闲事，检查数据在不在同一个卡
            
            #开始对训练集的DataLoader进行迭代
            totalTrainLoss = 0.0
            for step,(trainImgs,labels) in enumerate(processBar): # 对一个可以迭代的对象增加进度条
                #将图像和标签传输进device中
                trainImgs = trainImgs.to(train_device)
                labels = labels.to(train_device)
                
                #清空模型的梯度
                optimizer.zero_grad()
                
                #对模型进行前向推理
                outputs = net(trainImgs)
                
                #计算本轮推理的Loss值
                loss = critiria(outputs,labels)
                #计算本轮推理的准确率
                predictions = torch.argmax(outputs, dim = 1)
                accuracy = torch.sum(predictions == labels)/labels.shape[0]
                
                #进行反向传播求出模型参数的梯度
                loss.backward()
                #使用迭代器更新模型权重
                optimizer.step()
                    #将本step结果进行可视化处理
                processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f" % 
                                            (epoch,n_epochs,loss.item(),accuracy.item()))
                totalTrainLoss+= loss
                if step == len(processBar)-1:
                # if True:
                    testLoss, testAccuracy = test(test_loader, net, critiria)
                    trainLoss = totalTrainLoss/len(train_loader)
                    history['Train Loss'].append(trainLoss.item())
                    history['Test Loss'].append(testLoss.item())
                    history['Test Accuracy'].append(testAccuracy.item())
                    processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f, Test Loss: %.4f, Test Acc: %.4f" % 
                                        (epoch,n_epochs,loss.item(),accuracy.item(),testLoss.item(),testAccuracy.item()))
                processBar.close()
                
        except Exception as e:
            print(e)
        finally:
            print('训练已经中断')
            torch.save(net.state_dict(), f'./checkpoints/cifar_resnet_epoch{epoch}_step{step}.pth')

                