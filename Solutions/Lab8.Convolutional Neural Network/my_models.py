from torch import nn
# import torch.functional as F #错，注意两个functional的区别
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super().__init__()
        # 连续的2个卷积层
        self.longcut = nn.Sequential(
            # 这个在论文里面是 3x3卷积核
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True), # 这个relu是模块。inplace=True表示直接在原来的内存上修改，不再开辟新的内存
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential() 
        if stride == 1 and inchannel == outchannel:
            self.shortcut = nn.Sequential() # 输入输出通道一致，不需要做任何操作
        else:    
            #shortcut，这里为了跟2个卷积层的结果结构一致，要做处理
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
            
    def forward(self, x):
        out = self.longcut(x)
        #将2个卷积层的输出跟处理过的x相加，实现ResNet的基本结构
        out = out + self.shortcut(x)
        out = F.relu(out)
        
        return out
    
class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
        nn.Conv2d(3,16,kernel_size=3,stride=3,padding=0),
        nn.BatchNorm2d(16)
        )
        self.blk1 = ResBlock(16,32)
        self.blk2 = ResBlock(32,64)
        self.blk3 = ResBlock(64,128)
        self.blk4 = ResBlock(128,256)
        self.outlayer = nn.Linear(256*10*10,10)
        
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)
        x = x.view(x.size(0),-1)
        x = self.outlayer(x)
        return x
