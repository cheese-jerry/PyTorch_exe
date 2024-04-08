from MyData import MyData
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from torch import nn
import torch
import os

train_data_path = "archive/train"
#train_data_label_dir_lis = os.listdir(train_data_root_dir)
test_data_path = "archive/test"

test_data_label_dir_lis = os.listdir("archive/test")
print(len(test_data_label_dir_lis))

trans = transforms.Compose([transforms.Resize([32,32]),
                                    transforms.ToTensor()])

train_datas = datasets.ImageFolder(train_data_path,trans)
test_datas = datasets.ImageFolder(test_data_path,trans)


train_data_loader = DataLoader(train_datas,64)
test_data_loader = DataLoader(test_datas,64)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel,self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4,64),
            nn.Linear(64,36)
        )

    def forward(self,x):
        x = self.model(x)
        return x

netmodel = MyModel()

#损失函数
loss_fun = nn.CrossEntropyLoss()

#优化器
learning_rate = 0.001
optimizer = torch.optim.SGD(netmodel.parameters(),lr = learning_rate)

total_train_step = 0
total_test_step = 0

epoch = 10

for i in range(epoch):
    print("-------epoch is {} ----------".format(i+1))
    #训练
    for data in train_data_loader:
        img,label = data
        output = netmodel(img)
        loss = loss_fun(output,label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step+1

        print("train step is {},loss is {}".format(total_train_step,loss))

    
    #测试
    total_test_loss = 0
    with torch.no_grad():
        for data in test_data_loader:
            imgs,label = data
            output = netmodel(imgs)
            loss = loss_fun(output,label)
            total_test_loss = loss + total_test_loss
    print("test loss is {}".format(total_test_loss))        