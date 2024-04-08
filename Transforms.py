from torchvision import transforms
from MyData import MyData
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader


writer = SummaryWriter("logs")
trans_compose = transforms.Compose([transforms.Resize([256,256]),
                                    transforms.ToTensor()])
root_dir = "archive/train"
label_dir = "apple"
mydataset = MyData(root_dir,label_dir,trans_compose)
dataloader = DataLoader(mydataset,4)
step = 0
for data in dataloader:  
    imgs,label = data
    print(type(imgs))
    writer.add_images("totensorrr",imgs,step)
    step = step+1

writer.close()
