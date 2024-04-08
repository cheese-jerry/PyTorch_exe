from MyData import MyData
from torch.utils.tensorboard import SummaryWriter

root_dir = "archive/test"
label_dir = "apple"
mydataset = MyData(root_dir,label_dir)
img,label = mydataset[8]
#img.show()

'''
writer = SummaryWriter("logs") 
for i in range(100):
    writer.add_scalar("y=x",i,i)

writer.close()
writer.add_image("test",img,dataformat='HWC')

'''
#tensorboard --log dir="logs"
