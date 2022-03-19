from PIL import Image
import torch
from timm.models import create_model
import pvt, pvt_v2
from torch.utils.data import DataLoader,Dataset
import pandas as pd
from torchvision import transforms
import numpy as np

#arg_model = 'pvt_v2_b0'
#arg_reload = './checkpoints/pvt_v2_b0/pvt_v2_b0_95.pth'
arg_model = 'pvt_tiny'
arg_reload = './checkpoints/pvt_tiny/pvt_tiny_30.pth'
device = 'cuda'


class testDataset(Dataset): #different from train dataset, because the data organized in submission.csv is different from train.csv
    
    def __init__(self,transform=None):
        self.test_df = pd.read_csv('./dataset/sample_submission.csv')#pandas用来读取csv文件
        self.transform = transform
        
    def __getitem__(self,index):
        #data in submission.csv:
        #       img_pair               is_related
        #face05508.jpg-face01210.jpg       0
        #face05820.jpg-face03938.jpg       0
        
        img0_path = self.test_df.iloc[index].img_pair.split("-")[0]
        img1_path = self.test_df.iloc[index].img_pair.split("-")[1]
        #print(img0_path,'-',img1_path) #reserved to check whether test data is in order.
        
        img0 = Image.open('./dataset/test/'+img0_path)
        img1 = Image.open('./dataset/test/'+img1_path)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        
        return img0, img1
    
    def __len__(self):
        return len(self.test_df)

print(f"Creating model: {arg_model}")
model = create_model(
    arg_model,
    pretrained=False,
    num_classes=2,
)

if arg_reload:
    checkpoint = torch.load(arg_reload, map_location='cpu')

    if 'model' in checkpoint:
        checkpoint_model = checkpoint['model']
    else:
        checkpoint_model = checkpoint
    state_dict = model.state_dict()

    for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias', 'mcp.weight']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]
        model.load_state_dict(checkpoint_model, strict=False)

model.to(device)



testset = testDataset(transform=transforms.Compose([transforms.Resize((224,224)),
                                                                      transforms.ToTensor()
                                                                      ]))
testloader = DataLoader(testset,
                        shuffle=False,
                        num_workers=0,
                        batch_size=1)

test_df = pd.read_csv('./dataset/sample_submission.csv')#pandas用来读取csv文件
predictions=[]
with torch.no_grad():
    for data in testloader:
        img0, img1 = data
        img0, img1 = img0.cuda(), img1.cuda()
        outputs = model(img0,img1)
        _, predicted = torch.max(outputs, 1)
        predictions = np.concatenate((predictions,predicted.cpu().numpy()),0)#taking care of here, the output data format is important for transfer
        
test_df['is_related'] = predictions
test_df.to_csv("submission.csv", index=False)#submission.csv should be placed directly in current fold.
test_df.head(50)#show the result to be committed