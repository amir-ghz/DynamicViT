from PIL import Image
import torch
import timm
import requests
import torchvision.transforms as transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torch
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import random



sparse_ls = []
net_ls = []
activation = {}

def spaerse_calc(output):

    sparse_ls = []
        
    print("HEREEEEE", torch.numel(output) - int(torch.count_nonzero(output)))
    # B: batch size, N: sequence size, C: token size
    # batch is 512, 
    # sequence size is 14*14+1 = 197 --> patch size = 14 and +1 for cls embeddings
    # token size is: (H*W/P**2)*channel_size = (224*224/14**2)*3 = 768 
    
    B, N, C = 512, 197, 768

    # the following q, k and v retrieval is based on timm official implementation:

    qkv = output.reshape(B, N, 3, 12, C // 12).permute(2, 0, 3, 1, 4)

    q, k, v = qkv.unbind(0) 


    # print sparsity analysis in each head

    for i in range (0, 12):
        print("Density ratio: ", ((int(torch.count_nonzero(torch.chunk(q, 12, 1)[i])))/6455296)*100)
        print("Density ratio: ", ((int(torch.count_nonzero(torch.chunk(k, 12, 1)[i])))/6455296)*100)
        print("Density ratio: ", ((int(torch.count_nonzero(torch.chunk(v, 12, 1)[i])))/6455296)*100)
        sparse_ls.append([i, ((int(torch.count_nonzero(torch.chunk(q, 12, 1)[i])))/6455296)*100, ((int(torch.count_nonzero(torch.chunk(k, 12, 1)[i])))/6455296)*100, ((int(torch.count_nonzero(torch.chunk(v, 12, 1)[i])))/6455296)*100])
        
        print("******************************************************************************")

    net_ls.append(sparse_ls)
    #output = output*1000000000
    #output[abs(output) > threshold] = 0
    #print('tottal count is: ', torch.numel(output), ' # zeros: ', torch.numel(output) - int(torch.count_nonzero(output)))



def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

        threshold = 100000
        for i in range (0, threshold):

            index = random.randint(0, torch.numel(output)-1)

            output.view(-1)[index] = 0

        return output
    return hook

def fn_half(_, input, output):

        output[abs(output) < 0.5] = 0

        spaerse_calc(output)

        return output

def fn_one(_, input, output):
    

        output[abs(output) < 1] = 0

        spaerse_calc(output)

        return output



normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


# Creating validation set for ImageNet dataset

val_dataset = datasets.ImageFolder(
    'IMG/val/',
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))


val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=512, shuffle=False,
    num_workers=8, pin_memory=True, sampler=None)

# downlaoding the Deit model 

model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True).cuda()
model.eval()


correct = 0
total = 0




with torch.no_grad():
    for i, (images, target) in enumerate(val_loader):

        model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True).cuda()

        images = images.cuda()
        target = target.cuda()



        model.blocks[0].attn.qkv.register_forward_hook(fn_one)
        model.blocks[1].attn.qkv.register_forward_hook(fn_one)
        model.blocks[2].attn.qkv.register_forward_hook(fn_one)
        model.blocks[3].attn.qkv.register_forward_hook(fn_half)
        model.blocks[4].attn.qkv.register_forward_hook(fn_half)
        model.blocks[5].attn.qkv.register_forward_hook(fn_half)
        model.blocks[6].attn.qkv.register_forward_hook(fn_half)
        model.blocks[7].attn.qkv.register_forward_hook(fn_half)
        model.blocks[8].attn.qkv.register_forward_hook(fn_half)
        model.blocks[9].attn.qkv.register_forward_hook(fn_half)
        model.blocks[10].attn.qkv.register_forward_hook(fn_one)
        model.blocks[11].attn.qkv.register_forward_hook(fn_half)


        #model.blocks[0].attn.qkv.register_forward_hook(get_activation('blocks[0].attn.qkv'))

        # print('the siez is: ', activation['blocks[1].attn.qkv'].size())
        #h.remove()

        outputs = model(images).cuda()

        # att = temp#= activation['blocks[0].attn.qkv']

        # # B: batch size, N: sequence size, C: token size
        # # batch is 512, 
        # # sequence size is 14*14+1 = 197 --> patch size = 14 and +1 for cls embeddings
        # # token size is: (H*W/P**2)*channel_size = (224*224/14**2)*3 = 768 
        
        # B, N, C = 512, 197, 768

        # # the following q, k and v retrieval is based on timm official implementation:

        # qkv = att.reshape(B, N, 3, 12, C // 12).permute(2, 0, 3, 1, 4)

        # q, k, v = qkv.unbind(0) 


        # # print sparsity analysis in each head

        # for i in range (0, 12):
        #    print("Density ratio: ", ((int(torch.count_nonzero(torch.chunk(q, 12, 1)[i])))/6455296)*100)
        #    print("Density ratio: ", ((int(torch.count_nonzero(torch.chunk(k, 12, 1)[i])))/6455296)*100)
        #    print("Density ratio: ", ((int(torch.count_nonzero(torch.chunk(v, 12, 1)[i])))/6455296)*100)
        #    sparse_ls.append([i, ((int(torch.count_nonzero(torch.chunk(q, 12, 1)[i])))/6455296)*100, ((int(torch.count_nonzero(torch.chunk(k, 12, 1)[i])))/6455296)*100, ((int(torch.count_nonzero(torch.chunk(v, 12, 1)[i])))/6455296)*100])
            
        #    print("******************************************************************************")

        
        
        _, predicted = torch.max(outputs, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()


        break


# import numpy as np 
# import matplotlib.pyplot as plt 
  


# X = ['1','2','3','4', '5', '6', '7', '8', '9', '10', '11', '12']

  
# X_axis = np.arange(len(X))

# print([row[3] for row in sparse_ls])
  
# plt.bar(X_axis - 0.2, [row[1] for row in sparse_ls], 0.2, label = 'Query')
# plt.bar(X_axis + 0.2, [row[2] for row in sparse_ls], 0.2, label = 'Key')
# plt.bar(X_axis, [row[3] for row in sparse_ls], 0.2, label = 'Value')
  
# plt.xticks(X_axis, X)
# plt.xlabel("Heads")
# plt.ylabel("Density (%)")
# plt.title("Encoder Layer 6 Density")
# plt.legend()
# plt.show()

print(f'Accuracy of the network on the {512} test images: {100 * correct // total} %')

print(total)

torch.save(net_ls, 'temp.pt')
print(len(net_ls))