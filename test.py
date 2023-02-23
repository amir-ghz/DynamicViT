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





# register forward hooks for recording and manipulating the middle value activations/heads
sparse_ls = []
activation = {}


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

        # Our qkv linear projection pruning strategy is based on removing outliers based on tokens' normal distribution.
        # Emprically, we observe that if we pick only the values within the one standard deviation distance of the distribution the accuracy does not drop.
        # We can work on a smarter solution later, but what we know for sure is that removing outliers in activation distributions works!  
        mean = torch.mean(output)
        std = torch.std(output, unbiased=False)
        #threshold = mean + 2*std
        threshold = 100000000
        for i in range (0, threshold):

            index = random.randint(0, torch.numel(output)-1)


            output.view(-1)[index] = output.view(-1)[index]*0

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


        #output = output*1000000000
        #output[abs(output) > threshold] = 0
        #print('tottal count is: ', torch.numel(output), ' # zeros: ', torch.numel(output) - int(torch.count_nonzero(output)))
        return output

def fn_quart(_, input, output):

        # Our qkv linear projection pruning strategy is based on removing outliers based on tokens' normal distribution.
        # Emprically, we observe that if we pick only the values within the one standard deviation distance of the distribution the accuracy does not drop.
        # We can work on a smarter solution later, but what we know for sure is that removing outliers in activation distributions works!  
        mean = torch.mean(output)
        std = torch.std(output, unbiased=False)
        #threshold = mean + 2*std
        threshold = 0.25
        output[abs(output) > threshold] = 0
        #print('tottal count is: ', torch.numel(output), ' # zeros: ', torch.numel(output) - int(torch.count_nonzero(output)))
        return

def fn_100(_, input, output):

        # Our qkv linear projection pruning strategy is based on removing outliers based on tokens' normal distribution.
        # Emprically, we observe that if we pick only the values within the one standard deviation distance of the distribution the accuracy does not drop.
        # We can work on a smarter solution later, but what we know for sure is that removing outliers in activation distributions works!  
        mean = torch.mean(output)
        std = torch.std(output, unbiased=False)
        #threshold = mean + 2*std
        threshold = 0.01
        output[abs(output) > threshold] = 0
        #print('tottal count is: ', torch.numel(output), ' # zeros: ', torch.numel(output) - int(torch.count_nonzero(output)))
        return



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

model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
model.eval()


# there are 12 Encoder layers in this Deit-base architecture and each encoder layer is named blocks[i].
# for example blocks[1] points to the second encoder layer
# in each encoder layer, there are:
# Block(
    #   (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
    #   (attn): Attention(
    #     (qkv): Linear(in_features=768, out_features=2304, bias=True) <------------*********------
    #     (attn_drop): Dropout(p=0.0, inplace=False)
    #     (proj): Linear(in_features=768, out_features=768, bias=True)
    #     (proj_drop): Dropout(p=0.0, inplace=False)
    #   )
    #   (drop_path): Identity()
    #   (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
    #   (mlp): Mlp(
    #     (fc1): Linear(in_features=768, out_features=3072, bias=True)
    #     (act): GELU(approximate=none)
    #     (fc2): Linear(in_features=3072, out_features=768, bias=True)
    #     (drop): Dropout(p=0.0, inplace=False)
    #   )
# 
# in this block we are interested in the token outputs after their linear projection which will be later converted into chunks (e.g., heads).
# notice that we have two layer normalizations in each block
# The heads carry out the V = Q@K multiplications which result in attention scores in parallel.
# Note that we are referring to the (qkv) layer as the target in the block above. The format to access this layer output is: 'blocks[i].attn.qkv'


correct = 0
total = 0



# model.cuda() ,,,, please run the model on CPU for now. This simulation is on the validation set of the ImageNet for only 512 images and does not require much time




with torch.no_grad():
    for i, (images, target) in enumerate(val_loader):

        model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True).to('cpu')

        images = images
        target = target



        model.blocks[0].attn.qkv.register_forward_hook(fn_half)


        # sparsity analysis: ########################################

        #model.blocks[0].attn.qkv.register_forward_hook(get_activation('blocks[0].attn.qkv'))

        # print('the siez is: ', activation['blocks[1].attn.qkv'].size())
        #h.remove()

        outputs = model(images)

        # att = temp#= activation['blocks[0].attn.qkv']

        # print("HEREEEEEEEEEEEEEEEEEEEEE", torch.numel(temp) - torch.count_nonzero(temp))

        # print(att.size())
        
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


import numpy as np 
import matplotlib.pyplot as plt 
  


X = ['1','2','3','4', '5', '6', '7', '8', '9', '10', '11', '12']

  
X_axis = np.arange(len(X))

print([row[3] for row in sparse_ls])
  
plt.bar(X_axis - 0.2, [row[1] for row in sparse_ls], 0.2, label = 'Query')
plt.bar(X_axis + 0.2, [row[2] for row in sparse_ls], 0.2, label = 'Key')
plt.bar(X_axis, [row[3] for row in sparse_ls], 0.2, label = 'Value')
  
plt.xticks(X_axis, X)
plt.xlabel("Heads")
plt.ylabel("Density (%)")
plt.title("Encoder Layer 6 Density")
plt.legend()
plt.show()

print(f'Accuracy of the network on the {512} test images: {100 * correct // total} %')

print(total)