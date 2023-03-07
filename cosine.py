from PIL import Image
import torch
import torch as nn
import timm
import requests
import torchvision.transforms as transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torch
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import random
import xlwt
from xlwt import Workbook

# order matters in hooking!

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
    

        # torch.Size([512, 197, 768])

        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        
        print(cos(output[50][2], output[50][3]))
        print(output[0][0].size())







        return output



normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])





# Creating validation set for ImageNet dataset

val_dataset = datasets.ImageFolder(
    '../deit/IMG/val/',
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))


train_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=512, shuffle=False,
    num_workers=8, pin_memory=True, sampler=None)

# downlaoding the Deit model 

model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True).cuda()
model.eval()


correct = 0
total = 0



# print(model)

qkvS = ['blocks.0.attn.qkv.weight', 'blocks.1.attn.qkv.weight', 'blocks.2.attn.qkv.weight', 'blocks.3.attn.qkv.weight', 'blocks.4.attn.qkv.weight',
        'blocks.5.attn.qkv.weight', 'blocks.6.attn.qkv.weight']

projection = ['blocks.0.attn.proj.weight', 'blocks.1.attn.proj.weight', 'blocks.2.attn.proj.weight', 'blocks.3.attn.proj.weight', 'blocks.4.attn.proj.weight',
              'blocks.5.attn.proj.weight', 'blocks.6.attn.proj.weight']

for name, params in model.named_parameters():
    #print(name, params.data.size())
    if name in qkvS: # blocks.1.attn.proj.weight     and      blocks.10.attn.qkv.weight
        
        print(params.data.size())
        x = torch.transpose(params.data, 0, 1)
        #x = params.data
        print(x[0].size())

        ##############################################33333 excel block:

        # # Workbook is created
        # wb = Workbook()
        # # add_sheet is used to create sheet.
        # sheet1 = wb.add_sheet('Sheet 1')

        # sheet1.write(0, 0, name)
        # sheet1.write(0, 1, "weight index")
        # sheet1.write(0, 2, "CosineSim(i-1, i+1)")

        count = 0
         
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        for i in range(0, len(x)):
            for j in range(0, len(x)):

                if i != j:
                # #print("{:.3f}".format(cos(params.data[i], params.data[i+1]).item()))
                    simil = (cos(x[i], x[j]).item())#((cos(x[i], x[i-1]).item() + cos(x[i], x[i+1]).item())/2)
                    if simil > 0.5:
                        count+=1
                #     #print(i ,"{:.3f}".format(simil))
                #     count+=1
                #     #print("trueeee", (cos(params.data[i], params.data[i-1]).item()))
                # # sheet1.write(i, 1, str(i))
                # # sheet1.write(i, 2, "{:.3f}".format(simil))


                # x[i] = x[i+2]
                # x[i+1] = x[i+2]
                # x[i+3] = x[i+2]
                # x[i+4] = x[i+2]


        print(count)

        model.eval()



  
        # wb.save('cosine_simil.xls')

# blocks.0.mlp.fc1.weight torch.Size([3072, 768])
with torch.no_grad():
    for i, (images, target) in enumerate(train_loader):


        #model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True).cuda()

        images = images.cuda()
        target = target.cuda()



        #model.blocks[0].mlp.fc2.register_forward_hook(fn_one)


        outputs = model(images).cuda()

        
        
        _, predicted = torch.max(outputs, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()


        break



print(f'Accuracy of the network on the {512} test images: {100 * correct // total} %')

print(total)

# torch.save(net_ls, 'temp.pt')
# print(len(net_ls))
