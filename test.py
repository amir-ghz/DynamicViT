from PIL import Image
import torch
import timm
import requests
import torchvision.transforms as transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
model.eval()
import torch
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

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


correct = 0
total = 0



#model.cuda()

activation = {}

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def fn(_, input, output):
        #output[0][:] = 0
        output[abs(output) > 10] = 0
        #print('tottal count is: ', torch.numel(output), ' # zeros: ', torch.numel(output) - int(torch.count_nonzero(output)))
        return


with torch.no_grad():
    for i, (images, target) in enumerate(val_loader):

        #model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True).to('cpu')

        images = images
        target = target

        model.blocks[11].attn.qkv.register_forward_hook(fn)

        ########################################################## sparsity analysis ########################################

        h = model.blocks[11].attn.qkv.register_forward_hook(get_activation('blocks[11].attn.qkv'))

        # print('the siez is: ', activation['blocks[11].attn.qkv'].size())

        #h.remove()

        outputs = model(images)

        att = activation['blocks[11].attn.qkv']

        B, N, C = 512, 197, 768

        qkv = att.reshape(B, N, 3, 12, C // 12).permute(2, 0, 3, 1, 4)

        q, k, v = qkv.unbind(0) 

        for i in range (0, 12):
            print(f'for head {i} tottal count is: ', torch.numel(torch.chunk(q, 12, 1)[i]), ' # zeros: ', torch.numel(torch.chunk(q, 12, 1)[i]) - int(torch.count_nonzero(torch.chunk(q, 12, 1)[i])), "ratio: ", ((torch.numel(torch.chunk(q, 12, 1)[i]) - int(torch.count_nonzero(torch.chunk(q, 12, 1)[i])))/6455296)*100)
            print(f'for head {i} tottal count is: ', torch.numel(torch.chunk(k, 12, 1)[i]), ' # zeros: ', torch.numel(torch.chunk(k, 12, 1)[i]) - int(torch.count_nonzero(torch.chunk(k, 12, 1)[i])), "ratio: ", ((torch.numel(torch.chunk(k, 12, 1)[i]) - int(torch.count_nonzero(torch.chunk(k, 12, 1)[i])))/6455296)*100)
            print(f'for head {i} tottal count is: ', torch.numel(torch.chunk(v, 12, 1)[i]), ' # zeros: ', torch.numel(torch.chunk(v, 12, 1)[i]) - int(torch.count_nonzero(torch.chunk(v, 12, 1)[i])), "ratio: ", ((torch.numel(torch.chunk(v, 12, 1)[i]) - int(torch.count_nonzero(torch.chunk(v, 12, 1)[i])))/6455296)*100)
            print("******************************************************************************")
            #print(torch.chunk(q, 12, 1)[0].size())

        
        
        _, predicted = torch.max(outputs, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

        break



print(f'Accuracy of the network on the {512} test images: {100 * correct // total} %')

print(total)