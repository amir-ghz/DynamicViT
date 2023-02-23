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
from binary_utils import float2bit, bit2float
import random



# register forward hooks for recording and manipulating the middle value activations/heads

activation = {}

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def fn(_, input, output):

        # Our qkv linear projection pruning strategy is based on removing outliers based on tokens' normal distribution.
        # Emprically, we observe that if we pick only the values within the one standard deviation distance of the distribution the accuracy does not drop.
        # We can work on a smarter solution later, but what we know for sure is that removing outliers in activation distributions works!  
        mean = torch.mean(output)
        std = torch.std(output, unbiased=False)
        threshold = mean + std
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


backend = "fbgemm" # replaced with qnnpack causing much worse inference speed for quantized model on this notebook
model.qconfig = torch.quantization.get_default_qconfig(backend)
torch.backends.quantized.engine = backend

quantized_model = torch.quantization.quantize_dynamic(model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)
scripted_quantized_model = torch.jit.script(quantized_model)
scripted_quantized_model.save("fbdeit_scripted_quantized.pt")


scripted_model = torch.jit.script(model)
scripted_model.save("fbdeit_scripted.pt")


scripted_quantized_model.to('cpu')
scripted_quantized_model.eval()




correct = 0
total = 0


model.cuda()

for name, params in scripted_quantized_model.named_parameters():
    if name == 'blocks.11.norm1.weight':
        print('HIIIIIIIIII')
        binary_tensor = float2bit(params.data, num_e_bits=8, num_m_bits=23, bias=127.)
    
        ber = 100

        for i in range(0, ber):
            idx = random.randint(0, (torch.numel(binary_tensor))-1)
            
            if binary_tensor.view(-1,)[idx] == float(0):
                binary_tensor.view(-1,)[idx] = float(1)
            else:
                binary_tensor.view(-1,)[idx] = float(0)

        float_tensor = bit2float(binary_tensor, num_e_bits=8, num_m_bits=23, bias=127.)

        with torch.no_grad():
            params.data = float_tensor

scripted_quantized_model.eval()

with torch.no_grad():
    for i, (images, target) in enumerate(val_loader):

        #model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True).to('cpu')

        images = images
        target = target

        outputs = scripted_quantized_model(images)

        
        _, predicted = torch.max(outputs, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

        break



print(f'Accuracy of the network on the {512} test images: {100 * correct // total} %')

print(total)    