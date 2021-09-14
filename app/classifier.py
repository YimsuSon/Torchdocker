from torch import nn
from torchvision import transforms, models

def create_network():
    net = models.resnet18()

    fc_input_dim = net.fc.in_features
    net.fc = nn.Linear(fc_input_dim,2)
    return net 



class Classifier(object):
    def __init__(self, params):
        self.net = create_network()
        self.net.load_state_dict(params)
        self.net.eval()
        self.transformer = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])

        self.classes = ["burrito","taco"]

    def predict(self,img):
        x = self.transformer(img)
        x = x.unsqueeze(0)
        out = self.net(x)
        out = out.max(1)[1].item()
        return self.classes[out]
    