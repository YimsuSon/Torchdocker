import torch
from smart_getenv import getenv
from app import create_app
from app.classifier import Classifier

prm_file = getenv("PRM_FILE",default="taco_burrito.prm")
params = torch.load(prm_file, map_location=lambda storage, loc: storage)
classifier = Classifier(params)
app = create_app(classifier)



