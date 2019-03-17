from models import Generator,nz
import torch
from PIL import ImageEnhance
from torchvision import transforms
#
path = 'model/poke_generator.pt'
Gmodel = Generator(0)
Gmodel.load_state_dict(torch.load(path))
#
noise = torch.randn(1, nz, 1, 1)
img = Gmodel(noise)

# for better model convergence in train we use normalized images so generator normalized image
# it look better after unnormalization

transform = transforms.Compose([transforms.Normalize((0, 0, 0), (1/0.229, 1/0.224, 1/0.225)),
                                  transforms.Normalize((-0.485, -0.456, -0.406), (1 , 1, 1)),
                                  transforms.ToPILImage()
                        ])
img = img[0]
img = transform(img)
img = ImageEnhance.Contrast(img).enhance(2.0)
img.save('pokemon.png')
