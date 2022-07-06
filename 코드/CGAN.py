# CGAN 

import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs("images", exist_ok=True)
# argparse : 파이썬 인자값 추가하기 
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
# latent_dim : the number of noises
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False
print("CUDA : {}".format(cuda))


'''
클래스(class) 형태의 모델은 nn.Module 을 상속받습니다. 
그리고 __init__()에서 모델의 구조와 동작을 정의하는 생성자를 정의합니다. 
이는 파이썬에서 객체가 갖는 속성값을 초기화하는 역할로, 객체가 생성될 때 자동으호 호출됩니다. 
super() 함수를 부르면 여기서 만든 클래스는 nn.Module 클래스의 속성들을 가지고 초기화 됩니다. 
foward() 함수는 모델이 학습데이터를 입력받아서 forward 연산을 진행시키는 함수입니다. 
이 forward() 함수는 model 객체를 데이터와 함께 호출하면 자동으로 실행이됩니다. 
예를 들어 model이란 이름의 객체를 생성 후, model(입력 데이터)와 같은 형식으로 객체를 호출하면 자동으로 forward 연산이 수행됩니다.
'''
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        # noise와 label을 결합할 용도인 label embedding matrix을 생성 (10, 10)
        self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes)
        # 임베딩 파라미터를 선언한 후 forward 메소드를 수행하면 (입력차원, 임베딩차원) 크기를 가진 텐서가 출력
        # 이때 forward 메소드의 입력텐서는 임베딩 벡터를 추출할 범주의 인덱스이므로 무조건 정수타입(LongTensor)이 들어가야된다.
        # ex) forward 메소드에 (2,4) 크기의 텐서가 입력으로 들어가면 (2,4,10) 크기의 텐서가 출력

        

        self.model = nn.Sequential(
            # 첫 레이이어는 110 -> 책 참고  
            *self.create_layer(opt.latent_dim + opt.n_classes, 128, normalize=False),
            *self.create_layer(128, 256),
            *self.create_layer(256, 512),
            *self.create_layer(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )
    def create_layer(self, in_feat, out_feat, normalize=True):
        layers = [nn.Linear(in_feat, out_feat)]
        if normalize:
            layers.append(nn.BatchNorm1d(out_feat, 0.8))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers
    
    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Discriminator또한 y를 함께 임베딩
        self.label_embedding = nn.Embedding(opt.n_classes, opt.n_classes)
        
        # 모델 선언 부분을 더 깔끔하게 정리하였음
        self.model = nn.Sequential(
            *self._create_layer(opt.n_classes + int(np.prod(img_shape)), 1024, False, True),
            *self._create_layer(1024, 512, True, True),
            *self._create_layer(512, 256, True, True), #사실 이렇게 안해도 됨
            *self._create_layer(256, 128, False, False),
            *self._create_layer(128, 1, False, False),
            nn.Sigmoid()
        )
    
    def _create_layer(self, size_in, size_out, drop_out=True, act_func=True) :
        layers = [nn.Linear(size_in, size_out)]
        if drop_out == True :
            layers.append(nn.Dropout(0.4))
        if act_func == True :
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers   
    
    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity


# Loss functions
adversarial_loss = torch.nn.MSELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Configure data loader
os.makedirs("../../data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../../data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = generator(z, labels)
    save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True)


# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):

        batch_size = imgs.shape[0]

        # Adversarial ground truths
        # Variable = Tensor
        # valid는 1로, fake는 0으로 
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))

        # Generate a batch of images
        gen_imgs = generator(z, gen_labels) # 가짜 이미지를 GAN을 통해서 생성

        # Loss measures generator's ability to fool the discriminator
        # Discriminator에 가짜 이미지를 넣어서 결과를 출력. 가짜가 label과 같다고 하면 1, 아니면 0
        validity = discriminator(gen_imgs, gen_labels)
        # discriminator에서 나온 output을 1로 채워져 있는 label과 비교
        # 만약 discriminator가 fake image를 label과 같다고 판단해서 1을 출력했다면 loss는 0에 가까워짐
        # generator는 discriminator를 잘 속이는 방향(validity가 1이 나오게 학습)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        validity_real = discriminator(real_imgs, labels)
        # Discriminator가 real image를 real이라고 예측하면 d_real_loss는 0이 된다.
        d_real_loss = adversarial_loss(validity_real, valid)

        # Loss for fake images
        # gen_imgs는 Generator가 만든 fake
        validity_fake = discriminator(gen_imgs.detach(), gen_labels)
        # Discriminator가 진짜 이미지가 아닌 Generator가 생성한 가짜 이미지를 진짜라고 예측했을 떄 d_fake_loss는 커짐
        d_fake_loss = adversarial_loss(validity_fake, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            sample_image(n_row=10, batches_done=batches_done)