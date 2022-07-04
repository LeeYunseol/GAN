# DCGAN
# 이 코드는 'Pytorch를 이용한 GAN 실제' 책에서 발췌해왔습니다.
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

import utils
#%%

# 출력 경로와 하이퍼 파라미터를 정의
CUDA = True
DATA_PATH = '~/Data/mnist'
OUT_PATH = 'C:/Users/hyunj/Desktop/GAN/output'
LOG_FILE = os.path.join(OUT_PATH, 'log.txt')
BATCH_SIZE = 128
IMAGE_CHANNEL = 1
Z_DIM = 100
G_HIDDEN = 64
X_DIM = 64
D_HIDDEN = 64
EPOCH_NUM = 300
REAL_LABEL = 1
FAKE_LABEL = 0
lr = 2e-4
seed = 1

#%%

# 네트워크를 만들기 전에 사전 준비

# 이 코드는 출력 폴더를 비우고 존재하지 않는 경우 생성하는거지만 오류 나서 제외
#utils.clear_folder(OUT_PATH)
print("Logging to {} \n".format(LOG_FILE))

# 모든 메시지를 print에서 로그 파일로 방향 재설정하고 동시에 이러한 메시지를 콘솔에 표시
# 또 오류나서 제외
#sys.stdout = utils.StdOut(LOG_FILE)
CUDA = CUDA and torch.cuda.is_available()
print("PyTorch version : {}".format(torch.__version__))
if CUDA :
    print("CUDA version : {}".format(torch.version.cuda))
if seed is None :
    seed = np.random.randint(1, 1000)
    
print("RANDOM SEED : ", seed)
np.random.seed(seed)
torch.manual_seed(seed)

if CUDA : 
    torch.cuda.manual_seed(seed)

cudnn.benchmark = True
device = torch.device("cuda:0" if CUDA else "cpu")
print("Device : {}".format(device))
#%%

# 생성기 네트워크
# 생성기는 input이 이미지가 아닌 latent variable이기 때문에 이를 image로 만들기 위해서 convTranspose가 필요
# 마지막의 채널 수는 이미지의 채널수인 3
# kernel, stride, padding을 모르겠다면 제가 정리한 CNN 자료를 참고!
class Generator(nn.Module) :
    # 파이썬 클래스 생성자 코드
    def __init__(self) :
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            #1st layer
            # First parameter = Channels of input 
            # Second parameter = Channels of output 
            # Third parameter = Kernel size 
            # Fourth parameter = stride 
            # fifth parameter = padding
            nn.ConvTranspose2d(Z_DIM, G_HIDDEN * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(G_HIDDEN * 8),
            nn.ReLU(True),
            
            #2nd layer
            nn.ConvTranspose2d(G_HIDDEN * 8, G_HIDDEN * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(G_HIDDEN * 4),
            nn.ReLU(True),
            
            #3rd layer
            nn.ConvTranspose2d(G_HIDDEN*4, G_HIDDEN * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(G_HIDDEN * 2),
            nn.ReLU(True),
            
            #4th layer
            nn.ConvTranspose2d(G_HIDDEN*2, G_HIDDEN, 4, 2, 1, bias=False),
            nn.BatchNorm2d(G_HIDDEN),
            nn.ReLU(True),
            
            # output layer
            nn.ConvTranspose2d(G_HIDDEN, IMAGE_CHANNEL, 4, 2, 1, bias=False),
            nn.Tanh()
            )
        
    def forward(self, input):
        return self.main(input)
#%%
# 네트워크 매개변수를 초기화하는 도우미 함수

def weights_init(m) :
    classname = m.__class__.__name__
    if classname.find('CONV') != -1 :
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
#%%
# Generator 객체 확인
netG = Generator().to(device)
netG.apply(weights_init)
print(netG)
#%%
# 판별기(D) 네트워크
# 판별기는 이미지 하나를 latent variable로 만들어서 이미지의 특징을 압
class Discriminator(nn.Module) :
    def __init__(self) :
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # 1st layer
            nn.Conv2d(IMAGE_CHANNEL, D_HIDDEN, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 2nd layer
            nn.Conv2d(D_HIDDEN, D_HIDDEN*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(D_HIDDEN * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 3rd layer
            nn.Conv2d(D_HIDDEN*2, D_HIDDEN*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(D_HIDDEN * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 4th layer
            nn.Conv2d(D_HIDDEN*4, D_HIDDEN*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(D_HIDDEN * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # output layer
            nn.Conv2d(D_HIDDEN*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            )
        
    def forward(self, input) :
        return self.main(input).view(-1, 1).squeeze(1)
#%%
# Discriminator 객체 생성
netD = Discriminator().to(device)
netD.apply(weights_init)
print(netD)

# 두개의 모델 모두 처음의 가중치 초기화는 논문을 참고
#%%
# 모델 및 학습 평가
# 이진 교차 엔트로피 손실 함수
criterion = nn.BCELoss()

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))

optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
#%%
# MNIST 데이터 셋을 GPUT 메모리에 로딩

dataset = dset.MNIST(root=DATA_PATH, download=True,
                     transform=transforms.Compose([
                         transforms.Resize(X_DIM),
                         transforms.ToTensor(),
                         transforms.Normalize((0.5,), (0.5,))])
                     )
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size = BATCH_SIZE, shuffle=True, num_workers=0)
#%%

# 학습과 반복

viz_noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1, device=device)
for epoch in range(EPOCH_NUM) :
    for i, data in enumerate(dataloader):
        x_real = data[0].to(device)
        # torch.full(size, fill_value, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) 
        # → Tensor fill_value 로 채워진 크기 size 의 텐서를 만듭니다
        real_label = torch.full((x_real.size(0),), REAL_LABEL, device=device)
        fake_label = torch.full((x_real.size(0),), FAKE_LABEL, device=device)
        
    # update D with real data -> 먼저 판별기를 실제 데이터에 대해 업데이트
    netD.zero_grad()
    # y_real은 판별기가 예측한 것 
    y_real = netD(x_real)
    y_real = y_real.type(torch.FloatTensor).cuda()
    real_label = real_label.type(torch.FloatTensor).cuda()
    loss_D_real = criterion(y_real, real_label)
    # 판별기가 예측한 것과 실제 레이블에 대해서 역전
    loss_D_real.backward()
    
    # update D with fake data -> 판별기를 가짜 데이터에 대해 업데이트
    # Noise 부여
    z_noise = torch.randn(x_real.size(0), Z_DIM, 1, 1, device = device)
    # Generator가 만든 이미지가 x_fake
    x_fake = netG(z_noise)
    # detach : 이를 호출하여 연산 기록으로부터 분리하여 이후 연산들이 추적되는 것을 방지
    y_fake = netD(x_fake.detach())
    y_fake = y_fake.type(torch.FloatTensor).cuda()
    fake_label = fake_label.type(torch.FloatTensor).cuda()
    loss_D_fake = criterion(y_fake, fake_label)
    # 역전파 단계: 모델의 매개변수들에 대한 손실의 변화도를 계산합니다.
    loss_D_fake.backward()
    # optimizer의 step 함수를 호출하면 매개변수가 갱신됩니다.
    optimizerD.step()
    
    # update G with fake data
    netG.zero_grad()
    y_fake_r = netD(x_fake)
    y_fake_r = y_fake_r.type(torch.FloatTensor).cuda()
    loss_G = criterion(y_fake_r, real_label)
    # 역전파 단계: 모델의 매개변수들에 대한 손실의 변화도를 계산합니다.
    loss_G.backward()
    # optimizer의 step 함수를 호출하면 매개변수가 갱신됩니다.
    optimizerG.step()
    
    print('EPOCH {} [{}/{}] loss_D_real : {:.4f} loss_D_fake : {:.4f} \
              loss_G : {:.4f}'.format(
              epoch, i, len(dataloader),
              loss_D_real.mean().item(),
              loss_D_fake.mean().item(),
              loss_G.mean().item()
              ))
    # 10 epoch일 때마다 샘플 시각화
    if i % 10 == 0:
        with torch.no_grad() :
            viz_sample = netG(viz_noise)
            vutils.save_image(viz_sample, os.path.join(OUT_PATH, 'fake_samples_{}.png'.format(epoch)), 
                              normize=True)