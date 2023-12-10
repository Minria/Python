import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 定义生成器模型
class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes, img_size):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.img_size = img_size
        self.label_embedding = nn.Embedding(num_classes, latent_dim)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, img_size**2),
            nn.Tanh()
        )
    def forward(self, noise, labels):
        gen_input = torch.cat((self.label_embedding(labels), noise), dim=1)
        img = self.model(gen_input)
        img = img.view(img.size(0), 1, self.img_size, self.img_size)
        return img

# 定义判别器模型
class Discriminator(nn.Module):
    def __init__(self, num_classes, img_size):
        super(Discriminator, self).__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.label_embedding = nn.Embedding(num_classes, img_size**2)
        self.model = nn.Sequential(
            nn.Linear(img_size**2 + num_classes, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, img, labels):
        img = img.view(img.size(0), -1)
        dis_input = torch.cat((img, self.label_embedding(labels)), dim=1)
        validity = self.model(dis_input)
        return validity

# 设置超参数
latent_dim = 100
num_classes = 10
img_size = 28
batch_size = 64
lr = 0.0002
num_epochs = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 创建生成器和判别器实例
generator = Generator(latent_dim, num_classes, img_size).to(device)
discriminator = Discriminator(num_classes, img_size).to(device)

# 定义损失函数和优化器
adversarial_loss = nn.BCELoss()
generator_optimizer = optim.Adam(generator.parameters(), lr=lr)
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=lr)

# 加载MNIST数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 图像归一化至[-1, 1]
])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# 训练循环
for epoch in range(num_epochs):
    for batch_idx, (real_images, labels) in enumerate(train_loader):
        batch_size = real_images.size(0)
        
        # 将真实图像和标签转换为张量并将其移至设备上
        real_images = real_images.to(device)
        labels = labels.to(device)
        
        # 创建目标标签，用于生成器和判别器的训练
        valid = torch.ones(batch_size, 1).to(device)
        fake = torch.zeros(batch_size, 1).to(device)
        # ---------------------
        #  训练判别器
        # ---------------------
        discriminator_optimizer.zero_grad()
        # 生成器生成假样本
        noise = torch.randn(batch_size, latent_dim).to(device)
        gen_images = generator(noise, labels)
        # 判别器评估真样本和假样本的输出
        real_validity = discriminator(real_images, labels)
        fake_validity = discriminator(gen_images.detach(), labels)
        # 计算判别器的损失
        real_loss = adversarial_loss(real_validity, valid)
        fake_loss = adversarial_loss(fake_validity, fake)
        discriminator_loss = (real_loss + fake_loss) / 2
        # 反向传播和参数更新
        discriminator_loss.backward()
        discriminator_optimizer.step()
        # ---------------------
        #  训练生成器
        # ---------------------
        generator_optimizer.zero_grad()
        # 生成器生成假样本
        gen_images = generator(noise, labels)
        # 判别器评估假样本的输出
        fake_validity = discriminator(gen_images, labels)
        
        # 计算生成器的损失
        generator_loss = adversarial_loss(fake_validity, valid)
        
        # 反向传播和参数更新
        generator_loss.backward()
        generator_optimizer.step()
        
        # 输出训练过程中的损失和其他指标
        if (batch_idx + 1) % 200 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], "
                  f"Generator Loss: {generator_loss.item():.4f}, Discriminator Loss: {discriminator_loss.item():.4f}")
        
    # 在每个epoch结束后，保存生成器的权重并生成一些样本图像
    torch.save(generator.state_dict(), f'generator_epoch_{epoch+1}.pth')
    with torch.no_grad():
        # 生成一些样本图像
        sample_noise = torch.randn(num_classes, latent_dim).to(device)
        sample_labels = torch.arange(num_classes).to(device)
        sample_images = generator(sample_noise, sample_labels).unsqueeze(1)
        sample_images = sample_images * 0.5 + 0.5  # 反归一化至[0, 1]
        torchvision.utils.save_image(sample_images, f'sample_images_epoch_{epoch+1}.png', nrow=num_classes)