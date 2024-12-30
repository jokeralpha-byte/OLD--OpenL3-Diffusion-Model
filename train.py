from diffusers import DiffusionPipeline
import torch
import os
import numpy as np
import audio_utils as au
import audiocore
import core
import warnings
import torchvision
from PIL import Image
import numpy as np
from openl3model import OpenL3Embedding
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from unet_main import UNet

hop_size=1
device = 'cuda' if torch.cuda.is_available() else 'cpu'

pipeline = DiffusionPipeline.from_pretrained(
    'D:/huggingface/hub/models--lansinuote--diffsion_from_scratch.params/snapshots/310d9345e14b3b625635041dd573676c008d83ea', safety_checker=None)

scheduler = pipeline.scheduler
tokenizer = pipeline.tokenizer

del pipeline
################存取训练的data
sounds=[]
picdata=[]

#######################################encoder#############
encoder = OpenL3Embedding(input_repr='mel256', embedding_size=512, content_type="music")
###########################################################
dirpath="C:/Users/Lenovo/Desktop/test"
for filename in os.listdir(dirpath):
    for file_inside in os.listdir(dirpath+"/"+filename):
        if file_inside.endswith(".mp3"):
           
            audio=au.load_audio_file(dirpath+"/"+filename+"/"+file_inside)
            audiocore._check_audio_types(audio)
            if audiocore._is_zero(audio):
                warnings.warn(f'provided audio array is all zeros')
            audio = audiocore.downmix(audio)

            # split audio into overlapping windows as dictated by hop_size
            hop_len: int = hop_size * core.SAMPLE_RATE
            audio = audiocore.window(audio, window_len=hop_size*core.SAMPLE_RATE, hop_len=hop_len)
            index=np.arange(0,audio.shape[0])
            r_index=np.random.choice(index,30)
            audio_sample=audio[r_index]
            sounds.append(audio_sample)
        elif file_inside.endswith(".flac"):
            
            audio=au.load_audio_file(dirpath+"/"+filename+"/"+file_inside)
            
            audiocore._check_audio_types(audio)
            if audiocore._is_zero(audio):
                warnings.warn(f'provided audio array is all zeros')
            audio = audiocore.downmix(audio)

            # split audio into overlapping windows as dictated by hop_size
            hop_len: int = hop_size * core.SAMPLE_RATE
            audio = audiocore.window(audio, window_len=hop_size*core.SAMPLE_RATE, hop_len=hop_len)
            index=np.arange(0,audio.shape[0])
            r_index=np.random.choice(index,30)
            audio_sample=audio[r_index]
            sounds.append(audio_sample)
        elif file_inside.endswith(".jpg"):
            img = Image.open(dirpath+"/"+filename+"/"+file_inside)
            # 如果图像不是RGB格式，转换为RGB（即使是灰度图也会转换成3通道RGB）
            img = img.convert('RGB')
            # 将PIL图像转换为NumPy数组
            img_array = np.array(img)
            
            
            
            img_resized = np.array(Image.fromarray(img_array).resize((512, 512)))
            img_resized_array = img_resized.transpose(2, 0, 1)
            img_normalized = img_resized_array / 255.0
            img_normalized = 2 * img_normalized - 1
            # 调整图像大小为512x512
            picdata.append(img_normalized)

pixel_data=[torch.from_numpy(pic.astype(np.float32)).to(device) for pic in picdata]
input_data=[encoder(torch.from_numpy(sound)).detach().to(device) for sound in sounds]
###########################数据集#####################


class MyDataset(Dataset):
    def __init__(self, pixel_data, input_data):
        self.pixel_data = pixel_data
        self.input_data = input_data

    def __len__(self):
        return len(self.pixel_data)

    def __getitem__(self, idx):
        pixel_values = self.pixel_data[idx]
        input_ids = self.input_data[idx]
        return {'pixel_values': pixel_values, 'input_ids': input_ids}
traindataset = MyDataset(pixel_data, input_data)
dataloader = DataLoader(traindataset, batch_size=1, shuffle=True, num_workers=0)
#############################loadvae#################################
from vae_main import VAE
from diffusers import AutoencoderKL

#加载预训练模型的参数
params = AutoencoderKL.from_pretrained(
    'D:/huggingface/hub/models--lansinuote--diffsion_from_scratch.params/snapshots/310d9345e14b3b625635041dd573676c008d83ea', subfolder='vae')

vae = VAE()


def load_res(model, param):
    model.s[0].load_state_dict(param.norm1.state_dict())
    model.s[2].load_state_dict(param.conv1.state_dict())
    model.s[3].load_state_dict(param.norm2.state_dict())
    model.s[5].load_state_dict(param.conv2.state_dict())

    if isinstance(model.res, torch.nn.Module):
        model.res.load_state_dict(param.conv_shortcut.state_dict())

'''def load_custom_state_dict(model, state_dict):
    # 手动映射参数名称
    new_state_dict = {}
    for k, v in state_dict.items():
        if k == '0.weight':
            new_state_dict['out.weight'] = v
        elif k == '0.bias':
            new_state_dict['out.bias'] = v
        else:
            new_state_dict[k] = v

    # 加载状态字典
    model.load_state_dict(new_state_dict, strict=False)'''
def load_atten(model, param):
    model.norm.load_state_dict(param.group_norm.state_dict())
    model.q.load_state_dict(param.to_q.state_dict())
    model.k.load_state_dict(param.to_k.state_dict())
    model.v.load_state_dict(param.to_v.state_dict())
    model.out.load_state_dict(param.to_out.state_dict())


#encoder.in
vae.encoder[0].load_state_dict(params.encoder.conv_in.state_dict())

#encoder.down
for i in range(4):
    load_res(vae.encoder[i + 1][0], params.encoder.down_blocks[i].resnets[0])
    load_res(vae.encoder[i + 1][1], params.encoder.down_blocks[i].resnets[1])

    if i != 3:
        vae.encoder[i + 1][2][1].load_state_dict(
            params.encoder.down_blocks[i].downsamplers[0].conv.state_dict())

#encoder.mid
load_res(vae.encoder[5][0], params.encoder.mid_block.resnets[0])
load_res(vae.encoder[5][2], params.encoder.mid_block.resnets[1])
load_atten(vae.encoder[5][1], params.encoder.mid_block.attentions[0])

#encoder.out
vae.encoder[6][0].load_state_dict(params.encoder.conv_norm_out.state_dict())
vae.encoder[6][2].load_state_dict(params.encoder.conv_out.state_dict())

#encoder.正态分布层
vae.encoder[7].load_state_dict(params.quant_conv.state_dict())

#decoder.正态分布层
vae.decoder[0].load_state_dict(params.post_quant_conv.state_dict())

#decoder.in
vae.decoder[1].load_state_dict(params.decoder.conv_in.state_dict())

#decoder.mid
load_res(vae.decoder[2][0], params.decoder.mid_block.resnets[0])
load_res(vae.decoder[2][2], params.decoder.mid_block.resnets[1])
load_atten(vae.decoder[2][1], params.decoder.mid_block.attentions[0])

#decoder.up
for i in range(4):
    load_res(vae.decoder[i + 3][0], params.decoder.up_blocks[i].resnets[0])
    load_res(vae.decoder[i + 3][1], params.decoder.up_blocks[i].resnets[1])
    load_res(vae.decoder[i + 3][2], params.decoder.up_blocks[i].resnets[2])

    if i != 3:
        vae.decoder[i + 3][4].load_state_dict(
            params.decoder.up_blocks[i].upsamplers[0].conv.state_dict())

#decoder.out
vae.decoder[7][0].load_state_dict(params.decoder.conv_norm_out.state_dict())
vae.decoder[7][2].load_state_dict(params.decoder.conv_out.state_dict())

###########################################################################
unet=UNet()
encoder.requires_grad_(False)
vae.requires_grad_(False)
unet.requires_grad_(True)

encoder.eval()
vae.eval()
unet.train()

encoder.to(device)
vae.to(device)
unet.to(device)

optimizer = torch.optim.AdamW(unet.parameters(),
                              lr=1e-5,
                              betas=(0.9, 0.999),
                              weight_decay=0.01,
                              eps=1e-8)

criterion = torch.nn.MSELoss()

def get_loss(data):
    with torch.no_grad():
        
        out_encoder =data['input_ids']
        
        out_vae = vae.encoder(data['pixel_values'])
        out_vae = vae.sample(out_vae)

        #0.18215 = vae.config.scaling_factor
        out_vae = out_vae * 0.18215

    #随机数,unet的计算目标
    noise = torch.randn_like(out_vae)

    #往特征图中添加噪声
    #1000 = scheduler.num_train_timesteps
    #1 = batch size
    noise_step = torch.randint(0, 1000, (1, )).long().to(device)
    out_vae_noise = scheduler.add_noise(out_vae, noise, noise_step)

    #根据文字信息,把特征图中的噪声计算出来
    out_unet = unet(out_vae=out_vae_noise,
                    out_encoder=out_encoder,
                    time=noise_step)

    #计算mse loss
    #[1, 4, 64, 64],[1, 4, 64, 64]
    return criterion(out_unet, noise)
def train(epochs):
    loss_sum = 0
    for epoch in range(epochs):
        for i, data in enumerate(dataloader):
            loss = get_loss(data) / 4
            loss.backward()
            loss_sum += loss.item()

            if (epoch * len(dataloader) + i) % 4 == 0:
                torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        if epoch % 10 == 0:
            print(epoch, loss_sum)
            loss_sum = 0
        torch.save(unet, 'unet.pth')
train(40)