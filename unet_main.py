import torch


class Resnet(torch.nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.time = torch.nn.Sequential(
            torch.nn.SiLU(),
            torch.torch.nn.Linear(1280, dim_out),
            torch.nn.Unflatten(dim=1, unflattened_size=(dim_out, 1, 1)),
        )

        self.s0 = torch.nn.Sequential(
            torch.torch.nn.GroupNorm(num_groups=32,
                                     num_channels=dim_in,
                                     eps=1e-05,
                                     affine=True),
            torch.nn.SiLU(),
            torch.torch.nn.Conv2d(dim_in,
                                  dim_out,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1),
        )

        self.s1 = torch.nn.Sequential(
            torch.torch.nn.GroupNorm(num_groups=32,
                                     num_channels=dim_out,
                                     eps=1e-05,
                                     affine=True),
            torch.nn.SiLU(),
            torch.torch.nn.Conv2d(dim_out,
                                  dim_out,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1),
        )

        self.res = None
        if dim_in != dim_out:
            self.res = torch.torch.nn.Conv2d(dim_in,
                                             dim_out,
                                             kernel_size=1,
                                             stride=1,
                                             padding=0)

    def forward(self, x, time):
        #x -> [1, 320, 64, 64]
        #time -> [1, 1280]

        res = x

        #[1, 1280] -> [1, 640, 1, 1]
        time = self.time(time)

        #[1, 320, 64, 64] -> [1, 640, 32, 32]
        x = self.s0(x) + time

        #维度不变
        #[1, 640, 32, 32]
        x = self.s1(x)

        #[1, 320, 64, 64] -> [1, 640, 32, 32]
        if self.res:
            res = self.res(res)

        #维度不变
        #[1, 640, 32, 32]
        x = res + x

        return x


#################################resnet
class CrossAttention(torch.nn.Module):

    def __init__(self, dim_q, dim_kv):
        #dim_q -> 320
        #dim_kv -> 512

        super().__init__()

        self.dim_q = dim_q

        self.q = torch.nn.Linear(dim_q, dim_q, bias=False)
        self.k = torch.nn.Linear(dim_kv, dim_q, bias=False)
        self.v = torch.nn.Linear(dim_kv, dim_q, bias=False)

        self.out = torch.nn.Linear(dim_q, dim_q)

    def forward(self, q, kv):
        #x -> [1, 4096, 320]
        #kv -> [1, 77, 512]

        #[1, 4096, 320] -> [1, 4096, 320]
        q = self.q(q)
        #[1, 30, 512] -> [1, 30, 320]
        k = self.k(kv)
        #[1, 30, 512] -> [1, 30, 320]
        v = self.v(kv)

        def reshape(x):
            #x -> [1, 4096, 320]
            b, lens, dim = x.shape

            #[1, 4096, 320] -> [1, 4096, 8, 40]
            x = x.reshape(b, lens, 8, dim // 8)

            #[1, 4096, 8, 40] -> [1, 8, 4096, 40]
            x = x.transpose(1, 2)

            #[1, 8, 4096, 40] -> [8, 4096, 40]
            x = x.reshape(b * 8, lens, dim // 8)

            return x

        #[1, 4096, 320] -> [8, 4096, 40]
        q = reshape(q)
        #[1, 30, 320] -> [8, 307, 40]
        k = reshape(k)
        #[1, 30, 320] -> [8, 30, 40]
        v = reshape(v)

        #[8, 4096, 40] * [8, 40, 30] -> [8, 4096, 30]
        #atten = q.bmm(k.transpose(1, 2)) * (self.dim_q // 8)**-0.5

        #从数学上是等价的,但是在实际计算时会产生很小的误差
        atten = torch.baddbmm(
            torch.empty(q.shape[0], q.shape[1], k.shape[1], device=q.device),
            q,
            k.transpose(1, 2),
            beta=0,
            alpha=(self.dim_q // 8)**-0.5,
        )

        atten = atten.softmax(dim=-1)

        #[8, 4096, 30] * [8, 30, 40] -> [8, 4096, 40]
        atten = atten.bmm(v)

        def reshape(x):
            #x -> [8, 4096, 40]
            b, lens, dim = x.shape

            #[8, 4096, 40] -> [1, 8, 4096, 40]
            x = x.reshape(b // 8, 8, lens, dim)

            #[1, 8, 4096, 40] -> [1, 4096, 8, 40]
            x = x.transpose(1, 2)

            #[1, 4096, 320]
            x = x.reshape(b // 8, lens, dim * 8)

            return x

        #[8, 4096, 40] -> [1, 4096, 320]
        atten = reshape(atten)

        #[1, 4096, 320] -> [1, 4096, 320]
        atten = self.out(atten)

        return atten
##################################cross attention   
class Transformer(torch.nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.dim = dim

        #in
        self.norm_in = torch.nn.GroupNorm(num_groups=32,
                                          num_channels=dim,
                                          eps=1e-6,
                                          affine=True)
        self.cnn_in = torch.nn.Conv2d(dim,
                                      dim,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0)

        #atten
        self.norm_atten0 = torch.nn.LayerNorm(dim, elementwise_affine=True)
        self.atten1 = CrossAttention(dim, dim)
        self.norm_atten1 = torch.nn.LayerNorm(dim, elementwise_affine=True)
        self.atten2 = CrossAttention(dim, 512)

        #act
        self.norm_act = torch.nn.LayerNorm(dim, elementwise_affine=True)
        self.fc0 = torch.nn.Linear(dim, dim * 8)
        self.act = torch.nn.GELU()
        self.fc1 = torch.nn.Linear(dim * 4, dim)

        #out
        self.cnn_out = torch.nn.Conv2d(dim,
                                       dim,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0)

    def forward(self, q, kv):
        #q -> [1, 320, 64, 64]
        #kv -> [1, 30, 512]
        b, _, h, w = q.shape
        res1 = q

        #----in----
        #维度不变
        #[1, 320, 64, 64]
        q = self.cnn_in(self.norm_in(q))

        #[1, 320, 64, 64] -> [1, 64, 64, 320] -> [1, 4096, 320]
        q = q.permute(0, 2, 3, 1).reshape(b, h * w, self.dim)

        #----atten----
        #维度不变
        #[1, 4096, 320]
        q = self.atten1(q=self.norm_atten0(q), kv=self.norm_atten0(q)) + q
        q = self.atten2(q=self.norm_atten1(q), kv=kv) + q

        #----act----
        #[1, 4096, 320]
        res2 = q

        #[1, 4096, 320] -> [1, 4096, 2560]
        q = self.fc0(self.norm_act(q))

        #1280
        d = q.shape[2] // 2

        #[1, 4096, 1280] * [1, 4096, 1280] -> [1, 4096, 1280]
        q = q[:, :, :d] * self.act(q[:, :, d:])

        #[1, 4096, 1280] -> [1, 4096, 320]
        q = self.fc1(q) + res2

        #----out----
        #[1, 4096, 320] -> [1, 64, 64, 320] -> [1, 320, 64, 64]
        q = q.reshape(b, h, w, self.dim).permute(0, 3, 1, 2).contiguous()

        #维度不变
        #[1, 320, 64, 64]
        q = self.cnn_out(q) + res1

        return q
###############################transformer
class DownBlock(torch.nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.tf0 = Transformer(dim_out)
        self.res0 = Resnet(dim_in, dim_out)

        self.tf1 = Transformer(dim_out)
        self.res1 = Resnet(dim_out, dim_out)

        self.out = torch.nn.Conv2d(dim_out,
                                   dim_out,
                                   kernel_size=3,
                                   stride=2,
                                   padding=1)

    def forward(self, out_vae, out_encoder, time):
        outs = []

        out_vae = self.res0(out_vae, time)
        out_vae = self.tf0(out_vae, out_encoder)
        outs.append(out_vae)

        out_vae = self.res1(out_vae, time)
        out_vae = self.tf1(out_vae, out_encoder)
        outs.append(out_vae)

        out_vae = self.out(out_vae)
        outs.append(out_vae)

        return out_vae, outs
##############################down block
class UpBlock(torch.nn.Module):

    def __init__(self, dim_in, dim_out, dim_prev, add_up):
        super().__init__()

        self.res0 = Resnet(dim_out + dim_prev, dim_out)
        self.res1 = Resnet(dim_out + dim_out, dim_out)
        self.res2 = Resnet(dim_in + dim_out, dim_out)

        self.tf0 = Transformer(dim_out)
        self.tf1 = Transformer(dim_out)
        self.tf2 = Transformer(dim_out)

        self.out = None
        if add_up:
            self.out = torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=2, mode='nearest'),
                torch.nn.Conv2d(dim_out, dim_out, kernel_size=3, padding=1),
            )

    def forward(self, out_vae, out_encoder, time, out_down):
        out_vae = self.res0(torch.cat([out_vae, out_down.pop()], dim=1), time)
        out_vae = self.tf0(out_vae, out_encoder)

        out_vae = self.res1(torch.cat([out_vae, out_down.pop()], dim=1), time)
        out_vae = self.tf1(out_vae, out_encoder)

        out_vae = self.res2(torch.cat([out_vae, out_down.pop()], dim=1), time)
        out_vae = self.tf2(out_vae, out_encoder)

        if self.out:
            out_vae = self.out(out_vae)

        return out_vae
###########################up block
class UNet(torch.nn.Module):

    def __init__(self):
        super().__init__()

        #in
        self.in_vae = torch.nn.Conv2d(4, 320, kernel_size=3, padding=1)

        self.in_time = torch.nn.Sequential(
            torch.nn.Linear(320, 1280),
            torch.nn.SiLU(),
            torch.nn.Linear(1280, 1280),
        )

        #down
        self.down_block0 = DownBlock(320, 320)
        self.down_block1 = DownBlock(320, 640)
        self.down_block2 = DownBlock(640, 1280)

        self.down_res0 = Resnet(1280, 1280)
        self.down_res1 = Resnet(1280, 1280)

        #mid
        self.mid_res0 = Resnet(1280, 1280)
        self.mid_tf = Transformer(1280)
        self.mid_res1 = Resnet(1280, 1280)

        #up
        self.up_res0 = Resnet(2560, 1280)
        self.up_res1 = Resnet(2560, 1280)
        self.up_res2 = Resnet(2560, 1280)

        self.up_in = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='nearest'),
            torch.nn.Conv2d(1280, 1280, kernel_size=3, padding=1),
        )

        self.up_block0 = UpBlock(640, 1280, 1280, True)
        self.up_block1 = UpBlock(320, 640, 1280, True)
        self.up_block2 = UpBlock(320, 320, 640, False)

        #out
        self.out = torch.nn.Sequential(
            torch.nn.GroupNorm(num_channels=320, num_groups=32, eps=1e-5),
            torch.nn.SiLU(),
            torch.nn.Conv2d(320, 4, kernel_size=3, padding=1),
        )

    def forward(self, out_vae, out_encoder, time):
        #out_vae -> [1, 4, 64, 64]
        #out_encoder -> [1, 30, 512]
        #time -> [1]

        #----in----
        #[1, 4, 64, 64] -> [1, 320, 64, 64]
        out_vae = self.in_vae(out_vae)

        def get_time_embed(t):
            #-9.210340371976184 = -math.log(10000)
            e = torch.arange(160) * -9.210340371976184 / 160
            e = e.exp().to(t.device) * t

            #[160+160] -> [320] -> [1, 320]
            e = torch.cat([e.cos(), e.sin()]).unsqueeze(dim=0)

            return e

        #[1] -> [1, 320]
        time = get_time_embed(time)
        #[1, 320] -> [1, 1280]
        time = self.in_time(time)

        #----down----
        #[1, 320, 64, 64]
        #[1, 320, 64, 64]
        #[1, 320, 64, 64]
        #[1, 320, 32, 32]
        #[1, 640, 32, 32]
        #[1, 640, 32, 32]
        #[1, 640, 16, 16]
        #[1, 1280, 16, 16]
        #[1, 1280, 16, 16]
        #[1, 1280, 8, 8]
        #[1, 1280, 8, 8]
        #[1, 1280, 8, 8]
        out_down = [out_vae]

        #[1, 320, 64, 64],[1, 30, 512],[1, 1280] -> [1, 320, 32, 32]
        #out -> [1, 320, 64, 64],[1, 320, 64, 64][1, 320, 32, 32]
        out_vae, out = self.down_block0(out_vae=out_vae,
                                        out_encoder=out_encoder,
                                        time=time)
        out_down.extend(out)

        #[1, 320, 32, 32],[1, 30, 512],[1, 1280] -> [1, 640, 16, 16]
        #out -> [1, 640, 32, 32],[1, 640, 32, 32],[1, 640, 16, 16]
        out_vae, out = self.down_block1(out_vae=out_vae,
                                        out_encoder=out_encoder,
                                        time=time)
        out_down.extend(out)

        #[1, 640, 16, 16],[1, 30, 512],[1, 1280] -> [1, 1280, 8, 8]
        #out -> [1, 1280, 16, 16],[1, 1280, 16, 16],[1, 1280, 8, 8]
        out_vae, out = self.down_block2(out_vae=out_vae,
                                        out_encoder=out_encoder,
                                        time=time)
        out_down.extend(out)

        #[1, 1280, 8, 8],[1, 1280] -> [1, 1280, 8, 8]
        out_vae = self.down_res0(out_vae, time)
        out_down.append(out_vae)

        #[1, 1280, 8, 8],[1, 1280] -> [1, 1280, 8, 8]
        out_vae = self.down_res1(out_vae, time)
        out_down.append(out_vae)

        #----mid----
        #[1, 1280, 8, 8],[1, 1280] -> [1, 1280, 8, 8]
        out_vae = self.mid_res0(out_vae, time)

        #[1, 1280, 8, 8],[1, 30, 512] -> [1, 1280, 8, 8]
        out_vae = self.mid_tf(out_vae, out_encoder)

        #[1, 1280, 8, 8],[1, 1280] -> [1, 1280, 8, 8]
        out_vae = self.mid_res1(out_vae, time)

        #----up----
        #[1, 1280+1280, 8, 8],[1, 1280] -> [1, 1280, 8, 8]
        out_vae = self.up_res0(torch.cat([out_vae, out_down.pop()], dim=1),
                               time)

        #[1, 1280+1280, 8, 8],[1, 1280] -> [1, 1280, 8, 8]
        out_vae = self.up_res1(torch.cat([out_vae, out_down.pop()], dim=1),
                               time)

        #[1, 1280+1280, 8, 8],[1, 1280] -> [1, 1280, 8, 8]
        out_vae = self.up_res2(torch.cat([out_vae, out_down.pop()], dim=1),
                               time)

        #[1, 1280, 8, 8] -> [1, 1280, 16, 16]
        out_vae = self.up_in(out_vae)

        #[1, 1280, 16, 16],[1, 30, 512],[1, 1280] -> [1, 1280, 32, 32]
        #out_down -> [1, 640, 16, 16],[1, 1280, 16, 16],[1, 1280, 16, 16]
        out_vae = self.up_block0(out_vae=out_vae,
                                 out_encoder=out_encoder,
                                 time=time,
                                 out_down=out_down)

        #[1, 1280, 32, 32],[1, 30, 512],[1, 1280] -> [1, 640, 64, 64]
        #out_down -> [1, 320, 32, 32],[1, 640, 32, 32],[1, 640, 32, 32]
        out_vae = self.up_block1(out_vae=out_vae,
                                 out_encoder=out_encoder,
                                 time=time,
                                 out_down=out_down)

        #[1, 640, 64, 64],[1, 30, 512],[1, 1280] -> [1, 320, 64, 64]
        #out_down -> [1, 320, 64, 64],[1, 320, 64, 64],[1, 320, 64, 64]
        out_vae = self.up_block2(out_vae=out_vae,
                                 out_encoder=out_encoder,
                                 time=time,
                                 out_down=out_down)

        #----out----
        #[1, 320, 64, 64] -> [1, 4, 64, 64]
        out_vae = self.out(out_vae)

        return out_vae

