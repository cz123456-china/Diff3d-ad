import torch
import torch.nn.functional as F
from data_loader import MVTecDRAEMTestDataset, MVTecDRAEMTrainDataset
from torch.utils.data import DataLoader
from torch import optim
import os
import numpy as np
from dsr_model import SubspaceRestrictionModule, ImageReconstructionNetwork, AnomalyDetectionModule
from discrete_model import DiscreteLatentModel
from loss import FocalLoss
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from au_pro_util import calculate_au_pro
from sklearn.metrics import roc_auc_score, average_precision_score
import math
from math import exp

def cosine_annealing(epoch, total_epochs, min_d=0.99, max_d=0.999):
    # 计算余弦退火后的d值
    return min_d + (max_d - min_d) * (1 + math.cos(math.pi * epoch / total_epochs)) / 2

def dynamic_loss(d, loss_tensor):
    # 计算d-quantile
    #loss_tensor_flat = loss_tensor.view(-1)
    #quantile_value = torch.quantile(loss_tensor_flat, d)
    quantile_values = []
    for i in range(loss_tensor.size(0)):  # 遍历每个批次
        # 将当前批次的图像展平成一维，再计算分位数
        loss_tensor_flat = loss_tensor[i].view(-1)
        quantile_value = torch.quantile(loss_tensor_flat, d)
        quantile_values.append(quantile_value)

# 计算所有批次 quantile 的平均值（如果需要）
    quantile_value = torch.mean(torch.tensor(quantile_values))
    
    # 找出大于d-quantile的部分
    dynamic_loss_elements = loss_tensor[loss_tensor > quantile_value]
    
    # 计算这些部分的平均值作为动态损失
    return dynamic_loss_elements.mean() if dynamic_loss_elements.numel() > 0 else torch.tensor(0.0, device=loss_tensor.device)


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()
def create_window(window_size, channel=1, sigma=1.5):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(
    img1,
    img2,
    window_size=11,
    window=None,
    size_average=True,
    full=False,
    val_range=None,
):
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        l = max_val - min_val
    else:
        l = val_range

    padd = window_size // 2
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    c1 = (0.01 * l) ** 2
    c2 = (0.03 * l) ** 2

    v1 = 2.0 * sigma12 + c2
    v2 = sigma1_sq + sigma2_sq + c2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + c1) * v1) / ((mu1_sq + mu2_sq + c1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret, ssim_map


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size).cuda()

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = (
                create_window(self.window_size, channel)
                .to(img1.device)
                .type(img1.dtype)
            )
            self.window = window
            self.channel = channel

        s_score, ssim_map = ssim(
            img1,
            img2,
            window=window,
            window_size=self.window_size,
            size_average=self.size_average,
        )
        return 1.0 - s_score,ssim_map

    def get_map(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = (
                create_window(self.window_size, channel)
                .to(img1.device)
                .type(img1.dtype)
            )
            self.window = window
            self.channel = channel

        s_score, ssim_map = ssim(
            img1,
            img2,
            window=window,
            window_size=self.window_size,
            size_average=self.size_average,
        )
        return 1.0 - ssim_map



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def shuffle_patches(x, patch_size):
    # divide the batch of images into non-overlapping patches
    u = torch.nn.functional.unfold(x, kernel_size=patch_size, stride=patch_size, padding=0)
    # permute the patches of each image in the batch
    pu = torch.cat([b_[:, torch.randperm(b_.shape[-1])][None,...] for b_ in u], dim=0)
    # fold the permuted patches back together
    f = torch.nn.functional.fold(pu, x.shape[-2:], kernel_size=patch_size, stride=patch_size, padding=0)
    return f

def generate_fake_anomalies_joined(features,embeddings, memory_torch_original, mask, strength=None):  #输入经过编码数据，量化后的数据，权重，柏林噪声前景掩码,随机生成值
    random_embeddings = torch.zeros((embeddings.shape[0],embeddings.shape[2]*embeddings.shape[3], memory_torch_original.shape[1]))  #1*2304*256
    inputs = features.permute(0, 2, 3, 1).contiguous()  #1*48*48*256

    for k in range(embeddings.shape[0]):
        memory_torch = memory_torch_original  #赋值
        flat_input = inputs[k].view(-1, memory_torch.shape[1])  #2304*256

        distances_b = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(memory_torch ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, memory_torch.t()))   #算距离  欧式距离的平方

        percentage_vectors = strength[k]
        topk = max(1, min(int(percentage_vectors * memory_torch.shape[0]) + 1, memory_torch.shape[0] - 1))  #它似乎用于选择记忆中的前 k 个最近邻嵌入向量
        values, topk_indices = torch.topk(distances_b, topk, dim=1, largest=False)  #沿着第一个维度找topk最近距离的值  2304*1398
        topk_indices = topk_indices[:, int(memory_torch.shape[0] * 0.05):]  #2304*1296 切片操作
        topk = topk_indices.shape[1] #1296

        random_indices_hik = torch.randint(topk, size=(topk_indices.shape[0],))  #随机整数2304
        random_indices_t = topk_indices[torch.arange(random_indices_hik.shape[0]),random_indices_hik]  #2304
        random_embeddings[k] = memory_torch[random_indices_t,:]  #2304*256
    random_embeddings = random_embeddings.reshape((random_embeddings.shape[0],embeddings.shape[2],embeddings.shape[3],random_embeddings.shape[2])) #48*48*256
    random_embeddings_tensor = random_embeddings.permute(0,3,1,2).cuda()#1*256*48*48

    use_shuffle = torch.rand(1)[0].item()
    if use_shuffle > 0.5:
        psize_factor = torch.randint(0, 4, (1,)).item() # 1, 2, 4, 8
        random_embeddings_tensor = shuffle_patches(embeddings,2**psize_factor)  #对embeddings像素进行重新排列，图像增强


    down_ratio_y = int(mask.shape[2]/embeddings.shape[2]) #采样倍数
    down_ratio_x = int(mask.shape[3]/embeddings.shape[3])  #采样倍数
    anomaly_mask = torch.nn.functional.max_pool2d(mask, (down_ratio_y, down_ratio_x)).float()  #48*48*1 ，对mask进行降采样


    anomaly_embedding = anomaly_mask * random_embeddings_tensor + (1.0 - anomaly_mask) * embeddings    #

    return anomaly_embedding

def evaluate_model(model, sub_res_model_lo, sub_res_model_hi, embedder_hi, embedder_lo, model_decode, decoder_seg, visualizer, obj_name, n_iter, img_min, img_max, mvtec_path,loss_file):
    img_dim = 384
    dataset = MVTecDRAEMTestDataset(mvtec_path + obj_name + "/test/*/xyz/", resize_shape=[img_dim,img_dim], img_min=img_min, img_max=img_max)
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=0)

    total_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))
    total_gt_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))
    total_pixel_scores_2d = np.zeros((len(dataset),img_dim, img_dim))
    total_gt_pixel_scores_2d = np.zeros((len(dataset),img_dim, img_dim))
    mask_cnt = 0

    total_gt = []
    total_score = []
    loss_recon = 0.0
    cnt_normal = 0

    mask_cnt = 0
    a=0


    for i_batch, sample_batched in enumerate(dataloader):

            depth_image = sample_batched["image"].cuda()
            has_anomaly_np = sample_batched["has_anomaly"].detach().cpu().numpy()
            for i in range(has_anomaly_np.shape[0]):
                is_normal = has_anomaly_np[i, 0]
                total_gt.append(is_normal)
            #is_normal = sample_batched["has_anomaly"].detach().numpy()[0, 0]
            #total_gt.append(is_normal)
            #true_mask = sample_batched["mask"]
            #true_mask_cv = true_mask.detach().numpy()[0, :, :, :].transpose((1, 2, 0))
            true_mask = sample_batched["mask"]
            true_mask_np = true_mask.detach().cpu().numpy()

# 转换每个样本的掩码数据
            true_mask_cv_list = []
            for i in range(true_mask_np.shape[0]):
                true_mask_cv = true_mask_np[i, :, :, :].transpose((1, 2, 0))
                true_mask_cv_list.append(true_mask_cv)  
            rgb_image = sample_batched["rgb_image"].cuda()


            in_image = torch.cat((depth_image, rgb_image), dim=1)
            in_image = depth_image
            _, _, recon_out, embeddings_lo, embeddings_hi = model(in_image)
            recon_image_general = recon_out

            _, recon_embeddings_hi, _ = sub_res_model_hi(embeddings_hi, embedder_hi)
            _, recon_embeddings_lo, _ = sub_res_model_lo(embeddings_lo, embedder_lo)

            # Reconstruct the image from the reconstructed features
            # with the object-specific image reconstruction module
            up_quantized_recon_t = model.upsample_t(recon_embeddings_lo)
            quant_join = torch.cat((up_quantized_recon_t, recon_embeddings_hi), dim=1)
            recon_image_recon = model_decode(quant_join)

            # Generate the anomaly segmentation map
            out_mask = decoder_seg(recon_image_recon.detach(), recon_image_general.detach())
            out_mask_sm = torch.softmax(out_mask, dim=1)
            out_mask_averaged = torch.nn.functional.avg_pool2d(out_mask_sm[:, 1:, :, :], 21, stride=1,
                                                               padding=21 // 2).cpu().detach().numpy()
            for i in range(out_mask_averaged.shape[0]):
                out_mask_averaged_sample = out_mask_averaged[i].squeeze()
                image_score_sample = np.max(out_mask_averaged_sample)
                total_score.append(image_score_sample)
                true_mask_cv_sample = true_mask_cv_list[i]
                start_idx = mask_cnt * img_dim * img_dim
                end_idx = (mask_cnt + 1) * img_dim * img_dim
                flat_out_mask = out_mask_averaged_sample.flatten()
                total_pixel_scores[start_idx:end_idx] = flat_out_mask
                flat_true_mask = true_mask_cv_sample[:,:,0].flatten()
                total_gt_pixel_scores[start_idx:end_idx] = flat_true_mask
                total_pixel_scores_2d[mask_cnt] = out_mask_averaged_sample
                total_gt_pixel_scores_2d[mask_cnt] = true_mask_cv_sample[:, :, 0]
                
                # 保存深度图像为彩色图像，真实掩码和预测掩码为灰度图像
                depth_image_sample = rgb_image[i].detach().cpu().numpy().squeeze()
                true_mask_cv_sample_gray = true_mask_cv_sample[:, :, 0]
                
                # 缩放深度图像的数值范围至 0 到 255 之间
                depth_image_sample_scaled = (depth_image_sample - depth_image_sample.min()) / (depth_image_sample.max() - depth_image_sample.min()) * 255
                depth_image_sample_scaled = depth_image_sample_scaled.astype(np.uint8)
                depth_image_sample_scaled = depth_image_sample_scaled.transpose(1, 2, 0)
                true_mask_cv_sample_gray_binary = np.where(true_mask_cv_sample_gray != 0, 255, 0)
                out_mask_normalized = ((out_mask_averaged_sample - out_mask_averaged_sample.min()) / (out_mask_averaged_sample.max() - out_mask_averaged_sample.min()) * 255).astype(np.uint8)
                #cv2.imwrite(os.path.join(output_folder, f'out_mask_averaged_{a}.jpg'), out_mask_normalized)
# 保存灰度图像
                #cv2.imwrite(os.path.join(output_folder, f'true_mask_{a}.jpg'), true_mask_cv_sample_gray_binary)
                # 保存图像到指定文件夹
                #cv2.imwrite(os.path.join(output_folder, f'depth_image_{a}.jpg'), depth_image_sample_scaled)
                
                mask_cnt += 1
                a=a+1


    total_score = np.array(total_score)
    total_gt = np.array(total_gt)
    auroc = roc_auc_score(total_gt, total_score)
    ap = average_precision_score(total_gt, total_score)

    total_gt_pixel_scores = total_gt_pixel_scores.astype(np.uint8)
    total_gt_pixel_scores = total_gt_pixel_scores[:img_dim * img_dim * mask_cnt]
    total_pixel_scores = total_pixel_scores[:img_dim * img_dim * mask_cnt]
    auroc_pixel = roc_auc_score(total_gt_pixel_scores, total_pixel_scores)
    ap_pixel = average_precision_score(total_gt_pixel_scores, total_pixel_scores)
    aupro, _ = calculate_au_pro([total_gt_pixel_scores_2d[x] for x in range(total_gt_pixel_scores_2d.shape[0])], [total_pixel_scores_2d[x] for x in range(total_pixel_scores_2d.shape[0])])

    print(str(n_iter), end=' ')
    print(obj_name, end=' ')
    print("AUC Image: " + str(auroc), end=' ')
    print("AP Image: " + str(ap), end=' ')
    print("AUC Pixel: " + str(auroc_pixel), end=' ')
    print("AP Pixel: " + str(ap_pixel), end=' ')
    print("AUPRO: " + str(aupro))
    loss_file.write(f"Epoch {n_iter} | Image AUROC: {auroc} | Pixel AUROC: {auroc_pixel} | Pixel AUPRO: {aupro}\n")
    loss_file.flush()  # 强制刷新缓冲区，确保数据写入磁盘
    return auroc

def train_on_device(obj_names, mvtec_path, out_path, lr, batch_size, epochs, run_name_base):
    dada_model_path = "./checkpoints/DADA_D.pckl"
    ssim_loss = SSIM()

    img_dim = 384 # 设置了图像的尺寸为 384x384
    num_hiddens = 256  #指定了模型中隐藏层的大小为 256
    num_residual_hiddens = 128  #指定了残差隐藏层的大小为 128
    num_residual_layers = 2   #指定了残差隐藏层的大小为 128
    embedding_dim = 256  #设置了嵌入维度为 256
    num_embeddings = 2048  #指定了嵌入的数量为 2048
    commitment_cost = 0.25 #指定了嵌入的数量为 2048
    decay = 0.99   #设置了嵌入的衰减率为 0.99

    model = DiscreteLatentModel(num_hiddens, num_residual_layers, num_residual_hiddens, num_embeddings,
                                embedding_dim,
                                commitment_cost, decay, in_channels=1, out_channels=1)
    model.cuda()
    model.load_state_dict(torch.load(dada_model_path, map_location='cuda:0'))
    model.eval()

    # Modules using the codebooks K_hi and K_lo for feature quantization
    embedder_hi = model._vq_vae_bot
    embedder_lo = model._vq_vae_top

    for obj_name in obj_names:
        run_name = run_name_base+"_" + str(lr) + '_' + str(epochs) + '_bs' + str(batch_size) + "_" + obj_name + '_'
        visualizer=None

        # Define the subspace restriction modules - Encoder decoder networks
        sub_res_model_lo = SubspaceRestrictionModule(embedding_size=embedding_dim)
        sub_res_model_hi = SubspaceRestrictionModule(embedding_size=embedding_dim)
        sub_res_model_lo.cuda()
        sub_res_model_hi.cuda()

        # Define the anomaly detection module - UNet-based network
        #decoder_seg = AnomalyDetectionModule(in_channels=2, base_width=64)
        #decoder_seg = AnomalyDetectionModule(in_channels=2, base_width=128)
        decoder_seg = AnomalyDetectionModule(in_channels=2, base_width=32)
        decoder_seg.cuda()
        decoder_seg.apply(weights_init)  #特征初始化


        # Image reconstruction network reconstructs the image from discrete features.
        # It is trained for a specific object
        model_decode = ImageReconstructionNetwork(embedding_dim * 2,
                   num_hiddens,
                   num_residual_layers,
                   num_residual_hiddens, out_channels=1)
        model_decode.cuda()
        model_decode.apply(weights_init)



        optimizer = torch.optim.Adam([
                                      {"params": sub_res_model_lo.parameters(), "lr": lr},
                                      {"params": sub_res_model_hi.parameters(), "lr": lr},
                                      {"params": model_decode.parameters(), "lr": lr},
                                      {"params": decoder_seg.parameters(), "lr": lr}])

        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,[int(epochs*0.80)],gamma=0.1, last_epoch=-1)

        loss_focal = FocalLoss()
        dataset = MVTecDRAEMTrainDataset(mvtec_path + obj_name  + "/train/good/", resize_shape=[img_dim, img_dim])
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=True, num_workers=2)


        n_iter = 0.0
        filename = "dowel.txt"
        loss_file = open(filename, "w", buffering=1)
        for epoch in range(epochs):
            for i_batch, sample_batched in enumerate(dataloader):

                d = cosine_annealing(epoch, 3000)
                depth_image = sample_batched["image"].cuda()  #384*384*1  点云图，只提取了深度信息
                anomaly_mask = sample_batched["mask"].cuda()  #384*384*1  根据前景生成的柏林噪声的mask，非0即1

                optimizer.zero_grad()

                with torch.no_grad():
                    in_image = depth_image  #点云图，只提取了深度信息

                    anomaly_strength_lo = (torch.rand(in_image.shape[0]) * 0.90 + 0.1).cuda()  #生成一个值在0.1-1之间的值
                    anomaly_strength_hi = (torch.rand(in_image.shape[0]) * 0.90 + 0.1).cuda()  #同理生成值
                    # Extract features from the discrete model 
                    enc_b = model._encoder_b(in_image)  #返回256*96*96    ，输入1*384*384
                    enc_t = model._encoder_t(enc_b)     #1*256*48*48
                    zt = model._pre_vq_conv_top(enc_t)    #卷积，返回256*48*48

                    # Quantize the extracted features
                    _, quantized_t, _, _ = embedder_lo(zt)  #VQ-VAE模型  输出量化之后的模型 1*256*48*48

                    # Generate feature-based anomalies on F_lo
                    anomaly_embedding_lo = generate_fake_anomalies_joined(zt, quantized_t,
                                                                           embedder_lo._embedding.weight,
                                                                           anomaly_mask, strength=anomaly_strength_lo)  #256*48*48
 
                    # Upsample the extracted quantized features and the quantized features augmented with anomalies
                    up_quantized_t = model.upsample_t(anomaly_embedding_lo)  #256*96*96
                    up_quantized_t_real = model.upsample_t(quantized_t)      #256*96*96
                    feat = torch.cat((enc_b, up_quantized_t), dim=1)          #拼接   512*96*96
                    feat_real = torch.cat((enc_b, up_quantized_t_real), dim=1)   #拼接 512*96*96
                    zb = model._pre_vq_conv_bot(feat)    #256*96*96
                    zb_real = model._pre_vq_conv_bot(feat_real) #256*96*96
                    # Quantize the upsampled features - F_hi
                    _, quantized_b, _, _ = embedder_hi(zb)    #同上
                    _, quantized_b_real, _, _ = embedder_hi(zb_real)  #同上

                    # Generate feature-based anomalies on F_hi
                    anomaly_embedding = generate_fake_anomalies_joined(zb, quantized_b,
                                                                          embedder_hi._embedding.weight, anomaly_mask
                                                                         , strength=anomaly_strength_hi)  #同上

                    use_both = torch.randint(0, 2,(in_image.shape[0],1,1,1)).cuda().float()
                    use_lo = torch.randint(0, 2,(in_image.shape[0],1,1,1)).cuda().float()
                    use_hi = (1 - use_lo)  #随机生成数据
                    anomaly_embedding_hi_usebot = generate_fake_anomalies_joined(zb_real,
                                                                         quantized_b_real,
                                                                         embedder_hi._embedding.weight,
                                                                         anomaly_mask, strength=anomaly_strength_hi)  #同上   
                    anomaly_embedding_lo_usebot = quantized_t #第一层的量化模型的输出
                    anomaly_embedding_hi_usetop = quantized_b_real   #第一层量化输出模型直接进第二层量化的输出
                    anomaly_embedding_lo_usetop = anomaly_embedding_lo #第一层量化模型经过generate模型后的输出
                    anomaly_embedding_hi_not_both =  use_hi * anomaly_embedding_hi_usebot + use_lo * anomaly_embedding_hi_usetop  #第一层量化后的数据不经过generate，第二层经过generate
                    anomaly_embedding_lo_not_both =  use_hi * anomaly_embedding_lo_usebot + use_lo * anomaly_embedding_lo_usetop 
                    anomaly_embedding_hi = (anomaly_embedding * use_both + anomaly_embedding_hi_not_both * (1.0 - use_both)).detach().clone()  #第二层输出
                    anomaly_embedding_lo = (anomaly_embedding_lo * use_both + anomaly_embedding_lo_not_both * (1.0 - use_both)).detach().clone()  #第一层输出

                    anomaly_embedding_hi_copy = anomaly_embedding_hi.clone()
                    anomaly_embedding_lo_copy = anomaly_embedding_lo.clone()  #赋值操作

                # Restore the features to normality with the Subspace restriction modules
                recon_feat_hi, recon_embeddings_hi, loss_b = sub_res_model_hi(anomaly_embedding_hi_copy, embedder_hi)   #第二层的输出    返回loss和量化的数据以及编码解码的特征  
                recon_feat_lo, recon_embeddings_lo, loss_b_lo = sub_res_model_lo(anomaly_embedding_lo_copy, embedder_lo)  #第一层的输出  返回loss和量化的数据以及编码解码的特征

                # Reconstruct the image from the anomalous features with the general appearance decoder
                up_quantized_anomaly_t = model.upsample_t(anomaly_embedding_lo)  #上采样
                quant_join_anomaly = torch.cat((up_quantized_anomaly_t, anomaly_embedding_hi), dim=1)   #拼接
                recon_image_general = model._decoder_b(quant_join_anomaly)  #解码


                # Reconstruct the image from the reconstructed features
                # with the object-specific image reconstruction module
                up_quantized_recon_t = model.upsample_t(recon_embeddings_lo)
                quant_join = torch.cat((up_quantized_recon_t, recon_embeddings_hi), dim=1)
                recon_image_recon = model_decode(quant_join)

                # Generate the anomaly segmentation map
                #out_mask = decoder_seg(recon_image_recon.detach(),recon_image_general.detach())

                out_mask = decoder_seg(recon_image_recon,recon_image_general)  #1*2*384*384
                #out_mask = decoder_seg(recon_image_recon,recon_image_general)
                out_mask_sm = torch.softmax(out_mask, dim=1)

                # Calculate losses
                loss_feat_hi = torch.nn.functional.mse_loss(recon_feat_hi, quantized_b_real.detach())
                loss_feat_lo = torch.nn.functional.mse_loss(recon_feat_lo, quantized_t.detach())
                loss_l2_recon_img = torch.nn.functional.mse_loss(in_image, recon_image_recon)
                
                loss_feat_hi2 = torch.abs(recon_feat_hi - quantized_b_real.detach())  # 计算绝对值
                loss_feat_lo2 = torch.abs(recon_feat_lo - quantized_t.detach())      # 计算绝对值
                loss_l2_recon_img2 = torch.abs(in_image - recon_image_recon)         # 计算绝对值
                
                ssmloss,ssmloss_map=ssim_loss(in_image,recon_image_recon)
                loss_feat_hi1,loss_feat_hi1_map=ssim_loss(recon_feat_hi,quantized_b_real.detach())
                loss_feat_lo1,loss_feat_lo1_map = ssim_loss(recon_feat_lo, quantized_t.detach())


                loss_feat_hi_dynamic = dynamic_loss(d, loss_feat_hi2)
                loss_feat_lo_dynamic = dynamic_loss(d, loss_feat_lo2)
                loss_l2_recon_img_dynamic = dynamic_loss(d, loss_l2_recon_img2)

                loss_feat_hi_dynamic1 = dynamic_loss(d, loss_feat_hi1_map)
                loss_feat_lo_dynamic1 = dynamic_loss(d, loss_feat_lo1_map)
                
                ssmloss_dynamic = dynamic_loss(d, ssmloss_map)
                total_recon_loss = 0.1*loss_feat_lo_dynamic + 0.1*loss_feat_hi_dynamic + loss_l2_recon_img_dynamic * 10 + ssmloss_dynamic * 10+loss_feat_hi_dynamic1+loss_feat_lo_dynamic1
                #total_recon_loss = loss_feat_lo + loss_feat_hi + loss_l2_recon_img*10+ssmloss*10
                #total_recon_loss = loss_feat_lo + loss_feat_hi+ssmloss*10

                # Resize the ground truth anomaly map to closely match the augmented features
                down_ratio_x_hi = int(anomaly_mask.shape[3] / quantized_b.shape[3])
                anomaly_mask_hi = torch.nn.functional.max_pool2d(anomaly_mask,
                                                                  (down_ratio_x_hi, down_ratio_x_hi)).float()  #对anomaly_mask下采样4倍
                anomaly_mask_hi = torch.nn.functional.interpolate(anomaly_mask_hi, scale_factor=down_ratio_x_hi)  #再插值回去
                down_ratio_x_lo = int(anomaly_mask.shape[3] / quantized_t.shape[3])
                anomaly_mask_lo = torch.nn.functional.max_pool2d(anomaly_mask,
                                                                  (down_ratio_x_lo, down_ratio_x_lo)).float()
                anomaly_mask_lo = torch.nn.functional.interpolate(anomaly_mask_lo, scale_factor=down_ratio_x_lo)  #同上下采样再插值回去
                anomaly_mask = anomaly_mask_lo * use_both + (
                            anomaly_mask_lo * use_lo + anomaly_mask_hi * use_hi) * (1.0 - use_both)

                #anomaly_mask = anomaly_mask * anomaly_type_sum
                # Calculate the segmentation loss
                segment_loss = loss_focal(out_mask_sm, anomaly_mask)     
                l1_mask_loss = torch.mean(torch.abs(out_mask_sm - torch.cat((1.0 - anomaly_mask, anomaly_mask), dim=1)))
                
                #l1_mask_loss1=torch.abs(out_mask_sm - torch.cat((1.0 - anomaly_mask, anomaly_mask), dim=1))
                #l1_mask_loss_dynamic = dynamic_loss(d, l1_mask_loss1)
                #segment_loss = segment_loss + l1_mask_loss
                segment_loss = segment_loss + l1_mask_loss
                loss = segment_loss + total_recon_loss
                loss.backward()
                optimizer.step()

                n_iter +=1

            scheduler.step()


            if (epoch+1) % 1 == 0:
                with torch.no_grad():
                    evaluate_model(model, sub_res_model_lo, sub_res_model_hi, embedder_hi, embedder_lo, model_decode, 
                                   decoder_seg, visualizer, obj_name, n_iter, dataset.global_min, dataset.global_max, mvtec_path,loss_file)

            if (epoch+1) % 1 == 0:
                # Save models
                if not os.path.exists(out_path+"checkpoints/"):
                    os.makedirs(out_path+"checkpoints/")
                torch.save(decoder_seg.state_dict(), out_path+"checkpoints/"+str(epoch+1)+run_name+"_seg.pckl")
                torch.save(sub_res_model_lo.state_dict(), out_path+"checkpoints/"+str(epoch+1)+run_name+"_recon_lo.pckl")
                torch.save(sub_res_model_hi.state_dict(), out_path+"checkpoints/"+str(epoch+1)+run_name+"_recon_hi.pckl")
                torch.save(model_decode.state_dict(), out_path+"checkpoints/"+str(epoch+1)+run_name+"_decode.pckl")


        with torch.no_grad():
            print(run_name)
            evaluate_model(model, sub_res_model_lo, sub_res_model_hi, embedder_hi, embedder_lo, model_decode,
                           decoder_seg, visualizer, obj_name, n_iter, dataset.global_min, dataset.global_max, mvtec_path,loss_file)

        if not os.path.exists(out_path + "checkpoints/"):
            os.makedirs(out_path + "checkpoints/")
        torch.save(decoder_seg.state_dict(), out_path + "checkpoints/" +str(epoch+1)+ run_name + "_seg.pckl")
        torch.save(sub_res_model_lo.state_dict(), out_path + "checkpoints/" +str(epoch+1)+ run_name + "_recon_lo.pckl")
        torch.save(sub_res_model_hi.state_dict(), out_path + "checkpoints/" +str(epoch+1)+ run_name + "_recon_hi.pckl")
        torch.save(model_decode.state_dict(), out_path + "checkpoints/" +str(epoch+1)+ run_name + "_decode.pckl")

    return model, sub_res_model_hi, sub_res_model_lo, model_decode, decoder_seg

if __name__=="__main__":
    obj_classes = [["cable_gland"], ["bagel"], ["cookie"], ["carrot"], ["dowel"], ["foam"], ["peach"], ["potato"],
                   ["tire"], ["rope"]]

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_id', default=4, type=int, action='store', help='Object ID')
    parser.add_argument('--bs', default=4
    , type=int, action='store', help='Batch size')
    parser.add_argument('--lr', default=0.0002, type=float, action='store', help='Learning rate')
    parser.add_argument('--epochs', default=3000, type=int, action='store', help='Number of epochs')
    parser.add_argument('--gpu_id', default=0, type=int, action='store', help='GPU ID')
    parser.add_argument('--data_path', default="/netdisk-3.1/lijunhui/cz/mvtec_3d_anomaly_detection/", type=str, action='store', help='Data path')
    parser.add_argument('--out_path', default="output3", type=str, action='store', help='Output path')
    parser.add_argument('--run_name', default="dowel", type=str, action='store', help='Run name')

    args = parser.parse_args()


    with torch.cuda.device(args.gpu_id):
        train_on_device(obj_classes[int(args.obj_id)],args.data_path, args.out_path, args.lr, args.bs, args.epochs, args.run_name)
