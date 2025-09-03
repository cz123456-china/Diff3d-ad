import torch
from data_loader import MVTecDRAEMTestDataset
from torch.utils.data import DataLoader
import numpy as np
from dsr_model import SubspaceRestrictionModule, ImageReconstructionNetwork, AnomalyDetectionModule
from discrete_model import DiscreteLatentModel
from au_pro_util import calculate_au_pro
from sklearn.metrics import roc_auc_score, average_precision_score
from collections import defaultdict
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import auc, roc_curve,average_precision_score
from sklearn.metrics import roc_auc_score
import time
import numpy as np
from scipy.ndimage import gaussian_filter
import cv2
import torch.nn as nn
from models.Recon_subnetwork import UNetModel, update_ema_params
from models.Seg_subnetwork import SegmentationSubNetwork
import torch.nn as nn
from data.dataset_beta_thresh import MVTecTrainDataset,MVTecTestDataset,VisATrainDataset,VisATestDataset,DAGMTrainDataset,DAGMTestDataset,MPDDTestDataset,MPDDTrainDataset
from models.DDPM import GaussianDiffusionModel, get_beta_schedule
from math import exp
import torch.nn.functional as F
torch.cuda.empty_cache()
from tqdm import tqdm
import json
import os
from collections import defaultdict
import pandas as pd
import torchvision.utils
import os
from torch.utils.data import DataLoader
from skimage.measure import label, regionprops
import sys
from sklearn import linear_model
from data_loader import MVTecDRAEMTestDataset
import torchvision.utils as vutils

def pretraintrain(testing_dataset_loader, args,unet_model,seg_model,data_len,sub_class,class_type,checkpoint_type,device,obj_name):
    normal_t=args["eval_normal_t"]
    noiser_t=args["eval_noisier_t"]
    img_dim = 384
    s_map_lib = []
    s_lib=[]
    
    os.makedirs(f'{args["output_path"]}/metrics/ARGS={args["arg_num"]}/{sub_class}/visualization_{normal_t}_{noiser_t}_{args["condition_w"]}condition_{checkpoint_type}ck', exist_ok=True)
    
    in_channels = args["channels"]
    betas = get_beta_schedule(args['T'], args['beta_schedule'])

    ddpm_sample =  GaussianDiffusionModel(
            args['img_size'], betas, loss_weight=args['loss_weight'],
            loss_type=args['loss-type'], noise=args["noise_fn"], img_channels=in_channels
            )
    
    
    
    
    
    
    
    seg_fuser=linear_model.SGDOneClassSVM(random_state=42, nu=0.5,  max_iter=1000)
    detect_fuser = linear_model.SGDOneClassSVM(random_state=42, nu=0.5,  max_iter=1000)
    dada_model_path = "./checkpoints/DADA_D.pckl"
    out_path="outputcheckpoints/"
    mvtec_path='/netdisk-3.1/lijunhui/cz/mvtec_3d_anomaly_detection/'


    dataset = MVTecDRAEMTRAINDataset1(mvtec_path + obj_name + "/train/*/xyz/",
                                    resize_shape=[img_dim, img_dim],img_size=args["img_size"])
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=0)
    num_hiddens = 256
    num_residual_hiddens = 128
    num_residual_layers = 2
    embedding_dim = 256
    #num_embeddings = 1024
    num_embeddings = 2048
    commitment_cost = 0.25
    decay = 0.99

    model = DiscreteLatentModel(num_hiddens, num_residual_layers, num_residual_hiddens, num_embeddings,
                                embedding_dim,
                                commitment_cost, decay, in_channels=1, out_channels=1)

    model.cuda()
    model.load_state_dict(torch.load(dada_model_path, map_location='cuda:0'))
    model.eval()

    embedder_hi = model._vq_vae_bot
    embedder_lo = model._vq_vae_top
    #for obj_name in obj_names:
    
    run_name ="962"+obj_name+"_" + str(0.0002) + '_' + str(1000) + '_bs' + str(4) + "_" + obj_name + '_'


    sub_res_model_lo = SubspaceRestrictionModule(embedding_size=embedding_dim)
    sub_res_model_lo.load_state_dict(torch.load(out_path+run_name+"_recon_lo.pckl", map_location='cuda:0'))
    sub_res_model_hi = SubspaceRestrictionModule(embedding_size=embedding_dim)
    sub_res_model_hi.load_state_dict(torch.load(out_path+run_name+"_recon_hi.pckl", map_location='cuda:0'))
    sub_res_model_lo.cuda()
    sub_res_model_lo.eval()
    sub_res_model_hi.cuda()
    sub_res_model_hi.eval()




    decoder_seg = AnomalyDetectionModule(in_channels=2, base_width=32)
    decoder_seg.load_state_dict(torch.load(out_path+run_name+"_seg.pckl", map_location='cuda:0'))
    decoder_seg.cuda()
    decoder_seg.eval()

    model_decode = ImageReconstructionNetwork(embedding_dim * 2,
            num_hiddens,
            num_residual_layers,
            num_residual_hiddens, out_channels=1)
    model_decode.load_state_dict(torch.load(out_path+run_name+"_decode.pckl", map_location='cuda:0'))
    model_decode.cuda()
    model_decode.eval()
    mask_cnt = 0
    tbar = tqdm(dataloader)
    for i_batch, (sample_batched, sample) in enumerate(tbar):
        image = sample["image"].to(device)
        
        normal_t_tensor = torch.tensor([normal_t], device=image.device).repeat(image.shape[0])
        noiser_t_tensor = torch.tensor([noiser_t], device=image.device).repeat(image.shape[0])
        unet_model.eval()
        with torch.no_grad():
            loss,pred_x_0_condition,pred_x_0_normal,pred_x_0_noisier,x_normal_t,x_noiser_t,pred_x_t_noisier = ddpm_sample.norm_guided_one_step_denoising_eval(unet_model, image, normal_t_tensor,noiser_t_tensor,args)
            
        pred_mask = seg_model(torch.cat((image, pred_x_0_condition), dim=1)) 
        out_mask1 = pred_mask


        topk_out_mask = torch.flatten(out_mask1[0], start_dim=1)
        topk_out_mask = torch.topk(topk_out_mask, 50, dim=1, largest=True)[0]
        image_score1 = torch.mean(topk_out_mask)
        flatten_pred_mask=out_mask1[0]






        depth_image = sample_batched["image"].cuda()
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
        model_decode.eval()
        with torch.no_grad():
            recon_image_recon = model_decode(quant_join)


        # Generate the anomaly segmentation map
        out_mask = decoder_seg(recon_image_recon.detach(), recon_image_general.detach())
        out_mask_sm = torch.softmax(out_mask, dim=1)
        out_mask_averaged = torch.nn.functional.avg_pool2d(out_mask_sm[:, 1:, :, :], 21, stride=1,
                                                            padding=21 // 2).cpu().detach().numpy()
        flat_out_mask = out_mask_averaged[0,0,:,:].flatten()
        image_score = np.max(out_mask_averaged)
        out_mask_averaged_tensor = torch.tensor(out_mask_averaged, dtype=torch.float32)
        out_mask_averaged_tensor=torch.nn.functional.interpolate(out_mask_averaged_tensor, size=(256, 256), mode='bilinear', align_corners=False)

        
        out_mask_averaged_tensor = out_mask_averaged_tensor.to(flatten_pred_mask.device)

        '''mean1 = torch.mean(out_mask1)
        mean_avg = torch.mean(out_mask_averaged_tensor)

        # 调整数量级
        out_mask_averaged_tensor = (out_mask_averaged_tensor / mean_avg) * mean1'''

        image_score_sample = torch.max(out_mask_averaged_tensor).item()



        s=torch.tensor([[image_score_sample,image_score1]])
        s_lib.append(s)
        s_map = torch.cat([out_mask_averaged_tensor,out_mask1], dim=0).squeeze().reshape(2, -1).permute(1, 0)
        s_map = s_map.detach()
        s_map = s_map.cpu()
        s_map_lib.append(s_map)
        mask_cnt += 1
        #print(image_score_sample.size())
        

    s_map_lib=torch.cat(s_map_lib, 0) 
    s_lib = torch.cat(s_lib, 0)  
    detect_fuser.fit(s_lib)#       
    seg_fuser.fit(s_map_lib)
    return detect_fuser,seg_fuser















def pixel_pro(mask,pred):
    mask=np.asarray(mask, dtype=np.bool_)
    print(mask.shape)
    pred = np.asarray(pred)
    print(pred.shape)

    max_step = 1000
    expect_fpr = 0.3  # default 30%
    max_th = pred.max()
    min_th = pred.min()
    delta = (max_th - min_th) / max_step
    ious_mean = []
    ious_std = []
    pros_mean = []
    pros_std = []
    threds = []
    fprs = []
    binary_score_maps = np.zeros_like(pred, dtype=np.bool_)
    for step in range(max_step):
        thred = max_th - step * delta
        # segmentation
        binary_score_maps[pred <= thred] = 0
        binary_score_maps[pred >  thred] = 1
        pro = []  # per region overlap
        iou = []  # per image iou
        # pro: find each connected gt region, compute the overlapped pixels between the gt region and predicted region
        # iou: for each image, compute the ratio, i.e. intersection/union between the gt and predicted binary map 
        for i in range(len(binary_score_maps)):    # for i th image
            # pro (per region level)
            label_map = label(mask[i], connectivity=2)
            props = regionprops(label_map)
            for prop in props:
                x_min, y_min, x_max, y_max = prop.bbox    # find the bounding box of an anomaly region 
                cropped_pred_label = binary_score_maps[i][x_min:x_max, y_min:y_max]
                # cropped_mask = gt_mask[i][x_min:x_max, y_min:y_max]   # bug!
                cropped_mask = prop.filled_image    # corrected!
                intersection = np.logical_and(cropped_pred_label, cropped_mask).astype(np.float32).sum()
                pro.append(intersection / prop.area)
            # iou (per image level)
            intersection = np.logical_and(binary_score_maps[i], mask[i]).astype(np.float32).sum()
            union = np.logical_or(binary_score_maps[i], mask[i]).astype(np.float32).sum()
            if mask[i].any() > 0:    # when the gt have no anomaly pixels, skip it
                iou.append(intersection / union)
        # against steps and average metrics on the testing data
        ious_mean.append(np.array(iou).mean())
        #print("per image mean iou:", np.array(iou).mean())
        ious_std.append(np.array(iou).std())
        pros_mean.append(np.array(pro).mean())
        pros_std.append(np.array(pro).std())
        # fpr for pro-auc
        gt_masks_neg = ~mask
        fpr = np.logical_and(gt_masks_neg, binary_score_maps).sum() / gt_masks_neg.sum()
        fprs.append(fpr)
        threds.append(thred)
    # as array
    threds = np.array(threds)
    pros_mean = np.array(pros_mean)
    pros_std = np.array(pros_std)
    fprs = np.array(fprs)
    ious_mean = np.array(ious_mean)
    ious_std = np.array(ious_std)
    # best per image iou
    best_miou = ious_mean.max()
    #print(f"Best IOU: {best_miou:.4f}")
    # default 30% fpr vs pro, pro_auc
    idx = fprs <= expect_fpr  # find the indexs of fprs that is less than expect_fpr (default 0.3)
    fprs_selected = fprs[idx]
    fprs_selected = min_max_norm(fprs_selected)  # rescale fpr [0,0.3] -> [0, 1]
    pros_mean_selected = pros_mean[idx]    
    seg_pro_auc = auc(fprs_selected, pros_mean_selected)
    return seg_pro_auc


def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)

def load_parameters(device,sub_class,checkpoint_type):
    
    param = "args1.json"
    with open(f'./args/{param}', 'r') as f:
        args = json.load(f)
    args['arg_num'] = param[4:-5]
    args = defaultdict_from_json(args)

    output = load_checkpoint(param[4:-5], device,sub_class,checkpoint_type,args)
 
    return args, output

def defaultdict_from_json(jsonDict):
    func = lambda: defaultdict(str)
    dd = func()
    dd.update(jsonDict)
    return dd

def load_checkpoint(param, device,sub_class,checkpoint_type,args):

    ck_path = f'{args["output_path"]}/model/diff-params-ARGS={param}/{sub_class}/params-{checkpoint_type}.pt'
    print("checkpoint",ck_path)
    loaded_model = torch.load(ck_path, map_location=device)
          
    return loaded_model



def testing(testing_dataset_loader, args,unet_model,seg_model,data_len,sub_class,class_type,checkpoint_type,device):
    
    
    normal_t=args["eval_normal_t"]
    noiser_t=args["eval_noisier_t"]
    
    os.makedirs(f'{args["output_path"]}/metrics/ARGS={args["arg_num"]}/{sub_class}/visualization_{normal_t}_{noiser_t}_{args["condition_w"]}condition_{checkpoint_type}ck', exist_ok=True)
    
    in_channels = args["channels"]
    betas = get_beta_schedule(args['T'], args['beta_schedule'])

    ddpm_sample =  GaussianDiffusionModel(
            args['img_size'], betas, loss_weight=args['loss_weight'],
            loss_type=args['loss-type'], noise=args["noise_fn"], img_channels=in_channels
            )
    
    
    
    print("data_len",data_len)
    total_image_pred = np.array([])
    total_image_gt =np.array([])
    total_pixel_gt=np.array([])
    total_pixel_pred = np.array([])
    gt_matrix_pixel=[]
    pred_matrix_pixel=[]
    tbar = tqdm(testing_dataset_loader)
    for i, (_, sample) in enumerate(tbar):
        image = sample["image"].to(device)
        target=sample['has_anomaly'].to(device)
        gt_mask = sample["mask"].to(device)
        image_path = sample["file_name"]
        
        normal_t_tensor = torch.tensor([normal_t], device=image.device).repeat(image.shape[0])
        noiser_t_tensor = torch.tensor([noiser_t], device=image.device).repeat(image.shape[0])
        loss,pred_x_0_condition,pred_x_0_normal,pred_x_0_noisier,x_normal_t,x_noiser_t,pred_x_t_noisier = ddpm_sample.norm_guided_one_step_denoising_eval(unet_model, image, normal_t_tensor,noiser_t_tensor,args)
        
        pred_mask = seg_model(torch.cat((image, pred_x_0_condition), dim=1)) 


        out_mask = pred_mask

        #pixel_aupro

        gt_matrix_pixel.extend(gt_mask[0].detach().cpu().numpy().astype(int))
        pred_matrix_pixel.extend(out_mask[0].detach().cpu().numpy())

        topk_out_mask = torch.flatten(out_mask[0], start_dim=1)
        topk_out_mask = torch.topk(topk_out_mask, 50, dim=1, largest=True)[0]
        image_score = torch.mean(topk_out_mask)
        
        total_image_pred=np.append(total_image_pred,image_score.detach().cpu().numpy())
        total_image_gt=np.append(total_image_gt,target[0].detach().cpu().numpy())


        flatten_pred_mask=out_mask[0].flatten().detach().cpu().numpy()
        flatten_gt_mask =gt_mask[0].flatten().detach().cpu().numpy().astype(int)
            
        
        total_pixel_gt=np.append(total_pixel_gt,flatten_gt_mask)
        total_pixel_pred=np.append(total_pixel_pred,flatten_pred_mask)

        
        


    
    auroc_image = round(roc_auc_score(total_image_gt,total_image_pred),3)*100
    print("Image AUC-ROC: " ,auroc_image)
    
    
    auroc_pixel =  round(roc_auc_score(total_pixel_gt, total_pixel_pred),3)*100
    print("Pixel AUC-ROC:" ,auroc_pixel)

    ap_pixel =  round(average_precision_score(total_pixel_gt, total_pixel_pred),3)*100
    print("Pixel-AP:",ap_pixel) 
    


    aupro_pixel = round(pixel_pro(gt_matrix_pixel,pred_matrix_pixel),3)*100
    print("Pixel-AUPRO:" ,aupro_pixel)
    
    temp = {"classname":[sub_class],"Image-AUROC": [auroc_image],"Pixel-AUROC":[auroc_pixel],"Pixel-AUPRO":[aupro_pixel],"Pixel_AP":[ap_pixel]}
    df_class = pd.DataFrame(temp)
    df_class.to_csv(f"{args['output_path']}/metrics/ARGS={args['arg_num']}/{normal_t}_{noiser_t}t_{class_type}_{args['condition_w']}condition_{checkpoint_type}ck.csv", mode='a',header=False,index=False)

#def test(obj_names):
def test(obj_names, mvtec_path, out_path, run_name_base):
    mvtec_classes = ['cable_gland', 'bagel', 'carrot', 'cookie', 'dowel', 'foam', 'peach', 'potato', 'rope', 'tire']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_type='best'

    #for sub_class in current_classes:
    sub_class='cable_gland'
    args, output = load_parameters(device,sub_class,checkpoint_type)
    print(f"args{args['arg_num']}")
    print("class",sub_class)
    in_channels = args["channels"]

    unet_model = UNetModel(args['img_size'][0], args['base_channels'], channel_mults=args['channel_mults'], dropout=args[
                "dropout"], n_heads=args["num_heads"], n_head_channels=args["num_head_channels"],
            in_channels=in_channels
            ).to(device)
    obj_name=obj_names[0]
    img_dim = 384




    seg_model=SegmentationSubNetwork(in_channels=6, out_channels=1).to(device)

    unet_model.load_state_dict(output["unet_model_state_dict"])
    unet_model.to(device)
    unet_model.eval()

    seg_model.load_state_dict(output["seg_model_state_dict"])
    seg_model.to(device)
    seg_model.eval()

    print("EPOCH:",output['n_epoch'])

    if sub_class in mvtec_classes:
        subclass_path = os.path.join(args["mvtec_root_path"],sub_class)
        testing_dataset = MVTecTestDataset(
            subclass_path,sub_class,img_size=args["img_size"],
            )
        class_type='MVTec'
    
            

    test_loader = DataLoader(testing_dataset, batch_size=1,shuffle=False, num_workers=4)
    
    dataset = MVTecDRAEMTestDataset(mvtec_path + obj_name + "/test/*/xyz/",
                                    resize_shape=[img_dim, img_dim],img_size=args["img_size"])
    data_len = len(dataset) 
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=4)
    for i in [f'{args["output_path"]}/metrics/ARGS={args["arg_num"]}/{sub_class}']:
        try:
            os.makedirs(i)
        except OSError:
            pass
    detect_fuser, seg_fuser=pretraintrain(test_loader, args,unet_model,seg_model,data_len,sub_class,class_type,checkpoint_type,device,obj_name)    

    #testing(dataloader, args,unet_model,seg_model,data_len,sub_class,class_type,checkpoint_type,device)
    normal_t=args["eval_normal_t"]
    noiser_t=args["eval_noisier_t"]
    
    os.makedirs(f'{args["output_path"]}/metrics/ARGS={args["arg_num"]}/{sub_class}/visualization_{normal_t}_{noiser_t}_{args["condition_w"]}condition_{checkpoint_type}ck', exist_ok=True)
    
    in_channels = args["channels"]
    betas = get_beta_schedule(args['T'], args['beta_schedule'])

    ddpm_sample =  GaussianDiffusionModel(
            args['img_size'], betas, loss_weight=args['loss_weight'],
            loss_type=args['loss-type'], noise=args["noise_fn"], img_channels=in_channels
            )
    
    
    
    print("data_len",data_len)
    total_image_pred = np.array([])
    total_image_gt =np.array([])
    total_pixel_gt=np.array([])
    total_pixel_pred = np.array([])
    gt_matrix_pixel=[]
    pred_matrix_pixel=[]
    
    
    
    
    
    





    total_ap_pixel = []
    total_auroc_pixel = []
    total_ap = []
    total_auroc = []
    total_aupro = []

    dada_model_path = "./checkpoints/DADA_D.pckl"

    num_hiddens = 256
    num_residual_hiddens = 128
    num_residual_layers = 2
    embedding_dim = 256
    #num_embeddings = 1024
    num_embeddings = 2048
    commitment_cost = 0.25
    decay = 0.99

    model = DiscreteLatentModel(num_hiddens, num_residual_layers, num_residual_hiddens, num_embeddings,
                                embedding_dim,
                                commitment_cost, decay, in_channels=1, out_channels=1)

    model.cuda()
    model.load_state_dict(torch.load(dada_model_path, map_location='cuda:0'))
    model.eval()

    embedder_hi = model._vq_vae_bot
    embedder_lo = model._vq_vae_top

    run_name = "962"+run_name_base+"_" + str(0.0002) + '_' + str(1000) + '_bs' + str(4) + "_" + obj_name + '_'


    sub_res_model_lo = SubspaceRestrictionModule(embedding_size=embedding_dim)
    sub_res_model_lo.load_state_dict(torch.load(out_path+run_name+"_recon_lo.pckl", map_location='cuda:0'))
    sub_res_model_hi = SubspaceRestrictionModule(embedding_size=embedding_dim)
    sub_res_model_hi.load_state_dict(torch.load(out_path+run_name+"_recon_hi.pckl", map_location='cuda:0'))
    sub_res_model_lo.cuda()
    sub_res_model_lo.eval()
    sub_res_model_hi.cuda()
    sub_res_model_hi.eval()



    # Define the anomaly detection module - UNet-based network
    #decoder_seg = AnomalyDetectionModule(in_channels=2, base_width=64)
    #decoder_seg = AnomalyDetectionModule(in_channels=2, base_width=128)
    decoder_seg = AnomalyDetectionModule(in_channels=2, base_width=32)
    decoder_seg.load_state_dict(torch.load(out_path+run_name+"_seg.pckl", map_location='cuda:0'))
    decoder_seg.cuda()
    decoder_seg.eval()


    # Image reconstruction network reconstructs the image from discrete features.
    # It is trained for a specific object
    model_decode = ImageReconstructionNetwork(embedding_dim * 2,
                num_hiddens,
                num_residual_layers,
                num_residual_hiddens, out_channels=1)
    model_decode.load_state_dict(torch.load(out_path+run_name+"_decode.pckl", map_location='cuda:0'))
    model_decode.cuda()
    model_decode.eval()


    total_pixel_scores = np.zeros((256 * 256 * len(dataset)))
    total_gt_pixel_scores = np.zeros((256 * 256 * len(dataset)))
    total_pixel_scores_2d = np.zeros((len(dataset),256, 256))
    total_gt_pixel_scores_2d = np.zeros((len(dataset),256, 256))

    mask_cnt = 0

    total_gt = []
    total_score = []
    tbar = tqdm(dataloader)


    for i_batch, (sample_batched, sample) in enumerate(tbar):
        image = sample["image"].to(device)
        target=sample['has_anomaly'].to(device)
        gt_mask = sample["mask"].to(device)
        image_path = sample["file_name"]
        
        normal_t_tensor = torch.tensor([normal_t], device=image.device).repeat(image.shape[0])
        noiser_t_tensor = torch.tensor([noiser_t], device=image.device).repeat(image.shape[0])
        loss,pred_x_0_condition,pred_x_0_normal,pred_x_0_noisier,x_normal_t,x_noiser_t,pred_x_t_noisier = ddpm_sample.norm_guided_one_step_denoising_eval(unet_model, image, normal_t_tensor,noiser_t_tensor,args)
        
        pred_mask = seg_model(torch.cat((image, pred_x_0_condition), dim=1)) 


        out_mask1 = pred_mask
        

        #pixel_aupro

        gt_matrix_pixel.extend(gt_mask[0].detach().cpu().numpy().astype(int))
        

        topk_out_mask = torch.flatten(out_mask1[0], start_dim=1)
        topk_out_mask = torch.topk(topk_out_mask, 50, dim=1, largest=True)[0]
        image_score1 = torch.mean(topk_out_mask)
        
        
        total_image_gt=np.append(total_image_gt,target[0].detach().cpu().numpy())


        
        flatten_gt_mask =gt_mask[0].flatten().detach().cpu().numpy().astype(int)
            
        
        total_pixel_gt=np.append(total_pixel_gt,flatten_gt_mask)
        





        depth_image = sample_batched["image"].cuda()

        is_normal = sample_batched["has_anomaly"].detach().numpy()[0, 0]
        total_gt.append(is_normal)
        true_mask = sample_batched["mask"]
        true_mask_cv = F.interpolate(true_mask, size=(256, 256), mode='bilinear', align_corners=False)
        true_mask_cv = true_mask_cv.detach().numpy()[0, :, :, :].transpose((1, 2, 0))

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
        
        image_score = np.max(out_mask_averaged)
        out_mask_averaged_tensor = torch.tensor(out_mask_averaged, dtype=torch.float32)
        out_mask_averaged_tensor=torch.nn.functional.interpolate(out_mask_averaged_tensor, size=(256, 256), mode='bilinear', align_corners=False)
        out_mask_averaged_tensor = out_mask_averaged_tensor.to(out_mask1.device)
        #out_mask_averaged_tensor = out_mask_averaged_tensor.to(flatten_pred_mask.device)

        # 归一化 out_mask1





        image_score_sample = torch.max(out_mask_averaged_tensor).item()
        #s=torch.tensor([[image_score_sample, image_score]])
        
        '''mean1 = torch.mean(out_mask1)
        mean_avg = torch.mean(out_mask_averaged_tensor)

        # 调整数量级
        out_mask_averaged_tensor = (out_mask_averaged_tensor / mean_avg) * mean1'''


        s=torch.tensor([[image_score_sample,image_score1]])
        s= torch.tensor(detect_fuser.score_samples(s))





        s_map = torch.cat([out_mask_averaged_tensor,out_mask1], dim=0).squeeze().reshape(2, -1).permute(1, 0)
        s_map=s_map.cpu()
        s_map = torch.tensor(seg_fuser.score_samples(s_map.detach().numpy()))
        s_map = s_map.view(-1,1, 256, 256)
        pred_matrix_pixel.extend(s_map[0].detach().cpu().numpy())
        total_image_pred=np.append(total_image_pred,s.detach().cpu().numpy())
        flatten_pred_mask=s_map[0].flatten().detach().cpu().numpy()
        total_pixel_pred=np.append(total_pixel_pred,flatten_pred_mask)
        
        flat_out_mask = s_map[0,0,:,:].flatten()
        total_score.append(s.cpu().numpy())

        flat_true_mask = true_mask_cv.flatten()
        total_pixel_scores[mask_cnt * 256 * 256:(mask_cnt + 1) * 256 * 256] = flat_out_mask
        total_gt_pixel_scores[mask_cnt * 256 * 256:(mask_cnt + 1) * 256 * 256] = flat_true_mask
        total_pixel_scores_2d[mask_cnt] = s_map[0,0,:,:]
        total_gt_pixel_scores_2d[mask_cnt] = true_mask_cv[:,:,0]
        mask_cnt += 1
        
        '''output_dir = "output_images"
        os.makedirs(output_dir, exist_ok=True)

        # Inside your loop, after computing s_map, true_mask_cv, and gt_mask:
        image_index = mask_cnt  # or another suitable index for naming

        # Save s_map
        s_map_image = s_map[0, 0, :, :]  # 假设 s_map 的形状是 [1, 1, H, W]
        s_map_image = ((s_map_image - s_map_image.min()) / (s_map_image.max() - s_map_image.min())) # 归一化到 [0, 1]
        s_map_image = s_map_image.unsqueeze(0)  # 添加 batch 维度 [1, H, W]
        s_map_image = s_map_image.unsqueeze(0)  # 添加 channel 维度 [1, 1, H, W]
        vutils.save_image(s_map_image, f"{output_dir}/s_map_{image_index}.png")

        # 保存 true_mask_cv（不进行归一化）
        true_mask_cv_image = torch.tensor(true_mask_cv).float()  # 如果是 numpy 数组，转换为 tensor
        true_mask_cv_image = true_mask_cv_image.squeeze(-1)  # 移除最后的维度 [256, 256]
        true_mask_cv_image = true_mask_cv_image.unsqueeze(0)  # 添加 batch 维度 [1, 256, 256]
        true_mask_cv_image = true_mask_cv_image.unsqueeze(0)  # 添加 channel 维度 [1, 1, 256, 256]
        vutils.save_image(true_mask_cv_image, f"{output_dir}/true_mask_cv_{image_index}.png")

        # 保存 gt_mask（不进行归一化）
        gt_mask_image = gt_mask[0]  # 假设 gt_mask 已经是一个 tensor
        gt_mask_image = gt_mask_image.unsqueeze(0)  # 添加 batch 维度 [1, H, W]
        vutils.save_image(gt_mask_image, f"{output_dir}/gt_mask_{image_index}.png")'''

    auroc_image = round(roc_auc_score(total_image_gt,total_image_pred),3)*100
    
    print("Image AUC-ROC: " ,auroc_image)
    
    
    auroc_pixel =  round(roc_auc_score(total_pixel_gt, total_pixel_pred),3)*100
    print("Pixel AUC-ROC:" ,auroc_pixel)

    ap_pixel =  round(average_precision_score(total_pixel_gt, total_pixel_pred),3)*100
    print("Pixel-AP:",ap_pixel) 
    


    aupro_pixel = round(pixel_pro(gt_matrix_pixel,pred_matrix_pixel),3)*100
    print("Pixel-AUPRO:" ,aupro_pixel)
    
    temp = {"classname":[sub_class],"Image-AUROC": [auroc_image],"Pixel-AUROC":[auroc_pixel],"Pixel-AUPRO":[aupro_pixel],"Pixel_AP":[ap_pixel]}
    df_class = pd.DataFrame(temp)
    df_class.to_csv(f"{args['output_path']}/metrics/ARGS={args['arg_num']}/{normal_t}_{noiser_t}t_{class_type}_{args['condition_w']}condition_{checkpoint_type}ck.csv", mode='a',header=False,index=False)




    total_score = np.array(total_score)
    total_gt = np.array(total_gt)
    auroc = roc_auc_score(total_gt, total_score)
    ap = average_precision_score(total_gt, total_score)

    total_gt_pixel_scores = total_gt_pixel_scores.astype(np.uint8)
    total_gt_pixel_scores = total_gt_pixel_scores[:256 * 256 * mask_cnt]
    total_pixel_scores = total_pixel_scores[:256 * 256 * mask_cnt]
    auroc_pixel = roc_auc_score(total_gt_pixel_scores, total_pixel_scores)
    ap_pixel = average_precision_score(total_gt_pixel_scores, total_pixel_scores)
    aupro, _ = calculate_au_pro([total_gt_pixel_scores_2d[x] for x in range(total_gt_pixel_scores_2d.shape[0])], [total_pixel_scores_2d[x] for x in range(total_pixel_scores_2d.shape[0])])

    print("------------------")
    print(obj_name)
    print("AUC Image: " + str(auroc))
    print("AP Image: " + str(ap))
    print("AUC Pixel: " + str(auroc_pixel))
    print("AP Pixel: " + str(ap_pixel))
    print("AUPRO: " + str(aupro))

    total_aupro.append(aupro)
    total_auroc_pixel.append(auroc_pixel)
    total_auroc.append(auroc)
    total_ap.append(ap)
    total_ap_pixel.append(ap_pixel)
    print("--------MEAN---------------------------------------")
    print("AUC Image: " + str(np.mean(total_auroc)))
    print("AP Image: " + str(np.mean(total_ap)))
    print("AUC Pixel: " + str(np.mean(total_auroc_pixel)))
    print("AP Pixel: " + str(np.mean(total_ap_pixel)))
    print("AUPRO: " + str(np.mean(total_aupro)))

    print("AUC",*[np.round(x*100,2) for x in total_auroc],np.round(np.mean(total_auroc)*100,2))
    print("AUCp",*[np.round(x*100,2) for x in total_auroc_pixel],np.round(np.mean(total_auroc_pixel)*100,2))
    print("AUPRO",*[np.round(x*100,2) for x in total_aupro],np.round(np.mean(total_aupro)*100,2))
    print("AP",*[np.round(x*100,2) for x in total_ap],np.round(np.mean(total_ap)*100,2))


if __name__=="__main__":
    obj_classes = [["cable_gland"], ["bagel"], ["cookie"], ["carrot"], ["dowel"], ["foam"], ["peach"], ["potato"],
                   ["tire"], ["rope"]]

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id',default=0, action='store', type=int,)
    parser.add_argument('--data_path',default='/netdisk-3.1/lijunhui/cz/mvtec_3d_anomaly_detection/',action='store', type=str)
    parser.add_argument('--out_path',default='outputcheckpoints/',action='store', type=str)
    parser.add_argument('--run_name',default='cable_gland',action='store', type=str)


    


    args = parser.parse_args()
    with torch.cuda.device(args.gpu_id):
        test(obj_classes[0],args.data_path, args.out_path, args.run_name)
