
for eval_subset in 'test' 'unseen'; do

CUDA_VISIBLE_DEVICES=0 python train_net_step_rel.py \
--num_gpus 1 \
--video_frame 12 \
--dataset vhico \
--cfg configs/e2e_relcnn_X-101-64x4d-FPN_8_epochs_mit_y_loss_only_coco_pretrain.yaml \
--nw 8 \
--use_tfboard \
--lr 0.0001 \
--backbone_lr_scalar 0.01 \
--obj_loss contrastive_objloss \
--com_weight cat_video_soft_attention \
--weight_reg L2 \
--video_loss contrastive_max_plus \
--binary_loss \
--eval_subset ${eval_subset} \
--human_obj_spatial \
--load_ckpt Outputs/vhico/cat+Spa+Hum+Tem+Con/ckpt/best_model.pth \

done
