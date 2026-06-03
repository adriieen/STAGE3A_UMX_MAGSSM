python /users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/open-unmix-pytorch/openunmix/train_mask.py \
--root /Data/adrien.dubois  \
--target vocals \
--output /users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/outputs/magssm/s512_h512_nofc1_nobn1  \
--is-wav \
--epochs 300 \
--batch-size 6 \
--nb-workers 10 \
--seq-dur 3 \
--chunk-dur 0.7 \
--mel \
--nb_magssm_states 512 \
--hidden-size 512 \
--use_edge \
--amp


