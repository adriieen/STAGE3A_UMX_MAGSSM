CUDA_VISIBLE_DEVICES=0 \
python /home/adubois/openunmix/OpenUnmix/open-unmix-pytorch/openunmix/train_mask.py \
--root  /dev/shm/musdb18 \
--target vocals \
--output /home/adubois/openunmix/OpenUnmix/outputs/500ep/magssm_fixed_init/s512h512  \
--is-wav \
--epochs 500 \
--batch-size 8 \
--nb-workers 6 \
--seq-dur 3 \
--chunk-dur 0.7 \
--mel \
--nb_magssm_states 512 \
--use_edge \
--hidden-size 512 \

