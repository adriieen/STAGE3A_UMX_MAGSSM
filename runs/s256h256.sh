CUDA_VISIBLE_DEVICES=1 \
python /home/adubois/openunmix/OpenUnmix/open-unmix-pytorch/openunmix/train_mask.py \
--root  /dev/shm/musdb18 \
--target vocals \
--output /home/adubois/openunmix/OpenUnmix/outputs/500ep/magssm_fixed_init/s256h256  \
--is-wav \
--epochs 500 \
--batch-size 10 \
--nb-workers 6 \
--seq-dur 4 \
--chunk-dur 1 \
--mel \
--nb_magssm_states 256 \
--use_edge \
--hidden-size 256 \
