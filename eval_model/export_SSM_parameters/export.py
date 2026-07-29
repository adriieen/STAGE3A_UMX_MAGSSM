from get_continuous_time_parameters import get_parameters

path = "/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/outputs/trainable_spectograms/14.7kHz/Tests_post_soutenance/og_magssm_non_progressive_170bins/eps0.5000_l1_4.7027_l2_4.0000/vocals.pth"  # ou le dossier contenant vocals.pth et vocals.json
Parameters = get_parameters(path)

Parameters.save_npz("/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/outputs/trainable_spectograms/14.7kHz/Tests_post_soutenance/og_magssm_non_progressive_170bins/eps0.5000_l1_4.7027_l2_4.0000/model_params.npz")