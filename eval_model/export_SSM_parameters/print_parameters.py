from get_continuous_time_parameters import get_parameters
Parameters = get_parameters("/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/outputs/trainable_spectograms/14.7kHz/regularized_window_double_exp/NFFT=NSTATES=682_alpha0_TEST_NEW_INITv3/eps0.5000_l1_4.7027_l2_4.0000/model_params.npz")


print(Parameters.Lambda)
print(Parameters.Delta)
print(Parameters.B)
print(Parameters.C)
print(Parameters.B_bias)
print(Parameters.C_bias)