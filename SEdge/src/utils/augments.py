import torch
import torch_audiomentations

sample_rate = 16000


class Permute(torch.nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(self.dims)



augments_weak = torch_audiomentations.Compose([
    Permute([0,2,1]),
    torch_audiomentations.PolarityInversion(output_type='tensor'),
    # torch_audiomentations.Gain(),
    torch_audiomentations.Gain(min_gain_in_db=-3, max_gain_in_db=3,output_type='tensor'),
    # torch_audiomentations.PitchShift(sample_rate=sample_rate),
    torch_audiomentations.PitchShift(sample_rate=sample_rate,min_transpose_semitones=-2,max_transpose_semitones=2,output_type='tensor'),
    torch_audiomentations.Shift(min_shift=0.1, max_shift=0.1, p=0.5, rollover=False,output_type='tensor'),
    torch_audiomentations.AddColoredNoise(output_type='tensor'),
    Permute([0,2,1]),
],output_type='tensor')


augments_strong = torch_audiomentations.Compose([
    Permute([0,2,1]),
    torch_audiomentations.PolarityInversion(output_type='tensor'),
    torch_audiomentations.Gain(output_type='tensor'),
    torch_audiomentations.PitchShift(sample_rate=sample_rate,output_type='tensor'),
    torch_audiomentations.Shift(min_shift=0.1, max_shift=0.1, p=0.5, rollover=False,output_type='tensor'),
    torch_audiomentations.AddColoredNoise(output_type='tensor'),
    Permute([0,2,1]),
],output_type='tensor')