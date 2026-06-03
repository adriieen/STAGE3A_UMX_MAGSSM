import torch
import os

def test():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)
    
    batch = 16
    channels = 2
    timesteps = 264600
    n_fft = 4096
    n_hop = 1024
    
    x = torch.rand(batch, channels, timesteps, device=device)
    print("x shape:", x.shape)
    
    window = torch.hann_window(n_fft).to(device)
    
    x = x.view(-1, x.shape[-1])
    print("x viewed shape:", x.shape)
    
    try:
        # Simulate TorchSTFT forward
        complex_stft = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=n_hop,
            window=window,
            center=False,
            normalized=False,
            onesided=True,
            pad_mode="reflect",
            return_complex=True,
        )
        print("STFT Success! Shape:", complex_stft.shape)
    except Exception as e:
        print("STFT FAILED:", str(e))
        
    try:
        # Try clearing cufft cache
        torch.backends.cuda.cufft_plan_cache.clear()
        torch.backends.cuda.cufft_plan_cache.max_size = 0
        complex_stft = torch.stft(
            x.contiguous(),
            n_fft=n_fft,
            hop_length=n_hop,
            window=window.contiguous(),
            center=False,
            normalized=False,
            onesided=True,
            pad_mode="reflect",
            return_complex=True,
        )
        print("STFT Success with cache 0 and contiguous!")
    except Exception as e:
        print("STFT cache 0 FAILED:", str(e))

if __name__ == "__main__":
    test()
