"""FFTTemporalConv must reproduce the direct Conv2d temporal convolution."""
import copy

import torch

from eegop.models import get_model


def _check(dec, C, T, K, dev):
    torch.manual_seed(0)
    ref = get_model(dec, C, T, K, 250., fft_temporal=False).to(dev).eval()
    torch.manual_seed(0)
    fft = get_model(dec, C, T, K, 250., fft_temporal=True).to(dev).eval()
    # copy all weights; the temporal convolution weight has an identical shape
    fft.load_state_dict(copy.deepcopy(ref.state_dict()))
    x = torch.randn(8, C, T, device=dev, requires_grad=True)
    x2 = x.detach().clone().requires_grad_(True)
    y_ref, y_fft = ref(x), fft(x2)
    err = (y_ref - y_fft).abs().max().item() / (y_ref.abs().max().item() + 1e-12)
    y_ref.sum().backward(); y_fft.sum().backward()
    gerr = (x.grad - x2.grad).abs().max().item() / (x.grad.abs().max().item() + 1e-12)
    n_ref = sum(p.numel() for p in ref.parameters())
    n_fft = sum(p.numel() for p in fft.parameters())
    return err, gerr, n_ref, n_fft


def test_equivalence_cpu():
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    for dec in ("eegnet", "atcnet"):
        for C, T, K in ((22, 1001, 4), (8, 434, 2), (4, 151, 2)):
            err, gerr, n_ref, n_fft = _check(dec, C, T, K, "cpu")
            assert n_ref == n_fft
            assert err < 1e-4, (dec, C, T, err)
            assert gerr < 1e-4, (dec, C, T, gerr)


if __name__ == "__main__":
    test_equivalence_cpu()
    print("cpu equivalence ok")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        for dec in ("eegnet", "atcnet"):
            print(dec, "gpu relative error (output, gradient), parameter counts:",
                  _check(dec, 128, 2001, 4, "cuda"))
