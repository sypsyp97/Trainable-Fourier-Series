import torch
import torch.nn.functional as F


def gaussian_highpass_weights(size, cutoff=0.5, strength=1):
    # Generate a grid of frequencies
    freq_x = torch.fft.fftfreq(size[0]).reshape(-1, 1).repeat(1, size[1])
    freq_y = torch.fft.fftfreq(size[1]).reshape(1, -1).repeat(size[0], 1)

    # Calculate the magnitude of the frequencies
    freq_mag = torch.sqrt(freq_x ** 2 + freq_y ** 2)

    # Gaussian high-pass filter
    weights = torch.exp(-0.5 * ((freq_mag - cutoff) ** 2) / (strength ** 2))
    weights = 1 - weights  # Inverting to make it a high-pass filter

    return weights


def gee_loss(pred, gt):
    pred_fft = torch.fft.fft2(pred)
    gt_fft = torch.fft.fft2(gt)

    pred_mag = torch.sqrt(pred_fft.real ** 2 + pred_fft.imag ** 2)
    gt_mag = torch.sqrt(gt_fft.real ** 2 + gt_fft.imag ** 2)

    # Apply frequency weights
    weights = gaussian_highpass_weights(pred.size(), cutoff=0.5, strength=1).cuda()
    weighted_pred_mag = weights * pred_mag
    weighted_gt_mag = weights * gt_mag

    loss = F.l1_loss(weighted_pred_mag, weighted_gt_mag)
    return loss
