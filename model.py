import torch
import torch.nn as nn
from pyronn.ct_reconstruction.layers.backprojection_2d import ParallelBackProjection2D


class FourierFilter(nn.Module):
    def __init__(self, num_detectors, number_of_projections):
        super(FourierFilter, self).__init__()

        self.filter_1d = None
        torch.set_default_tensor_type(
            "torch.cuda.FloatTensor" if torch.cuda.is_available() else "torch.FloatTensor"
        )

        self.frequencies = torch.fft.fftfreq(num_detectors)
        self.number_of_projections = number_of_projections
        self.pi = torch.tensor(torch.pi)
        self.num_harmonics = 50

        initial_a_n = torch.tensor(
            [
                22.52177631,
                -14.36855749,
                -15.46172335,
                -13.80045632,
                -7.25623712,
                -3.29460343,
                -0.89431335,
                -0.43939094,
                -0.54756908,
                -0.77002806,
                -0.7294812,
                -0.73489638,
                -0.62724951,
                -0.57507896,
                -0.34530626,
                -0.21673895,
                -0.20965273,
                -0.26026577,
                -0.20969586,
                -0.24222277,
                -0.17879291,
                -0.11930541,
                -0.15606102,
                -0.15482003,
                -0.12276921,
                -0.17289476,
                -0.07536781,
                -0.06925928,
                -0.08416083,
                -0.05711419,
                -0.04268535,
                -0.09427819,
                -0.07077185,
                -0.06556709,
                -0.03610009,
                -0.02205418,
                -0.06557365,
                -0.06891148,
                -0.03277604,
                -0.02013158,
                -0.02870531,
                -0.04042139,
                -0.02883675,
                -0.02608390,
                -0.02797781,
                -0.02348708,
                -0.02043181,
                -0.02118805,
                -0.02051532,
                -0.01986274,
            ]
        )
        initial_b_n = torch.tensor(
            [
                -3.54578831e-15,
                1.77289416e-15,
                2.65934123e-15,
                5.31868247e-15,
                5.09707070e-15,
                5.54029424e-15,
                3.60119125e-15,
                1.10805885e-15,
                0.0,
                0.0,
                2.10531181e-15,
                9.08608255e-15,
                -2.00004622e-14,
                -1.10805885e-15,
                5.01396628e-15,
                -1.55128239e-15,
                -1.93910298e-15,
                1.10805885e-16,
                -2.71474418e-15,
                -4.32142950e-15,
                -8.44894871e-15,
                -9.97252962e-15,
                -2.30199226e-14,
                4.65384716e-15,
                5.31868247e-15,
                -1.10805885e-15,
                8.08882958e-15,
                2.93635595e-15,
                6.86996485e-15,
                2.99175889e-15,
                1.66208827e-15,
                -1.86153886e-14,
                -9.69551491e-15,
                3.93360891e-15,
                -2.29922211e-15,
                1.39061385e-14,
                -1.56402506e-13,
                -1.44047650e-15,
                1.02218429e-14,
                -1.77843445e-14,
                5.54029424e-15,
                1.93356269e-14,
                1.15238120e-14,
                1.20778414e-14,
                1.80890607e-14,
                -6.92536779e-15,
                -6.23283102e-15,
                1.24213397e-13,
                5.64001953e-14,
                2.20503711e-14,
            ]
        )

        self.a_0 = nn.Parameter(torch.tensor([42.8742]), requires_grad=True).cuda()
        self.a_n = nn.Parameter(initial_a_n, requires_grad=True).cuda()
        self.b_n = nn.Parameter(initial_b_n, requires_grad=True).cuda()

    def forward(self, x):
        n = torch.arange(1, self.num_harmonics + 1).unsqueeze(1)

        cos_terms = torch.cos(2 * self.pi * n * self.frequencies.unsqueeze(0))
        sin_terms = torch.sin(2 * self.pi * n * self.frequencies.unsqueeze(0))

        a_n_reshaped = self.a_n.view(-1, 1)
        b_n_reshaped = self.b_n.view(-1, 1)

        self.filter_1d = self.a_0 + torch.sum(
            a_n_reshaped * cos_terms + b_n_reshaped * sin_terms, dim=0
        )

        filter_2d = self.filter_1d.repeat(self.number_of_projections, 1)
        return x * filter_2d


class ParReconstruction2D_Eff(nn.Module):
    def __init__(self, geometry):
        super(ParReconstruction2D_Eff, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.geometry = geometry

        self.filter1 = FourierFilter(
            geometry.detector_shape[-1],
            geometry.number_of_projections,
        ).to(self.device)

        self.AT = ParallelBackProjection2D().to(self.device)

    def forward(self, proj):
        x = torch.fft.fft(proj, dim=-1, norm="ortho")
        x = self.filter1(x)
        proj = torch.fft.ifft(x, dim=-1, norm="ortho").real.float()

        rco = self.AT.forward(proj.contiguous(), **self.geometry)

        rco = nn.ReLU()(rco)

        return rco, proj.contiguous()