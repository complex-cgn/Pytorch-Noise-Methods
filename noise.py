
import torch
import numpy
from dataclasses import dataclass
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"current device={device}")

@dataclass
class Noise:
    # Hyper-Params
    width: int
    height: int
    scale: float = 4.0
    seed: int | None = None
    device: torch.device = device

    # Initalizing variables
    def __post_init__(self):

        # Distance vectors
        self.d00 = torch.empty(self.height, self.width, 2, device=self.device)
        self.d10 = torch.empty_like(self.d00)
        self.d01 = torch.empty_like(self.d00)
        self.d11 = torch.empty_like(self.d00)

        # Gradients
        self.g00 = torch.empty_like(self.d00)
        self.g10 = torch.empty_like(self.d00)
        self.g01 = torch.empty_like(self.d00)
        self.g11 = torch.empty_like(self.d00)

        # Noise Values
        self.n00 = torch.empty(self.height, self.width, device=self.device)
        self.n10 = torch.empty_like(self.n00)
        self.n01 = torch.empty_like(self.n00)
        self.n11 = torch.empty_like(self.n00)

    def fade(self, t):
        # return 3*t**2-2*t**3
        # return 6*t**5-15*t**4+10*t**3
        # return t**3 * (t * (6*t - 15) + 10)
        # return t**4 * (35 - 84*t + 70*t**2 - 20*t**3)
        return t * t * t * (t * (t * 6 - 15) + 10)

    def lerp(self, a, b, t):
        return a + t * (b - a)
    
    def Perlin(self, seed, scale):

        # Match Arguments
        width = self.width
        height = self.height
        device = self.device

        d00 = self.d00
        d10 = self.d10
        d01 = self.d01
        d11 = self.d11

        g00 = self.g00
        g10 = self.g10
        g01 = self.g01
        g11 = self.g11

        n00 = self.n00
        n10 = self.n10
        n01 = self.n01
        n11 = self.n11

        if seed is not None:
            torch.manual_seed(seed)

        # Grid
        x_lin = torch.linspace(0, scale, width , device = device)
        y_lin = torch.linspace(0, scale, height, device = device)
        y, x = torch.meshgrid(y_lin, x_lin, indexing="ij")

        # Random Rotation
        rotation = torch.empty(width+1, height+1, device=device).uniform_(0, 2 * torch.pi) # among 0 - (2 * π)

        x0 = x.floor().long()
        y0 = y.floor().long()
        x1 = x0 + 1
        y1 = y0 + 1

        xf = x - x0
        yf = y - y0

        # Calculate The Distance
        d00[..., 0] = xf
        d00[..., 1] = yf

        d10[..., 0] = xf - 1
        d10[..., 1] = yf

        d01[..., 0] = xf
        d01[..., 1] = yf - 1

        d11[..., 0] = xf - 1
        d11[..., 1] = yf - 1

        # Calculate The Gradient
        g00[..., 0] = rotation[y0, x0].cos()
        g00[..., 1] = rotation[y0, x0].sin()

        g10[..., 0] = rotation[y0, x1].cos()
        g10[..., 1] = rotation[y0, x1].sin()

        g01[..., 0] = rotation[y1, x0].cos()
        g01[..., 1] = rotation[y1, x0].sin()

        g11[..., 0] = rotation[y1, x1].cos()
        g11[..., 1] = rotation[y1, x1].sin()

        # Dot Product
        n00 = (g00 * d00).sum(dim=-1)
        n10 = (g10 * d10).sum(dim=-1)
        n01 = (g01 * d01).sum(dim=-1)
        n11 = (g11 * d11).sum(dim=-1)

        u = self.fade(xf) # u is among to (0 - 1)
        v = self.fade(yf) # same for v

        nx0   = self.lerp(n00, n10, u)
        nx1   = self.lerp(n01, n11, u)
        value = self.lerp(nx0, nx1, v)

        return value # Shape: (height, width)
    
    def perlin_octave_2d(
        self,
        octaves: int = 6,
        persistence: float = 0.5,
        lacunarity: float = 2.0,
        turbulence: bool = False
    ):
        width   =   self.width
        height  =   self.height
        scale   =   self.scale
        seed    =   self.seed
        device  =   self.device

        perlin = self.Perlin

        """
        octaves: How many layers of noise will be added.
        persistence: How much the amplitude will decrease in each octave (Typically 0.5).
        lacunarity: How much the frequency (detail) will increase in each octave (Typically 2.0).
        """

        total_noise = torch.zeros((height, width), device=device)
        amp = 0.1

        for i in range(octaves):
            layer_seed = seed + i if seed is not None else None

            layer = perlin(layer_seed, scale)

            if(turbulence == False): # if turbulence mode is off?
                total_noise += layer * amp # for normal usage
            else:
                gamma = 0.5 # usable when gamma among to (0 - 1)
                total_noise += (torch.abs(layer) ** gamma) * amp
            
            amp *= persistence
            scale *= lacunarity

        return total_noise

start_event = torch.cuda.Event(enable_timing=True)
end_event   = torch.cuda.Event(enable_timing=True)
start_event.record()

PerlinNoise = Noise(
    width=4096, 
    height=4096, 
    scale=4.0,
    seed=None,
    device=device
)

noise = PerlinNoise.perlin_octave_2d()

end_event.record()        # Save Finish Time

torch.cuda.synchronize()  # Synchronized for time measurement
elapsed_time_ms = start_event.elapsed_time(end_event)  # ms type
print(f"GPU time: {elapsed_time_ms} ms")

plt.figure(figsize=(6, 6))
plt.imshow(noise.cpu().numpy(), cmap='gray', origin='upper')
plt.axis('off')
plt.title('Perlin Noise')
plt.savefig("noise.png", dpi=150)
# plt.show()