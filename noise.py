
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
    grayscale: bool = True

    # Initalizing variables
    def __post_init__(self):

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

        n00, n10, n01, n11 = self.n00, self.n10, self.n01, self.n11

        if seed is not None:
            torch.manual_seed(seed)

        # Grid
        x_lin = torch.linspace(0, scale, width , device = device)
        y_lin = torch.linspace(0, scale, height, device = device)
        y, x = torch.meshgrid(y_lin, x_lin, indexing="ij")

        # Random Rotation
        grid_w = int(scale) + 2
        grid_h = int(scale) + 2

        rotation = torch.empty(
            grid_h, grid_w, device=device
        ).uniform_(0, 2 * torch.pi)

        x0 = x.to(torch.int64)
        y0 = y.to(torch.int64)
        x1 = x0 + 1
        y1 = y0 + 1

        xf = x - x0
        yf = y - y0

        # Dot Product
        n00 = rotation[y0, x0].cos()*xf     + rotation[y0, x0].sin()*yf
        n10 = rotation[y0, x1].cos()*(xf-1) + rotation[y0, x1].sin()*yf
        n01 = rotation[y1, x0].cos()*xf     + rotation[y1, x0].sin()*(yf-1)
        n11 = rotation[y1, x1].cos()*(xf-1) + rotation[y1, x1].sin()*(yf-1)

        u = self.fade(xf) # u is among to (0 - 1)

        value = self.lerp(
            self.lerp(n00, n10, u),
            self.lerp(n01, n11, u),
            self.fade(yf)
        )

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
    
    def white_noise_2d(self):
        if self.grayscale is True:
            return torch.rand(self.height, self.width)
        else:
            return torch.rand(3, self.height, self.width)

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

noise = PerlinNoise.perlin_octave_2d(8)

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
