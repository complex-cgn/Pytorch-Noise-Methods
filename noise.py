import torch
import numpy
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"current device={device}")


def fade(t):
    return 3*t**2-2*t**3

def lerp(a, b, t):
    return a + t * (b - a)

def perlin_2d(
    width: int,
    height: int,
    scale: float = 5.0,
    seed: int | None = None,
    device: torch.device = device
):
    if seed is not None:
        torch.manual_seed(seed)
    
    x_lin = torch.linspace(0, scale, width, device=device, dtype=torch.float32) # (0 - scale) there will be as many different values as the width variable
    y_lin = torch.linspace(0, scale, height, device=device, dtype=torch.float32)
    y, x  = torch.meshgrid(y_lin, x_lin, indexing='ij')

    x0 = x.floor().long()
    y0 = y.floor().long()
    x1 = x0 + 1
    y1 = y0 + 1

    xf = x - x0
    yf = y - y0
    

    # Random rotation
    rotation = torch.rand(width+1, height+1, device=device) * (2 * torch.pi) # among 0 - (2 * π)
    grad = torch.stack([rotation.cos(), rotation.sin()], dim=-1)

    g00 = grad[y0, x0]
    g01 = grad[y1, x0]
    g10 = grad[y0, x1]
    g11 = grad[y1, x1]

    d00 = torch.stack([xf    , yf    ], dim=-1)
    d10 = torch.stack([xf - 1, yf    ], dim=-1)
    d01 = torch.stack([xf    , yf - 1], dim=-1)
    d11 = torch.stack([xf - 1, yf - 1], dim=-1)

    n00 = (g00 * d00).sum(dim=-1)
    n10 = (g10 * d10).sum(dim=-1)
    n01 = (g01 * d01).sum(dim=-1)
    n11 = (g11 * d11).sum(dim=-1)

    u = fade(xf) # u is among to (0 - 1)
    v = fade(yf) # same for v

    nx0 = lerp(n00, n10, u)
    nx1 = lerp(n01, n11, u)
    value = lerp(nx0, nx1, v)
    return value # Shape: (height, width)

def perlin_octave_2d(
    width: int,
    height: int,
    base_scale: float = 5.0,
    octaves: int = 6,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    seed: int | None = None,
    turbulence: bool = False,
    device: torch.device = device
):
    """
    octaves: How many layers of noise will be added.
    persistence: How much the amplitude will decrease in each octave (Typically 0.5).
    lacunarity: How much the frequency (detail) will increase in each octave (Typically 2.0).
    """
    total_noise = torch.zeros((height, width), device=device)
    amp = 0.1
    scale = base_scale
    max_value = 0.0

    for i in range(octaves):
        layer_seed = seed + i if seed is not None else None

        layer = perlin_2d(width, height, scale, seed=layer_seed, device=device)

        if(turbulence == False): # if turbulence mode is off?
            total_noise += layer * amp # for normal usage
        else:
            gamma = 0.5 # usable when gamma among to (0 - 1)
            total_noise += (torch.abs(layer) ** gamma) * amp
        
        max_value += amp
        amp *= persistence
        scale *= lacunarity

    # To scale the result between -1 and 1 (or normalize it to 0-1)
    return total_noise / max_value

noise = perlin_octave_2d(
    width=1024, 
    height=1024, 
    base_scale=4.0, 
    octaves=6, 
    persistence=0.6, 
    lacunarity=2.0,
    seed=None
)

plt.figure(figsize=(6, 6))
plt.imshow(noise.cpu().numpy(), cmap='gray', origin='upper')
plt.axis('off')
plt.title('Perlin Noise')
plt.savefig("noise.png", dpi=150)