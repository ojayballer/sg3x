import jax
import jax.numpy as jnp
import numpy as np
import pickle
from PIL import Image
import jaxtyping as jt
import os 

Float=jt.Float
Array=jt.Array

def EMA_update(G_params, G_ema_params, decay:float=0.999):
    return jax.tree.map(lambda p, p_ema: decay * p_ema + (1 - decay) * p, G_params, G_ema_params)

def EMA_update_betas(G_params, G_ema_params, beta, cur_nimg:int, ema_kimg:int, ema_rampup:float=0.05):
    ema_nimg = ema_kimg * 1000
    if ema_rampup is not None:
        ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
    beta = 0.5 ** (1 / max(ema_nimg, 1e-8))
    return jax.tree.map(lambda p, p_ema: p * (1 - beta) + p_ema * beta, G_params, G_ema_params), beta

def save_checkpoint(G_params, D_params, G_ema_params, G_opt_state, D_opt_state,
                    step:int, p:float, r_t_ema:float=0.0, path:str='outputs') -> None:
    # Ensure the directory exists
    os.makedirs(path, exist_ok=True)
    
    checkpoint = {
        'G': jax.tree.map(lambda x: np.array(x), G_params),
        'D': jax.tree.map(lambda x: np.array(x), D_params),
        'G_ema_params': jax.tree.map(lambda x: np.array(x), G_ema_params),
        'G_opt_state': jax.tree.map(lambda x: np.array(x), G_opt_state),
        'D_opt_state': jax.tree.map(lambda x: np.array(x), D_opt_state),
        'step': int(step),
        'p': float(p),
        'r_t_ema': float(r_t_ema)
    }
    
    filename = os.path.join(path, f'chkpt_{step}.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(checkpoint, f)
    print(f"Checkpoint saved: {filename}")




def load_checkpoint(step:int, path:str='outputs/'):
    with open(f'{path}/chkpt_{step}.pkl', 'rb') as f:
        checkpoint = pickle.load(f)
    G_params = jax.tree.map(lambda x: jnp.array(x), checkpoint['G'])
    D_params = jax.tree.map(lambda x: jnp.array(x), checkpoint['D'])
    G_ema_params = jax.tree.map(lambda x: jnp.array(x), checkpoint['G_ema_params'])
    G_opt_state = jax.tree.map(lambda x: jnp.array(x), checkpoint['G_opt_state'])
    D_opt_state = jax.tree.map(lambda x: jnp.array(x), checkpoint['D_opt_state'])
    step = checkpoint['step']
    p = checkpoint['p']
    
    r_t_ema = checkpoint.get('r_t_ema', 0.0)
    return G_params, D_params, G_ema_params, G_opt_state, D_opt_state, step, p, r_t_ema

def normalize_images(images, drange=(-1, 1)):
    lo, hi = drange
    images = (images - lo) * (255 / (hi - lo))
    images = np.clip(np.rint(images), 0, 255).astype(np.uint8)
    return images

def generate_images(G_ema, z, step, path='outputs/', drange=(-1, 1), grid_size=None):
    images = G_ema(z, 16, 16)
    ##map from output range to [0, 255]
    lo, hi = drange
    images = (images - lo) / (hi - lo)
    images = jnp.clip(images, 0, 1)
    images = (np.array(images) * 255).astype(np.uint8)  
    B = images.shape[0]
    if grid_size is None:
        cols = int(np.sqrt(B))
        rows = (B + cols - 1) // cols
    else:
        rows, cols = grid_size
    img_rows = []
    for r in range(rows):
        row_images = images[r * cols:(r + 1) * cols]
        row = np.concatenate(row_images, axis=2)
        img_rows.append(row)
    grid = np.concatenate(img_rows, axis=1)
    grid = np.transpose(grid, (1, 2, 0))
    Image.fromarray(grid).save(f'{path}/sample_{step}.png')

def save_image_grid(images, fname:str, drange=(0, 255)):
    lo, hi = drange
    img = np.asarray(images, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)
    N, C, H, W = img.shape
    gw = int(np.sqrt(N))
    gh = (N + gw - 1) // gw
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape([gh * H, gw * W, C])
    if C == 1:
        Image.fromarray(img[:, :, 0], 'L').save(fname)
    else:
        Image.fromarray(img, 'RGB').save(fname)
        