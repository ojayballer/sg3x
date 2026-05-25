import jax
import jax.numpy as jnp
import jaxtyping as jt
from jax import vmap

Float=jt.Float
Array=jt.Array

def apply_geometric_augment(image:Float[Array,"B C H W"], p:float, key) ->Float[Array,"B C H W"]:
    B = image.shape[0]
    
    ##split PRNG keys 
    key, k_aug, k_xflip, k_rot, k_trans = jax.random.split(key, 5)
    
    do_aug = jax.random.uniform(k_aug, (B,)) < p

    ##X-flip
    do_xflip = jax.random.uniform(k_xflip, (B,)) < 0.5
    image = jnp.where((do_xflip & do_aug)[:, None, None, None], image[:, :, :, ::-1], image)

    ##Rotation (90 degrees)
    do_rot90 = jax.random.uniform(k_rot, (B,)) < 0.25
    image = jnp.where((do_rot90 & do_aug)[:, None, None, None], jnp.swapaxes(image, 2, 3)[:, :, ::-1, :], image)

    ##Integer translation
    max_shift = 0.125
    k_tx, k_ty = jax.random.split(k_trans)
    tx = jax.random.uniform(k_tx, (B,)) * max_shift * 2 - max_shift
    ty = jax.random.uniform(k_ty, (B,)) * max_shift * 2 - max_shift
    
    shift_x = jnp.round(tx * image.shape[3]).astype(jnp.int32)
    shift_y = jnp.round(ty * image.shape[2]).astype(jnp.int32)
    
    ##vectorized image rolling
    image = vmap(lambda img, sx, sy, aug: jnp.where(aug, jnp.roll(jnp.roll(img, sx, axis=2), sy, axis=1), img))(image, shift_x, shift_y, do_aug)

    return image

def apply_color_augment(image:Float[Array,"B C H W"], p:float, key) ->Float[Array,"B C H W"]:
    B = image.shape[0]
    
    ##split PRNG keys
    key, k_aug, k_bright, k_contrast, k_sat = jax.random.split(key, 5)
    
    do_aug = jax.random.uniform(k_aug, (B,)) < p

    ##Brightness(add Gaussian noise)
    brightness = jax.random.normal(k_bright, (B,)) * 0.2
    brightness = jnp.where(do_aug, brightness, 0.0)[:, None, None, None]
    image = image + brightness

    ##Contrast(rescale around mean) 
    log_contrast = jax.random.normal(k_contrast, (B,)) * 0.5
    contrast = jnp.clip(jnp.exp2(log_contrast), 0.1, 10.0)
    contrast = jnp.where(do_aug, contrast, 1.0)[:, None, None, None]
    mean = jnp.mean(image, axis=(1, 2, 3), keepdims=True)
    image = (image - mean) * contrast + mean

    ##Saturation(blend with grayscale) 
    log_sat = jax.random.normal(k_sat, (B,)) * 1.0
    saturation = jnp.clip(jnp.exp2(log_sat), 0.1, 10.0)
    saturation = jnp.where(do_aug, saturation, 1.0)[:, None, None, None]
    gray = jnp.mean(image, axis=1, keepdims=True)
    image = gray + (image - gray) * saturation

    return image

def apply_hue_rotation(image:Float[Array,"B C H W"], p:float, key) ->Float[Array,"B C H W"]:
    ##rotate RGB colors around the luma axis [1,1,1]/sqrt(3)
    ##nvidia's standard hue augmentation for all small-dataset configs
    B,C,H,W = image.shape

    key, k_aug, k_theta = jax.random.split(key, 3)
    do_aug = jax.random.uniform(k_aug, (B,)) < p
    theta = (jax.random.uniform(k_theta, (B,)) * 2 - 1) * jnp.pi
    theta = jnp.where(do_aug, theta, 0.0)

    ##rodrigues rotation around v=[1,1,1]/sqrt(3)
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    one_third = 1.0 / 3.0
    inv_sqrt3 = 1.0 / jnp.sqrt(3.0)

    ##build per-sample 3x3 rotation matrix
    R00 = c + (1 - c) * one_third
    R01 = (1 - c) * one_third - s * inv_sqrt3
    R02 = (1 - c) * one_third + s * inv_sqrt3
    R10 = (1 - c) * one_third + s * inv_sqrt3
    R11 = c + (1 - c) * one_third
    R12 = (1 - c) * one_third - s * inv_sqrt3
    R20 = (1 - c) * one_third - s * inv_sqrt3
    R21 = (1 - c) * one_third + s * inv_sqrt3
    R22 = c + (1 - c) * one_third

    ##[B,3,3] rotation matrix
    R = jnp.stack([
        jnp.stack([R00, R01, R02], axis=-1),
        jnp.stack([R10, R11, R12], axis=-1),
        jnp.stack([R20, R21, R22], axis=-1)
    ], axis=-2)

    ##apply: [B,3,HW] = [B,3,3] @ [B,3,HW]
    flat = jnp.reshape(image, (B, 3, H * W))
    rotated = jnp.matmul(R, flat)
    return jnp.reshape(rotated, (B, C, H, W))

def apply_luma_flip(image:Float[Array,"B C H W"], p:float, key) ->Float[Array,"B C H W"]:
    ##householder reflection across luma plane
    ##nvidia: I - 2*v*v^T where v=[1,1,1]/sqrt(3)
    B,C,H,W = image.shape

    key, k_aug, k_flip = jax.random.split(key, 3)
    do_aug = jax.random.uniform(k_aug, (B,)) < p
    do_flip = jax.random.uniform(k_flip, (B,)) < 0.5
    mask = (do_aug & do_flip)

    ##reflection matrix: R = I - 2/3 * ones(3,3)
    R = jnp.array([[1.0/3, -2.0/3, -2.0/3],
                    [-2.0/3, 1.0/3, -2.0/3],
                    [-2.0/3, -2.0/3, 1.0/3]])

    flat = jnp.reshape(image, (B, 3, H * W))
    flipped = jnp.matmul(R, flat)
    result = jnp.where(mask[:, None, None], flipped, flat)
    return jnp.reshape(result, (B, C, H, W))

def apply_cutout(image:Float[Array,"B C H W"], p:float, key) ->Float[Array,"B C H W"]:
    B = image.shape[0]
    
    ##split PRNG keys
    key, k_aug, k1, k2 = jax.random.split(key, 4)
    
    do_cutout = jax.random.uniform(k_aug, (B,)) < p
    cx = jax.random.uniform(k1, (B,))
    cy = jax.random.uniform(k2, (B,))
    cutout_size = 0.5  ##nvidia default (was 0.2, too small)
    
    H, W = image.shape[2], image.shape[3]
    xs = jnp.arange(W, dtype=jnp.float32) / W
    ys = jnp.arange(H, dtype=jnp.float32) / H

    mask_x = jnp.abs(xs[None, :] - cx[:, None]) >= cutout_size / 2
    mask_y = jnp.abs(ys[None, :] - cy[:, None]) >= cutout_size / 2
    mask = jnp.logical_or(mask_y[:, :, None], mask_x[:, None, :])
    mask = mask[:, None, :, :].astype(jnp.float32)
    
    return jnp.where(do_cutout[:, None, None, None], image * mask, image)

def add_noise(image:Float[Array,"B C H W"], p:float, key, noise_std:float=0.1) ->Float[Array,"B C H W"]:
    B = image.shape[0]
    
    ##split PRNG keys
    key, k_flag, k_sigma, k_noise = jax.random.split(key, 4)
    
    do_noise = jax.random.uniform(k_flag, (B,)) < p
    ##per-sample noise std (nvidia samples from half-normal)
    sigma = jnp.abs(jax.random.normal(k_sigma, (B, 1, 1, 1))) * noise_std
    sigma = jnp.where(do_noise[:, None, None, None], sigma, 0.0)
    noise = jax.random.normal(k_noise, image.shape) * sigma
    return image + noise

def augment_pipe(images:Float[Array,"B C H W"], p:float, key) ->Float[Array,"B C H W"]:
    keys = jax.random.split(key, 7)
    ##geometric (nvidia pixel blitting tier)
    images = apply_geometric_augment(images, p, keys[0])
    ##color (nvidia standard: brightness, contrast, saturation, hue, luma flip)
    images = apply_color_augment(images, p, keys[1])
    images = apply_hue_rotation(images, p, keys[2])
    images = apply_luma_flip(images, p, keys[3])
    ##corruptions
    images = apply_cutout(images, p, keys[4])
    images = add_noise(images, p * 0.5, keys[5])
    return images

def update_p(p:float, real_scores:Float[Array,"B 1"], r_t_ema:float, ada_target:float=0.6, ada_kimg:int=500, batch_size:int=32):
    r_t = jnp.mean(jnp.sign(real_scores))
    r_t_ema = 0.995 * r_t_ema + 0.005 * r_t
    ##nvidia formula: batch_size / (ada_kimg * 1000) per update
    adjust = jnp.sign(r_t_ema - ada_target) * batch_size / (ada_kimg * 1000)
    p = p + adjust
    p = jnp.clip(p, 0.0, 0.9)
    return p, r_t_ema