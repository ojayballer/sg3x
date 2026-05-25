import jax
import jax.numpy as jnp
from jax import scipy
import jaxtyping as jt
from jax import jit
from flax import nnx
import functools
import numpy as np
import scipy.signal as sp_signal

Float=jt.Float
Array=jt.Array

def equalized_lr(param)->float:
    shape = param.shape
    if len(shape) == 4:
        fan_in = shape[1] * shape[2] * shape[3]
    elif len(shape) == 2:
        fan_in = shape[0]
    else:
        fan_in = shape[0]
    return 1.0 / jnp.sqrt(fan_in + 1e-8)

class EqualizedLinear(nnx.Module):
    def __init__(self, in_features: int, out_features: int, rngs: nnx.Rngs, lr_mul: float = 1.0):
        self.kernel = nnx.Param(jax.random.normal(rngs.params(), (in_features, out_features)) / lr_mul)
        self.bias = nnx.Param(jnp.zeros((out_features,), dtype=jnp.float32))
        self.lr_mul = lr_mul

    def __call__(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        k = self.kernel.value * equalized_lr(self.kernel.value) * self.lr_mul
        b = self.bias.value * self.lr_mul
        return jnp.dot(x, k) + b

def design_lowpass_filter(numtaps:int,cutoff:float,width:float,fs:float):
    
    if numtaps<=1:
        return None
    f=sp_signal.firwin(numtaps,cutoff=cutoff,width=width,fs=fs)
    return jnp.array(f,dtype=jnp.float32)

def compute_layer_params(num_layers:int,filter_size:int=6,lrelu_upsampling:int=2)->list:
    ##replaces compute_layer_filters
    ##returns per-layer dicts: fu,fd,up,down,in_size,out_size
    ##
    ##lrelu_upsampling=2 means every layer
    ##computes activation at 2x the sampling rate then downsamples back.
    ##this is what makes stylegan3 alias-free.

    N_crit=2
    s_N=128
    fc_0=2.0
    fc_N=s_N/2
    ft_0=2**2.1
    ft_N=fc_N * 2**0.3

    ##boundaries between layers (num_layers+1 of them)
    rates=[]
    cutoffs=[]
    halfwidths=[]

    for i in range(num_layers+1):
        r=min(i/(num_layers - N_crit),1.0)
        fci=fc_0 * (fc_N/fc_0)**r
        fti=ft_0 * (ft_N/ft_0)**r
        si=min(2**int(np.ceil(np.log2(2*fti))),s_N)
        fhi=max(fti,si/2) - fci
        rates.append(si)
        cutoffs.append(fci)
        halfwidths.append(fhi)

    layers=[]
    for i in range(num_layers):
        in_rate =rates[i]
        out_rate=rates[i+1]

        ##temporary rate = 2x for activation anti-aliasing
        tmp_rate=max(in_rate,out_rate)*lrelu_upsampling
        up =int(tmp_rate/in_rate)
        down=int(tmp_rate/out_rate)

        ##filter taps = filter_size * resample_factor 
        up_taps  =filter_size*up   if up>1   else 1
        down_taps=filter_size*down if down>1 else 1

        ##separate filters: fu at input cutoff, fd at output cutoff
        ##both designed at the temporary sampling rate
        fu=design_lowpass_filter(up_taps, cutoffs[i],  halfwidths[i]*2,  tmp_rate)
        fd=design_lowpass_filter(down_taps,cutoffs[i+1],halfwidths[i+1]*2,tmp_rate)

        layers.append({
            'fu':fu,'fd':fd,
            'up':up,'down':down,
            'in_size':int(in_rate),'out_size':int(out_rate),
        })

    return layers


def apply_filter_1d(x:Float[Array,"B C H W"],kf:Float[Array,"N"],gain:float=1.0)->Float[Array,"B C H W"]:
    ##apply 1d filter separably, horizontal(1xN) then vertical(Nx1)
    ##uses depthwise conv (feature_group_count=C) for TPU MXU acceleration
    
    B,C,H,W=x.shape
    n=kf.shape[0]

    ##scale filter in float32 for precision, then cast to match activation dtype
    kf_scaled=(kf * jnp.sqrt(jnp.float32(gain))).astype(x.dtype)

    ## tile kernel across C channels for depthwise conv
    kf_h=jnp.tile(jnp.reshape(kf_scaled,(1,1,1,n)),(C,1,1,1))
    x=jax.lax.conv_general_dilated(
        lhs=x,rhs=kf_h,window_strides=(1,1),padding='SAME',
        feature_group_count=C,
        dimension_numbers=('NCHW','OIHW','NCHW'))

    ##vertical pass, tile kernel across C channels for depthwise conv
    kf_v=jnp.tile(jnp.reshape(kf_scaled,(1,1,n,1)),(C,1,1,1))
    x=jax.lax.conv_general_dilated(
        lhs=x,rhs=kf_v,window_strides=(1,1),padding='SAME',
        feature_group_count=C,
        dimension_numbers=('NCHW','OIHW','NCHW'))

    return x


def apply_filter_1d_down(x:Float[Array,"B C H W"],kf:Float[Array,"N"],down:int)->Float[Array,"B C Hd Wd"]:
    ##fused filter and  downsample using strided depthwise convolution
    ##avoids filtering at full resolution then discarding 3/4 of pixels
    B,C,H,W=x.shape
    n=kf.shape[0]
    kf_scaled=kf.astype(x.dtype)

    ##horizontal pass->strided conv
    kf_h=jnp.tile(jnp.reshape(kf_scaled,(1,1,1,n)),(C,1,1,1))
    x=jax.lax.conv_general_dilated(
        lhs=x,rhs=kf_h,window_strides=(1,down),padding='SAME',
        feature_group_count=C,
        dimension_numbers=('NCHW','OIHW','NCHW'))

    ##vertical pass->strided conv
    kf_v=jnp.tile(jnp.reshape(kf_scaled,(1,1,n,1)),(C,1,1,1))
    x=jax.lax.conv_general_dilated(
        lhs=x,rhs=kf_v,window_strides=(down,1),padding='SAME',
        feature_group_count=C,
        dimension_numbers=('NCHW','OIHW','NCHW'))

    return x

def filtered_nonlinearity(x:Float[Array,"B C H W"],fu,fd,up:int,down:int,
                          gain:float=1.4142135,slope:float=0.2,clamp:float=256.0)->Float[Array,"B C H2 W2"]:
    ##The operation:
    ##  upsample(zero insert) -> fu(interpolate) -> leaky_relu -> fd and downsample(fused)
    ##
    ##fu = upsample filter (designed at input cutoff)
    ##fd = downsample filter (designed at output cutoff)
    ##gain = sqrt(2) by default (preserves variance through leaky relu)
    ##clamp = 256 (nvidia default, prevents fp16 overflow)
    B,C,H,W=x.shape

    ##upsample by zero insertion
    if up>1:
        x_up=jnp.zeros((B,C,H*up,W*up),dtype=x.dtype)
        x_up=x_up.at[:,:,::up,::up].set(x)
        x=x_up

    ##apply upsample filter with gain=up^2 to compensate zero insertion
    if fu is not None:
        x=apply_filter_1d(x,fu,gain=float(up*up))

    ##leaky relu with gain and clamp
    x=jax.nn.leaky_relu(x,slope) * gain
    x=jnp.clip(x,-clamp,clamp)

    ##fused anti-alias filter and  downsample strided conv
    if down>1 and fd is not None:
        x=apply_filter_1d_down(x,fd,down)
    elif fd is not None:
        x=apply_filter_1d(x,fd)
    elif down>1:
        x=x[:,:,::down,::down]

    return x


def modulated_convolution(x:Float[Array ,"B Cin H W"],w:Float[Array, "Cout Cin K K "],style :Float[Array,"B Cin "]) -> Float[Array,"B Cout H W "]:
    ##pre-normalize weights and styles for bfloat16/fp16 stability (NVIDIA SG3)
    w_f32=w.astype(jnp.float32)
    s_f32=style.astype(jnp.float32)
    w = w * (1.0 / jnp.sqrt(jnp.mean(w_f32**2, axis=[1,2,3], keepdims=True) + 1e-8)).astype(w.dtype)
    style = style * (1.0 / jnp.sqrt(jnp.mean(s_f32**2) + 1e-8)).astype(style.dtype)

    ##apply style scaling
    style = style + 1.0

    ##broadcasting
    modulated= w * jnp.reshape(style,(style.shape[0],1,style.shape[1],1,1)) #(B,cout,cin,K,K)
    demodulated = modulated / jnp.sqrt(jnp.sum(modulated.astype(jnp.float32)**2, axis=[2,3,4], keepdims=True) + 1e-8).astype(modulated.dtype)
    kernel_weights=demodulated

    ##grouped convolution instead of vmap
    B=x.shape[0]  ; Cout=kernel_weights.shape[1] ;H=x.shape[2] ; W=x.shape[3]
    x=jnp.reshape(x,(1,B*x.shape[1],H,W))
    kernel_weights=jnp.reshape(kernel_weights,(kernel_weights.shape[0]*Cout,kernel_weights.shape[2],
                               kernel_weights.shape[3],kernel_weights.shape[4]))
    grouped_convolution=jax.lax.conv_general_dilated(lhs=x,rhs=kernel_weights,window_strides=(1,1),padding='SAME',feature_group_count=B,dimension_numbers=('NCHW','OIHW','NCHW'))
    output=jnp.reshape(grouped_convolution,(B,Cout,H,W))
    return output

def fourier_features(freqs:Float[Array,"C2 two"],h:int,w:int,tx:float=0.0,ty:float=0.0)-> Float[Array,"C H W " ]:
    x=jnp.linspace(-1,1,h)
    y=jnp.linspace(-1,1,w)
    gx,gy=jnp.meshgrid(x,y,indexing='ij')

    ##apply phase shifts
    gx = gx + tx
    gy = gy + ty

    coords=jnp.stack([gx,gy],axis=-1)
    output=coords @ freqs.T
    sin=jnp.sin(2*jnp.pi*output)
    cos=jnp.cos(2*jnp.pi*output)
    joined=jnp.concatenate([sin,cos],axis=-1)
    joined=jnp.transpose(joined,(2,0,1))
    return joined