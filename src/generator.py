import jax
import jax.numpy as jnp
from flax import nnx
import jaxtyping as jt
from jax import vmap
from src.ops import filtered_nonlinearity, modulated_convolution, fourier_features, compute_layer_params, equalized_lr, EqualizedLinear
Float=jt.Float
Array=jt.Array

class MappingNetwork(nnx.Module):
    def __init__(self,z_dim:int,w_dim:int,num_ws:int,rngs :nnx.Rngs):
        self.num_ws=num_ws
        layers = []
        layers.append(EqualizedLinear(z_dim,w_dim,rngs=rngs, lr_mul=0.01))
        for layer in range(1):
            layers.append(EqualizedLinear(w_dim,w_dim,rngs=rngs, lr_mul=0.01))
        self.layers=nnx.List(layers)

    def __call__(self, z:Float[Array,"B z_dim"]) ->Float[Array ,"B num_ws w_dim"]:
         ##normalize z first
         z = z / jnp.sqrt(jnp.mean(z**2, axis=-1, keepdims=True) + 1e-8)

         ##mapping networks
         for layer in self.layers:
             z=jax.nn.leaky_relu(layer(z),0.2)

         ##broadcast w to [B, num_ws, w_dim] for per-layer styles
         w=jnp.tile(z[:,None,:],(1,self.num_ws,1))
         return w

class SynthesisLayer(nnx.Module):
    def __init__(self,
                 w_dim:int ,
                 in_channels:int ,
                 out_channels:int,
                 rngs:nnx.Rngs,
                 kernel_size:int,
                 fu,fd,
                 up:int,
                 down:int):

        self.Affine=EqualizedLinear(w_dim,in_channels,rngs=rngs)
        self.kernel_weight=nnx.Param(jax.random.normal(rngs.params(),(out_channels,in_channels,kernel_size,kernel_size)))
        self.bias=nnx.Param(jnp.zeros(out_channels))
        self.magnitude_ema=nnx.Variable(jnp.float32(1.0))

        ##store filter params from compute_layer_params
        ##fu = upsample filter, fd = downsample filter (both 1d, from firwin)
        ##up/down come from lrelu_upsampling=2 (nvidia default)
        self.fu=nnx.Variable(fu)
        self.fd=nnx.Variable(fd)
        self.up=up
        self.down=down
    
    
    @nnx.remat
    def __call__(self,x:Float[Array ,"B  Cin H W"],
                 w:Float[Array ,"B  w_dim"],
                 ) ->Float[Array ,"B Cout H2 W2 "]:

        style=self.Affine(w)

        ##EMA normalization of x 
        batch_var=jax.lax.stop_gradient(jnp.mean(x.astype(jnp.float32)**2))
        self.magnitude_ema.value=jax.lax.stop_gradient(
            0.999 * self.magnitude_ema.value + 0.001 * batch_var)
        input_gain = jax.lax.stop_gradient(
            1.0 / jnp.sqrt(jnp.maximum(self.magnitude_ema.value, 1e-4)))
        x = x * input_gain

        scaled_weight = self.kernel_weight.value * equalized_lr(self.kernel_weight.value)
        grouped_convolution=modulated_convolution(x,scaled_weight,style)
        bias=jnp.reshape(self.bias.value,(1,-1,1,1))
        grouped_convolution=grouped_convolution +bias

        ## up -> fu -> leaky_relu -> fd -> down
        out=filtered_nonlinearity(grouped_convolution,self.fu.value,self.fd.value,self.up,self.down)

        return out

class ToRGB(nnx.Module):
    def __init__(self,w_dim:int
                 ,in_channels:int
                 ,out_channels:int,
                 rngs:nnx.Rngs,
                 kernel_size=1):

        self.out_channels=out_channels
        self.in_channels=in_channels
        self.conv_kernel=kernel_size
        self.Affine=EqualizedLinear(w_dim,in_channels,rngs=rngs)
        self.kernel_weights=nnx.Param(jax.random.normal(rngs.params(),(out_channels,in_channels,kernel_size,kernel_size)))
        self.bias=nnx.Param(jnp.zeros(out_channels))

    def __call__(self,out:Float[Array,"B C H' W' "],w:Float[Array ,"B  w_dim"]):
        style=self.Affine(w)

        ##scale style by 1/sqrt(in_channels * kernel^2)
        style_scale = 1.0 / jnp.sqrt(self.in_channels * (self.conv_kernel**2))
        style = style * style_scale

        B=out.shape[0] ;H=out.shape[2] ; W=out.shape[3]

        scaled_kernel = self.kernel_weights.value * equalized_lr(self.kernel_weights.value)
        
        style = style + 1.0
        modulated = scaled_kernel * jnp.reshape(style,(style.shape[0],1,style.shape[1],1,1))

        ##reshape for grouped convolution
        out=jnp.reshape(out,(1,B*out.shape[1],H,W))
        modulated=jnp.reshape(modulated,(modulated.shape[0]*self.out_channels,modulated.shape[2],
                               modulated.shape[3],modulated.shape[4]))

        grouped_convolution=jax.lax.conv_general_dilated(lhs=out,rhs=modulated,window_strides=(1,1),padding='SAME',feature_group_count=B,dimension_numbers=('NCHW','OIHW','NCHW'))

        output=jnp.reshape(grouped_convolution,(B,self.out_channels,H,W))
        bias=jnp.reshape(self.bias.value,(1,-1,1,1))
        output=output+bias

        return output

class Generator(nnx.Module):
    def __init__(self,z_dim:int
                 ,w_dim:int,
                 freq_channels:int,
                 layer_configs:list,
                 kernel_size:int,
                 rngs:nnx.Rngs,
                 filter_size:int=6,
                 lrelu_upsampling:int=2,
                 output_scale:float=0.25) :

        self.output_scale=output_scale
        ##num_ws = synthesis layers + 1 for toRGB + 1 for transform_affine
        self.num_ws=len(layer_configs) + 2

        freqs = jax.random.normal(rngs.params(), (freq_channels // 2, 2))
        self.fixed_freqs = freqs / jnp.linalg.norm(freqs, axis=-1, keepdims=True)
        self.conv1x1_kernel = nnx.Param(jax.random.normal(rngs.params(),(layer_configs[0][0], freq_channels, 1, 1)))
        self.mapping=MappingNetwork(z_dim,w_dim,self.num_ws,rngs)
        self.transform_affine=EqualizedLinear(w_dim,4,rngs=rngs)

        ##compute per-layer filter params (fu, fd, up, down)
        filter_params=compute_layer_params(len(layer_configs),filter_size,lrelu_upsampling)

        layers=[]
        for idx,(in_ch, out_ch, *_rest) in enumerate(layer_configs):
            ##up/down now come from compute_layer_params, 
            p=filter_params[idx]
            layers.append(SynthesisLayer(w_dim,in_ch,out_ch,rngs,kernel_size,
                                         p['fu'],p['fd'],p['up'],p['down']))

        last_out_ch=layer_configs[-1][1]
        self.layers=nnx.List(layers)
        self.to_RGB=ToRGB(w_dim,last_out_ch,3,rngs)

    def __call__(self,z:Float[Array,"B z_dim"],hi:int ,wi:int) ->Float[Array,"B 3 H W"]:
        ws=self.mapping(z)  ##[B, num_ws, w_dim]

        ##use first w slice for transform affine
        t=self.transform_affine(ws[:,0,:])
        rc=t[:,0:1] ;rs=t[:,1:2] ; tx=t[:,2:3] ;ty=t[:,3:4]

        ##normalize rc and rs
        norm=jnp.sqrt(rc**2+rs**2+1e-8)
        rc=rc/norm ; rs=rs/norm

        fx=self.fixed_freqs[:,0]
        fy=self.fixed_freqs[:,1]
        rot_x=rc*fx-rs*fy
        rot_y=rs*fx+rc*fy
        transformed = jnp.stack([rot_x, rot_y], axis=-1)
        
        ##apply tx and ty directly in the fourier features function call
        x=vmap(lambda f, dx, dy :fourier_features(f,hi,wi,dx,dy))(transformed, tx.flatten(), ty.flatten())

        k_1x1 = self.conv1x1_kernel.value * equalized_lr(self.conv1x1_kernel.value)
        x=jax.lax.conv_general_dilated(x, k_1x1, (1,1), 'SAME', dimension_numbers=('NCHW','OIHW','NCHW'))
        x=x.astype(jnp.bfloat16)

        ##per-layer style indexing: ws[:,1:-1,:] for synthesis, ws[:,-1,:] for toRGB
        for i,layer in enumerate(self.layers):
            x=layer(x,ws[:,1+i,:])

        ##apply output_scale before toRGB 
        if self.output_scale != 1.0:
            x = x * self.output_scale

        x=self.to_RGB(x,ws[:,-1,:])

        return x