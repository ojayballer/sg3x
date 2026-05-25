import jax
import jax.numpy as jnp
from flax import nnx
import jaxtyping as jt
from src.ops import equalized_lr, EqualizedLinear
Float=jt.Float
Array=jt.Array

def _downsample_2d(x:Float[Array,"B C H W"]) -> Float[Array,"B C H2 W2"]:
    #Apply [1,3,3,1] lowpass filter then stride-2 downsample
    ##separable 1D filter [1,3,3,1] / 8 (normalized)
    B,C,H,W=x.shape
    filt=jnp.array([1.0, 3.0, 3.0, 1.0], dtype=x.dtype) / 8.0
    ##horizontal pass: depthwise conv
    kf_h=jnp.tile(filt.reshape(1,1,1,4),(C,1,1,1))
    x=jax.lax.conv_general_dilated(x,kf_h,(1,1),'SAME',feature_group_count=C,dimension_numbers=('NCHW','OIHW','NCHW'))
    ##vertical pass: depthwise conv
    kf_v=jnp.tile(filt.reshape(1,1,4,1),(C,1,1,1))
    x=jax.lax.conv_general_dilated(x,kf_v,(1,1),'SAME',feature_group_count=C,dimension_numbers=('NCHW','OIHW','NCHW'))
    ##stride-2 subsample
    return x[:,:,::2,::2]

class  DiscriminatorBlock(nnx.Module):
    def __init__(self,in_channels:int,out_channels:int,rngs:nnx.Rngs):
        self.kernel1=nnx.Param(jax.random.normal(rngs.params(),(out_channels,in_channels,3,3)))
        self.kernel2=nnx.Param(jax.random.normal(rngs.params(),(out_channels,out_channels,3,3)))
        self.skip=nnx.Param(jax.random.normal(rngs.params(),(out_channels,in_channels,1,1)))

    def __call__(self,x:Float[Array,"B 32 H W "]):
        k1 = self.kernel1.value * equalized_lr(self.kernel1.value)
        conv1=jax.lax.conv_general_dilated(lhs=x,rhs=k1,window_strides=(1,1),padding='SAME',dimension_numbers=('NCHW','OIHW','NCHW'))
        conv1=jax.nn.leaky_relu(conv1,0.2)
        
        k2 = self.kernel2.value * equalized_lr(self.kernel2.value)
        conv2=jax.lax.conv_general_dilated(lhs=conv1,rhs=k2,window_strides=(1,1),padding='SAME',dimension_numbers=('NCHW','OIHW','NCHW'))
        conv2=jax.nn.leaky_relu(conv2,0.2)

        ##anti-aliased downsample (lowpass filterand  stride-2)
        conv2=_downsample_2d(conv2)
        k_skip = self.skip.value * equalized_lr(self.skip.value)
        skip=jax.lax.conv_general_dilated(lhs=x,rhs=k_skip,window_strides=(1,1),padding='SAME',dimension_numbers=('NCHW','OIHW','NCHW'))
        skip=_downsample_2d(skip)

        return (conv2+skip)/jnp.sqrt(2.0)

class Discriminator(nnx.Module):
     def __init__(self,layer_configs:list,rngs:nnx.Rngs):
         self.layer_configs=layer_configs
         self.dense=EqualizedLinear(self.layer_configs[-1][1]*4*4,self.layer_configs[-1][1],rngs=rngs)
         self.dense2=EqualizedLinear(self.layer_configs[-1][1],1,rngs=rngs)
         layers=[]
         for in_ch,out_ch in layer_configs :
             layers.append(DiscriminatorBlock(in_ch,out_ch,rngs))

         last_ch=self.layer_configs[-1][-1]
         self.layers=nnx.List(layers)
         self.kernel=nnx.Param(jax.random.normal(rngs.params(),(last_ch,last_ch+1,3,3)))
         self.fromRGB=nnx.Param(jax.random.normal(rngs.params(),(layer_configs[0][0],3,1,1)))

     def minibatch_stddev(self,x,groups=4):
         B, C, H, W = x.shape
         groups = min(groups, B)
         groups = groups if B % groups == 0 else 1  

         ##split the batch into groups
         y = jnp.reshape(x, (groups, -1, C, H, W))
         y = y - jnp.mean(y, axis=0, keepdims=True)
         y = jnp.mean(y ** 2, axis=0)
         y = jnp.sqrt(y + 1e-8)
         y = jnp.mean(y, axis=(1,2,3), keepdims=True)
         y = jnp.repeat(y, groups, axis=0)
         y = jnp.broadcast_to(y, (B, 1, H, W))
         
         return jnp.concatenate([x,y],axis=1)

     def __call__(self,x:Float[Array,"B 3 H W "])->Float[Array ,"B 1"]:
         k_rgb = self.fromRGB.value * equalized_lr(self.fromRGB.value)
         x=jax.lax.conv_general_dilated(lhs=x,rhs=k_rgb,window_strides=(1,1),padding='SAME',dimension_numbers=('NCHW','OIHW','NCHW'))
         x=jax.nn.leaky_relu(x,0.2)
         
         for layer in self.layers:
             x=layer(x)

         ##minibatch std dev
         x=self.minibatch_stddev(x)
         k_last = self.kernel.value * equalized_lr(self.kernel.value)
         conv=jax.lax.conv_general_dilated(lhs=x,rhs=k_last,window_strides=(1,1),padding='SAME',dimension_numbers=('NCHW','OIHW','NCHW'))
         conv=jax.nn.leaky_relu(conv,0.2)

         ##flatten
         output=jnp.reshape(conv,(conv.shape[0],-1))
         output=self.dense(output)
         output=jax.nn.leaky_relu(output,0.2)
         output=self.dense2(output)
         return output