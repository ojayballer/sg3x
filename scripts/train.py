import os
import jax
from PIL import Image
import jax.numpy as jnp
from src.generator import Generator
from src.discriminator import Discriminator
import numpy as np
from flax import nnx
import optax
import functools
from src.augment import  augment_pipe,update_p
from src.utils import EMA_update,save_checkpoint
import time
from src.utils import generate_images
import yaml
import jaxtyping as jt
from src.utils import load_checkpoint
Float=jt.Float
Array=jt.Array

with open('configs/default.yml','r') as f :
   config=yaml.safe_load(f)

jax_conv = jax.lax.conv_general_dilated

def cast_conv(lhs, rhs, *args, **kwargs):
    rhs = rhs.astype(lhs.dtype)
    return jax_conv(lhs, rhs, *args, **kwargs)

jax.lax.conv_general_dilated = cast_conv

def load_data(path:str) ->Float[Array,"N 3 H W"]:
    image_path=[os.path.join(path,f) for f in os.listdir(path)]  
    images=[]
    for img_path in image_path:
      image=Image.open(img_path)
      image=image.resize((128,128))
      image=jnp.array(image)/127.5 -1
      image=jnp.transpose(image,(2,0,1)).astype(jnp.float32)
      images.append(image)

    data=jnp.array(images)
    ##shuffle to prevent class-ordered batches causing mode oscillation
    perm=np.random.permutation(data.shape[0])
    return data[perm]

def intialise(config):
   G=Generator(config['generator']['z_dim'],
               config['generator']['w_dim'],
               config['generator']['freq_channels'],
               config['generator']['layers'],
               config['generator']['kernel_size'],
               nnx.Rngs(config['training']['seed']),
               config['generator']['filter_size'],
               config['generator']['lrelu_upsampling'])

   D=Discriminator(config['discriminator']['d_layers'],
                   rngs=nnx.Rngs(config['training']['seed']+1))

   G_ema_params=jax.tree.map(lambda x: x.copy() ,nnx.state(G,nnx.Param))
   
   mb_ratio=16/17
   G_opt=optax.chain(optax.clip_by_global_norm(10.0), optax.adam(learning_rate=config['training']['g_lr'],b1=0,b2=0.99,eps=1e-8))
   D_opt=optax.chain(optax.clip_by_global_norm(10.0), optax.adam(learning_rate=config['training']['d_lr']*mb_ratio,b1=0,b2=0.99**mb_ratio,eps=1e-8))

   G_opt_state=G_opt.init(nnx.state(G,nnx.Param))
   D_opt_state=D_opt.init(nnx.state(D,nnx.Param))

   fixed_z=jax.random.normal(jax.random.PRNGKey(0),(16,config['generator']['z_dim']))

   return G,D,G_ema_params,G_opt,D_opt,G_opt_state,D_opt_state,fixed_z

def step_up(G,D,G_ema_params,G_opt,D_opt,G_opt_state,D_opt_state,
            batch:Float[Array,"B 3 H W"],p,r_t_ema,step:int,key,z_dim:int,r1_gamma:float):

   batch=batch.astype(jnp.bfloat16)
   key1,key2,key3,key4,key5=jax.random.split(key,5)
   z=jax.random.normal(key1,(batch.shape[0],z_dim))
   fake_images=G(z,16,16)

   real_aug=jax.lax.stop_gradient(augment_pipe(batch,p,key2))
   fake_aug_d=jax.lax.stop_gradient(augment_pipe(jax.lax.stop_gradient(fake_images),p,key3))

   def loss_fn(D):
      D_real=D(real_aug).astype(jnp.float32)
      D_fake=D(fake_aug_d).astype(jnp.float32)
      D_loss=jnp.mean(jax.nn.softplus(-D_real)) + jnp.mean(jax.nn.softplus(D_fake))

      def r1():
         grads=jax.grad(lambda x:jnp.sum(D(x).astype(jnp.float32)))(batch)
         grads=grads.astype(jnp.float32)
         grads=jnp.reshape(grads,(grads.shape[0],-1))
         return jnp.mean(jnp.sum(grads**2,axis=1))

      R1=jax.lax.cond((step % 16) == 0,
                   r1,
                   lambda:jnp.float32(0.0))

      final_d_loss = D_loss +r1_gamma *0.5 * R1*16
      ##add NaN guards to prevent multi-hour TPU run crashes
      return jnp.nan_to_num(final_d_loss, nan=10.0, posinf=1e4, neginf=-1e4)

   D_loss,D_grads=nnx.value_and_grad(loss_fn)(D)
   D_param_grads=nnx.state(D_grads,nnx.Param)
   updates,D_opt_state =D_opt.update(D_param_grads,D_opt_state,nnx.state(D,nnx.Param))
   nnx.update(D,optax.apply_updates(nnx.state(D,nnx.Param),updates))

   z=jax.random.normal(key4,(batch.shape[0],z_dim))

   def loss_fn_g(G):
      fake_images=G(z,16,16)
      ##augment fakes for G loss so D evaluates from the same distribution it was trained on
      fake_aug_g=augment_pipe(fake_images,p,key5)
      D_fake=D(fake_aug_g).astype(jnp.float32)
      G_loss=jnp.mean(jax.nn.softplus(-D_fake))
      ##add NaN guards
      return jnp.nan_to_num(G_loss, nan=10.0, posinf=1e4, neginf=-1e4)

   G_loss,G_grads=nnx.value_and_grad(loss_fn_g)(G)
   G_param_grads=nnx.state(G_grads,nnx.Param)
   updates,G_opt_state=G_opt.update(G_param_grads,G_opt_state,nnx.state(G,nnx.Param))
   nnx.update(G,optax.apply_updates(nnx.state(G,nnx.Param),updates))

   G_ema_params=EMA_update((nnx.state(G,nnx.Param)),G_ema_params)

   p,r_t_ema=jax.lax.cond((step % 4) == 0,
                  lambda :update_p(p,D(real_aug),r_t_ema,batch_size=batch.shape[0]),
                  lambda: (p,r_t_ema) )
   return G,D,G_ema_params,G_opt_state,D_opt_state,p,r_t_ema,D_loss,G_loss

def train(config,data,curr_step=None):
   G,D,G_ema_params,G_opt,D_opt,G_opt_state,D_opt_state,fixed_z=intialise(config)
   os.makedirs('outputs', exist_ok=True)
   
    
   p=jnp.float32(0.0) ; r_t_ema=jnp.float32(0.0)
    #resuming chekpoint,this makes it easier to load pretrained chekpoints 
   if curr_step is not  None :                                   #step,#p
       G_params, D_params, G_ema_params, G_opt_state, D_opt_state, _, loaded_p, loaded_rt=load_checkpoint(curr_step)
       nnx.update(G,G_params)
       nnx.update(D,D_params)
       p=jnp.float32(loaded_p)
       r_t_ema=jnp.float32(loaded_rt)
       start_step=curr_step
   else:
       start_step=0



   mesh = jax.sharding.Mesh(jax.devices(), ('batch',))
   data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('batch'))
   data = jax.device_put(data, data_sharding)

   batch_size=config['training']['batch_size']
   total_steps=(config['training']['total_kimg']*1000)//batch_size
   total_batches=data.shape[0]//batch_size
   z_dim=config['generator']['z_dim']
   r1_gamma=config['training']['r1_gamma']

  
   key=jax.random.PRNGKey(config['training']['seed'])
   start=time.time()
   last_checkpoint=time.time()

   compiled_step=nnx.jit(step_up,static_argnums=(3,4,12,13))

   for step in range(start_step,total_steps):
      ##re-shuffle data at each epoch boundary to prevent mode oscillation
      epoch_idx = step % total_batches
      if epoch_idx == 0 and step > 0:
          perm = jax.random.permutation(jax.random.PRNGKey(step), data.shape[0])
          data = data[perm]
      batch=data[epoch_idx *batch_size:(epoch_idx+1)*batch_size]
      key,subkey=jax.random.split(key)


      G,D,G_ema_params,G_opt_state,D_opt_state,p,r_t_ema,D_loss,G_loss=compiled_step(G,D,G_ema_params,G_opt,D_opt,G_opt_state,D_opt_state,batch,p,r_t_ema,jnp.int32(step),
     subkey,z_dim,r1_gamma)
      


      if step % config['logging']['log_interval'] ==0:
         kimg=(step*batch_size)/1000
         elapsed=time.time()-start
         sps=max(step,1)/max(elapsed,1)
         print(f"step {step}|kimg {kimg:.1f}|D:{D_loss.item():.4f}|G:{G_loss.item():.4f}|p:{p.item():.3f}|r_t:{r_t_ema.item():.3f}|{elapsed:.0f}s|{sps:.2f}stp/s")
         
      ''' if step % config['logging']['sample_interval'] == 0 and step > 0:
         generate_images(G, fixed_z, step)'''
      #display images while training to see improvement using EMA weights for better quality
      if step % config['logging']['sample_interval'] == 0 and step > 0:
         ##temporarily swap in EMA params for generation.
         ##we capture the FULL state (nnx.state(G)) so we can restore magnitude_ema too.
         live_state = nnx.state(G)
         nnx.update(G, G_ema_params)
         
         generate_images(G, fixed_z, step)
         
         ##swap the full live state back, undoing any inference side-effects
         nnx.update(G, live_state)
         
         from IPython.display import display, Image as IPImage
         img_path = f"outputs/sample_{step}.png"
         if os.path.exists(img_path):
            display(IPImage(filename=img_path, width=512))

      if time.time() -last_checkpoint >config['logging']['checkpoint_interval']:
        save_checkpoint(nnx.state(G),nnx.state(D),G_ema_params,G_opt_state,D_opt_state,step,p,r_t_ema)
        last_checkpoint=time.time()

   save_checkpoint(nnx.state(G),nnx.state(D),G_ema_params,G_opt_state,D_opt_state,total_steps,p,r_t_ema)