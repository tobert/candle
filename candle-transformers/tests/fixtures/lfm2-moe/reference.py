"""Independent NumPy reference; regenerate with llama.cpp gguf-py on PYTHONPATH.
No Candle operations. Tiny hybrid: conv/dense, attention/MoE, conv/MoE.
"""
from pathlib import Path
import json
import numpy as np
from gguf import GGUFWriter
root=Path(__file__).parent
w=GGUFWriter(str(root/'tiny.gguf'),'lfm2moe')
config={'block_count':3,'context_length':32,'embedding_length':8,'feed_forward_length':12,
        'attention.head_count':2,'leading_dense_block_count':1,'expert_count':3,
        'expert_used_count':2,'expert_feed_forward_length':4,'vocab_size':16,'shortconv.l_cache':3,
        'expert_gating_func':2}
for key,val in config.items(): w.add_uint32('lfm2moe.'+key,val)
w.add_array('lfm2moe.attention.head_count_kv',[0,1,0])
w.add_float32('lfm2moe.rope.freq_base',5000000.)
w.add_float32('lfm2moe.attention.layer_norm_rms_epsilon',1e-5)
weights={}
counter=0
def weight(name,shape,norm=False):
 global counter
 counter+=1
 a=(np.sin(np.arange(np.prod(shape),dtype=np.float32)*.7+counter)*.15).reshape(shape)
 if norm: a+=1
 weights[name]=a
 w.add_tensor(name,a)
 return a
weight('token_embd.weight',(16,8))
weight('token_embd_norm.weight',(8,),True)
for i in range(3):
 p=f'blk.{i}.'
 weight(p+'attn_norm.weight',(8,),True)
 weight(p+'ffn_norm.weight',(8,),True)
 if i==0:
  for n,s in [('ffn_gate',(12,8)),('ffn_up',(12,8)),('ffn_down',(8,12))]: weight(p+n+'.weight',s)
 else:
  weight(p+'ffn_gate_inp.weight',(3,8))
  weight(p+'exp_probs_b.bias',(3,))
  for n,s in [('ffn_gate_exps',(3,4,8)),('ffn_up_exps',(3,4,8)),('ffn_down_exps',(3,8,4))]: weight(p+n+'.weight',s)
 if i==1:
  for n,s in [('attn_q',(8,8)),('attn_k',(4,8)),('attn_v',(4,8)),('attn_output',(8,8))]: weight(p+n+'.weight',s)
  weight(p+'attn_q_norm.weight',(4,),True);weight(p+'attn_k_norm.weight',(4,),True)
 else:
  weight(p+'shortconv.in_proj.weight',(24,8));weight(p+'shortconv.out_proj.weight',(8,8));weight(p+'shortconv.conv.weight',(8,3))
w.write_header_to_file();w.write_kv_data_to_file();w.write_tensors_to_file();w.close()
def norm(x,w): return x / np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+1e-5)*w
def silu(x): return x/(1+np.exp(-x))
def evaluate(tokens):
 x=weights['token_embd.weight'][tokens].astype(np.float64)
 s=len(tokens)
 for i in range(3):
  p=f'blk.{i}.'
  get=lambda n: weights[p+n].astype(np.float64)
  z=norm(x,get('attn_norm.weight'))
  if i!=1:
   b,c,v=np.split(z@get('shortconv.in_proj.weight').T,3,axis=-1)
   bx=b*v
   padded=np.pad(bx,((2,0),(0,0)))
   conv=sum(padded[t:t+s]*get('shortconv.conv.weight')[:,t] for t in range(3))
   y=(c*conv)@get('shortconv.out_proj.weight').T
  else:
   q=norm((z@get('attn_q.weight').T).reshape(s,2,4),get('attn_q_norm.weight'))
   k=norm((z@get('attn_k.weight').T).reshape(s,1,4),get('attn_k_norm.weight'))
   v=(z@get('attn_v.weight').T).reshape(s,1,4)
   freq=np.arange(s)[:,None]/(5000000.**(np.arange(0,4,2)/4))
   cs=np.concatenate([np.cos(freq)]*2,axis=-1)[:,None,:]
   sn=np.concatenate([np.sin(freq)]*2,axis=-1)[:,None,:]
   def rope(a): return a*cs+np.concatenate([-a[:,:,2:],a[:,:,:2]],axis=-1)*sn
   q=rope(q).transpose(1,0,2); k=rope(k).transpose(1,0,2);v=v.transpose(1,0,2)
   scores=q@k.transpose(0,2,1)/2
   scores=np.where(np.triu(np.ones((s,s)),1),-np.inf,scores)
   prob=np.exp(scores-np.max(scores,axis=-1,keepdims=True));prob/=prob.sum(axis=-1,keepdims=True)
   y=(prob@v).transpose(1,0,2).reshape(s,8)@get('attn_output.weight').T
  x=x+y
  z=norm(x,get('ffn_norm.weight'))
  if i==0:
   y=(silu(z@get('ffn_gate.weight').T)*(z@get('ffn_up.weight').T))@get('ffn_down.weight').T
  else:
   scores=1/(1+np.exp(-(z@get('ffn_gate_inp.weight').T)))
   ids=np.argsort(-(scores+get('exp_probs_b.bias')),axis=-1)[:,:2]
   selected=np.take_along_axis(scores,ids,axis=-1);selected/=selected.sum(axis=-1,keepdims=True)+1e-6
   y=np.zeros_like(z)
   for t in range(s):
    for slot,e in enumerate(ids[t]):
     y[t]+=selected[t,slot]*(silu(z[t]@get('ffn_gate_exps.weight')[e].T)*(z[t]@get('ffn_up_exps.weight')[e].T))@get('ffn_down_exps.weight')[e].T
  x=x+y
 return (norm(x,weights['token_embd_norm.weight'])@weights['token_embd.weight'].T).tolist()
cases=[[1,2,3,4,5,6,7],[1],[9,8,7,1]]
(root/'reference.json').write_text(json.dumps([{'tokens':t,'logits':evaluate(t)} for t in cases],indent=2)+'\n')
print('wrote independent GGUF/logit fixture')
