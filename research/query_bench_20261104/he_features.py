"""Pinned Phikon features at H&E-derived centroids, restartable by crop."""
from common import *
import time
import pandas as pd

def main():
 torch.set_num_threads(8);root=RUN/'he_control';assert (root/'dataset_complete.json').exists();device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 from transformers import AutoModel
 revision='2ae989a9c40cffaa27f0a6cb29cc94d1d6f9a5fd';snapshot=DATA/'data/hf_cache/hub/models--owkin--phikon-v2/snapshots'/revision
 net=AutoModel.from_pretrained(str(snapshot),local_files_only=True).to(device).eval()
 if device.type=='cuda':net=net.half()
 mean=torch.tensor([.485,.456,.406],device=device).view(1,3,1,1);sd=torch.tensor([.229,.224,.225],device=device).view(1,3,1,1)
 frame=pd.read_parquet(root/'data/cells/cells.parquet');dest=root/'feature_shards';dest.mkdir(exist_ok=True);start=time.monotonic()
 for k,(crop,g) in enumerate(frame.groupby('crop_id',sort=False)):
  path=dest/(crop+'.npy')
  if path.exists():assert np.load(path,mmap_mode='r').shape==(len(g),int(net.config.hidden_size));continue
  patches=np.load(root/'shards'/(crop+'.patch.npy'),mmap_mode='r');outs=[]
  for i in range(0,len(patches),64):
   # Exactly the primary 64px field of view from 96px stored patches.
   x=torch.from_numpy(np.array(patches[i:i+64,16:80,16:80])).to(device).permute(0,3,1,2).float()/255
   x=torch.nn.functional.interpolate(x,size=(224,224),mode='bicubic',align_corners=False).clamp(0,1);x=(x-mean)/sd
   with torch.no_grad():f=net(pixel_values=x.to(dtype=next(net.parameters()).dtype)).last_hidden_state[:,0].float()
   if not torch.isfinite(f).all():raise RuntimeError('Nonfinite pinned encoder output')
   outs.append(f.cpu().numpy().astype(np.float16))
  np.save(path,np.concatenate(outs))
  if k%25==0:print('H&E features',k,'seconds',round(time.monotonic()-start,1),flush=True)
 out=root/'data/features';out.mkdir(exist_ok=True);X=np.zeros((len(frame),int(net.config.hidden_size)),np.float16)
 for crop,g in frame.groupby('crop_id',sort=False):X[g.index]=np.load(dest/(crop+'.npy'))
 np.save(out/'X_p64.npy',X);np.save(out/'cell_idx.npy',frame.cell_idx.to_numpy())
 jsonwrite(root/'features_complete.json',dict(cells=len(frame),shape=X.shape,seconds=time.monotonic()-start,encoder='owkin/phikon-v2',revision=revision,input_field_pixels=64,cell_centers='H&E-derived proxy centroids',assay_segmentation_features_reused=False,device=str(device)));print('H&E features complete',flush=True)
if __name__=='__main__':main()
