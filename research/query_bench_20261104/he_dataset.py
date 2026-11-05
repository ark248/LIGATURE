"""Construct paired targets on H&E-only geometry; supplied assay masks never read."""
from common import *
from he_pilot import segment
import io,time,shutil
import pandas as pd,pyarrow.parquet as pq,tifffile
from PIL import Image
from skimage.measure import regionprops
from skimage.segmentation import expand_labels

def main():
 torch.set_num_threads(2);he_root=RUN/'he_control';(he_root/'data/cells').mkdir(parents=True,exist_ok=True);(he_root/'shards').mkdir(exist_ok=True)
 shutil.copy2(Path(__file__).with_name('HE_CONTROL_PLAN.md'),he_root/'PLAN.md')
 table=pd.read_csv(DATA/'data/cells/crops.csv').sort_values(['case','x_start','y_start','source_index']);chosen=[]
 for case,g in table.groupby('case',sort=True):
  boxes=[]
  for _,r in g.iterrows():
   if any(r.x_start<x1 and r.x_end>x0 and r.y_start<y1 and r.y_end>y0 for x0,y0,x1,y1 in boxes):continue
   chosen.append(r);boxes.append((r.x_start,r.y_start,r.x_end,r.y_end))
   if len(boxes)==20:break
 selection=pd.DataFrame(chosen);selection.to_csv(he_root/'selected_crops.csv',index=False)
 model_path=RUN/'segmentation_pilot/model/instanseg.pt';assert sha(model_path)=='0c724da169d507d8dd2e206ad3675a0b1566824eee6e20ad9a0ba916311d9dd0';model=torch.jit.load(str(model_path));model.eval()
 t,se,dg=load_modules();channels=['Hoechst','AF1','CD31','CD45','CD68','Argo550','CD4','FOXP3','CD8a','CD45RO','CD20','PD-L1','CD3e','CD163','E-cadherin','PD-1','Ki67','Pan-CK','SMA'];pf={};summaries=[];start=time.monotonic()
 for k,(_,r) in enumerate(selection.iterrows()):
  target=he_root/'shards'/str(r.id);done=Path(str(target)+'.json')
  if done.exists():summaries.append(json.loads(done.read_text()));continue
  if r.file not in pf:pf[r.file]=pq.ParquetFile(r.file)
  row=pf[r.file].read_row_group(int(r.row_group),columns=['image','image_if','channel_names']).to_pylist()[0]
  assert row['channel_names']==channels;he=np.array(Image.open(io.BytesIO(row['image'])).convert('RGB'));ifc=tifffile.imread(io.BytesIO(row['image_if']))
  labels=segment(model,he);proxies=expand_labels(labels,distance=6);nucs={v.label:v for v in regionprops(labels)};props=regionprops(proxies);records=[];patches=[];s=48
  for v in props:
   nuc=nucs[v.label];cy,cx=v.centroid
   if nuc.area<20 or v.area<40 or nuc.perimeter<=0 or min(cx,cy,512-cx,512-cy)<48:continue
   yy,xx=v.coords.T;values=ifc[:,yy,xx].astype(np.float64).mean(1);ix,iy=int(round(cx)),int(round(cy));patches.append(he[iy-s:iy+s,ix-s:ix+s])
   records.append(dict(crop_id=str(r.id),case=str(r.case),sample=str(r['sample']),cell_label=int(v.label),cx=cx,cy=cy,gx=r.x_start+cx,gy=r.y_start+cy,cell_area_px=float(v.area),nuc_area_px=float(nuc.area),nuc_perimeter_px=float(nuc.perimeter),nuc_eccentricity=float(nuc.eccentricity),**{'y_'+m:float(values[channels.index(m)]) for m in se.MARKERS}))
  frame=pd.DataFrame(records);frame.to_parquet(str(target)+'.parquet',index=False);np.save(str(target)+'.patch.npy',np.stack(patches).astype(np.uint8) if patches else np.empty((0,96,96,3),np.uint8));np.save(str(target)+'.nuclei.npy',labels)
  summary=dict(crop=str(r.id),case=str(r.case),nuclei=int(len(nucs)),retained_cells=len(frame));jsonwrite(done,summary);summaries.append(summary)
  if k%25==0:print('H&E cells',k,'/',len(selection),'seconds',round(time.monotonic()-start,1),flush=True)
 frames=[pd.read_parquet(he_root/'shards'/(s['crop']+'.parquet')) for s in summaries if s['retained_cells']];frame=pd.concat(frames,ignore_index=True);frame.insert(0,'cell_idx',np.arange(len(frame)));frame.to_parquet(he_root/'data/cells/cells.parquet',index=False)
 per=frame.groupby('case').size();assert len(per)==40,'Missing patient; do not silently change the coordinate selection'
 jsonwrite(he_root/'dataset_complete.json',dict(crops=len(selection),nonempty_crops=frame.crop_id.nunique(),patients=len(per),cells=len(frame),cells_per_patient=per.to_dict(),seconds=time.monotonic()-start,plan_sha256=sha(he_root/'PLAN.md'),segmentation_model_sha256=sha(model_path),assay_masks_read=False));print('H&E dataset complete',len(frame),flush=True)
if __name__=='__main__':main()
