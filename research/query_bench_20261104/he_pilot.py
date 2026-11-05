"""Timing/availability pilot only, without assay segmentations or model fitting."""
from common import *
import io,time,hashlib
import pandas as pd,pyarrow.parquet as pq
from PIL import Image

def segment(model,he):
 inp=torch.from_numpy(np.array(he)).permute(2,0,1)[None].float();inp=torch.nn.functional.interpolate(inp,scale_factor=.325/float(model.pixel_size),mode='bilinear')
 a=inp[0].numpy()
 for j in range(3):
  lo,hi=np.percentile(a[j],[.1,99.9]);a[j]=(a[j]-lo)/max(.001,hi-lo)
 with torch.no_grad():out=model(inp)
 return torch.nn.functional.interpolate(out.float(),size=he.shape[:2],mode='nearest')[0,0].numpy().astype(np.int32)

def main():
 torch.set_num_threads(2);p=RUN/'segmentation_pilot';model_path=p/'model/instanseg.pt'
 assert sha(model_path)=='0c724da169d507d8dd2e206ad3675a0b1566824eee6e20ad9a0ba916311d9dd0'
 model=torch.jit.load(str(model_path));model.eval()
 meta=pd.read_csv(DATA/'data/cells/crops.csv').sort_values(['case','x_start','y_start','source_index']).iloc[0]
 # Read H&E bytes only: assay segmentations are never read.
 row=pq.ParquetFile(meta.file).read_row_group(int(meta.row_group),columns=['image']).to_pylist()[0];he=np.array(Image.open(io.BytesIO(row['image'])).convert('RGB'))
 start=time.monotonic();labels=segment(model,he);elapsed=time.monotonic()-start
 np.save(p/'pilot_nuclei.npy',labels);Image.fromarray(he).save(p/'pilot_he.png')
 from skimage.segmentation import mark_boundaries
 Image.fromarray((255*mark_boundaries(he,labels,color=(0,.75,.5))).astype(np.uint8)).save(p/'pilot_overlay.png')
 jsonwrite(p/'pilot.json',dict(crop=str(meta.id),selection='first crop by case, x_start, y_start, source_index',seconds=elapsed,nuclei=int(len(np.unique(labels))-1),cpu_threads=2,model_sha256=sha(model_path),model_version='brightfield_nuclei v0.1.1 / instanseg-torch 0.1.2',model_url='https://github.com/instanseg/instanseg/releases/download/instanseg_models_v0.1.1/brightfield_nuclei.zip',assay_segmentations_read=False,pixel_size=.325,model_pixel_size=float(model.pixel_size),preprocessing='official small-image bilinear rescale and per-channel 0.1/99.9 percentile normalization, epsilon 0.001; nearest-neighbor output rescale'))
 print((p/'pilot.json').read_text())
if __name__=='__main__':main()
