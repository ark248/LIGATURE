"""Fetch the public, hash-verified segmentation artifact into the owned output root."""
from common import *
import urllib.request,zipfile

def main():
 root=RUN/'segmentation_pilot';root.mkdir(parents=True,exist_ok=True);archive=root/'brightfield_nuclei.zip';url='https://github.com/instanseg/instanseg/releases/download/instanseg_models_v0.1.1/brightfield_nuclei.zip'
 if not archive.exists():
  with urllib.request.urlopen(url,timeout=120) as r:archive.write_bytes(r.read())
 with zipfile.ZipFile(archive) as z:
  assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist())
  z.extractall(root/'model')
 expected='0c724da169d507d8dd2e206ad3675a0b1566824eee6e20ad9a0ba916311d9dd0';assert sha(root/'model/instanseg.pt')==expected
 jsonwrite(root/'download_manifest.json',dict(url=url,model_sha256=expected,archive_sha256=sha(archive),license='Apache-2.0; see bundled model README for training-dataset licenses'))
 print('Verified official InstanSeg brightfield model',expected)
if __name__=='__main__':main()
