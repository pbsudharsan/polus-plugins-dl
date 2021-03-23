from pathlib import Path
from  bfio import  BioReader,BioWriter
import numpy as np

inpDir='/home/sudharsan/Desktop/work/test/training/input'
outDir='/home/sudharsan/Desktop/work/test/training/split'
#root = zarr.open('/home/sudharsan/Desktop/work/ome/location.zarr/', mode='r')



image_names = [f.name for f in Path(inpDir).iterdir() if f.is_file() and "".join(f.suffixes) == '.ome.tif'  ]

for f in image_names:
    br = BioReader(str(Path(inpDir).joinpath(f).absolute()))
    img = br.read().squeeze()
    for z in range(br.Z):
        out=img[:,:,z:z+1]
        bw = BioWriter(file_path=Path(outDir).joinpath(str(f)), backend='python', metadata=br.metadata)
  #      print(out[:,:,:,np.newaxis,np.newaxis].shape)
        bw.write(out[:,:,:,np.newaxis,np.newaxis])


