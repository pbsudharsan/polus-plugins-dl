import zarr

a='x003_y017_c001.ome.tif'
root = zarr.open('/home/sudharsan/Desktop/work/ome/location.zarr/', mode='r')

for m,l in root.groups():
    u=root[a]['probablity']
    print(type('x003_y017_c001.ome.tif'),type(a))
    print('x003_y017_c001.ome.tif' in root ,u.shape)
    a= l['probablity']
    print('dfsfsf',a.shape)