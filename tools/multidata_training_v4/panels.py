"""Local-only source alignment panels; microscopy/coordinates are not Git artifacts."""
from .common import *

def run():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import zarr
    fig,axes=plt.subplots(2,2,figsize=(12,10))
    for ax,sid,kind in [(axes[0,0],'vol_00000','static'),(axes[0,1],'seq_0000','sequence')]:
        raw=arrays(SYNTH/('static' if kind=='static' else 'sequences')/f'{sid}.npz');lab=arrays(PREPARED/'synthetic'/f'{sid}_labels.npz')
        from .adapters import synthetic_record
        image,pts,spacing=synthetic_record(raw,lab,kind)
        if kind=='sequence':pts=pts[lab['t']==0]
        z=int(np.median(pts[:,0]));sl=image[0,z];p=pts[np.abs(pts[:,0]-z)<1.5]
        ax.imshow(sl,cmap='gray',vmin=np.quantile(sl,.02),vmax=np.quantile(sl,.995));ax.scatter(p[:,2],p[:,1],s=40,facecolors='none',edgecolors='tomato')
        ax.set(title=f'{sid}: actual {kind} grid, t=0 z={z}',xlabel='image X (voxels)',ylabel='image Y (voxels)')
    lab=arrays(PREPARED/'zoo/ascidian_graph.npz');p=int(lab['division_parent_ids'][0]);children=lab['edges'][lab['edges'][:,0]==p,1]
    ax=axes[1,0];pts=lab['zyx_source'];t=lab['t'][p];ids=np.flatnonzero((lab['t']>=t)&(lab['t']<=t+1))
    center=pts[p];span=np.maximum(np.quantile(pts,.75,axis=0)-np.quantile(pts,.25,axis=0),1e-5);q=(pts-center)/span
    ax.scatter(q[ids,2],q[ids,1],s=9,c=lab['t'][ids]-t,cmap='Greys')
    for child in children:ax.plot(q[[p,child],2],q[[p,child],1],color='tomato')
    ax.set(title='Ascidian: checked direct-parent geometry (weak labels)',xlabel='X / per-axis IQR (unitless)',ylabel='Y / per-axis IQR (unitless)')
    row=next(r for r in inputs() if r['embryo']=='44b6');gt=arrays(V1/'evaluation/gt'/f'{row["dataset"]}.npz')['nodes'];n=gt[len(gt)//2];t,z=map(int,n[1:3]);im=zarr.open_group(row['image_path'],mode='r')['0'][t,z]
    ax=axes[1,1];ax.imshow(im,cmap='gray',vmin=np.quantile(im,.02),vmax=np.quantile(im,.995));p=gt[(gt[:,1]==t)&(np.abs(gt[:,2]-z)<1.5)]
    ax.scatter(p[:,4],p[:,3],s=40,facecolors='none',edgecolors='tomato');ax.set(title=f'44b6 source: sparse centers, t={t} z={z}',xlabel='native X (voxels)',ylabel='native Y (voxels)')
    fig.suptitle('W410 source sanity panels · numeric coordinates, original arrays',fontsize=15);fig.tight_layout()
    dest=OUT/'local_source_panels.png';fig.savefig(dest,dpi=150);plt.close(fig)
    write(OUT/'source_panel_receipt.json',dict(path=str(dest),sha256=sha(dest),local_only=True,synthetic_native_and_pooled=True,
        zoo_no_images=True,biohub_source='44b6',coordinates_from_metadata_not_ocr=True))
    print(dest)
