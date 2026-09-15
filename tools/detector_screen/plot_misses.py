"""Visual inspection of four measured incumbent errors near the scoring gate."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .evaluate import ROOT, SPACING


def main():
    rows={r['key']:r for r in json.loads((ROOT/'panel.json').read_text())['frames']}
    truth=json.loads((ROOT/'evaluation/ground_truth.json').read_text())['frames']
    cases=json.loads((ROOT/'evaluation/failure-analysis.json').read_text())['cases']
    wanted=['6bba_786893ac-t010','6bba_c328f2fd-t090','44b6_7e557709-t009','6bba_5c824876-t069']
    fig,axes=plt.subplots(4,2,figsize=(12,15),layout='constrained')
    for index,key in enumerate(wanted):
        case=next(c for c in cases if c['key']==key)
        row=rows[key];gt=np.array(next(g for g in truth[key]['gt_nodes'] if g[0]==case['gt_node_id'])[2:]);raw=np.load(row['image_path'])
        displayed=np.log1p(raw.astype(np.float32));lo,hi=np.percentile(displayed,[1,99.7])
        centers=np.clip(np.rint(np.load(ROOT/'predictions/incumbent'/(key+'.npz'))['centers_zyx']),0,np.array(raw.shape)-1)
        others=np.asarray(truth[key]['gt_nodes']).reshape(-1,5)[:,2:]
        nearest=centers[np.argmin(np.linalg.norm((centers-gt)*SPACING,axis=1))];half=np.ceil(15/SPACING).astype(int)
        lower=np.maximum(0,gt-half);upper=np.minimum(raw.shape,gt+half+1)
        for column,(vertical,slab_axis,slab_half,title) in enumerate([(1,0,1,'XY'),(0,1,3,'XZ')]):
            ax=axes[index,column]
            slices=[slice(int(a),int(b)) for a,b in zip(lower,upper)]
            slices[slab_axis]=slice(max(0,int(gt[slab_axis])-slab_half),min(raw.shape[slab_axis],int(gt[slab_axis])+slab_half+1))
            plane=displayed[tuple(slices)].max(axis=slab_axis)
            extent=[(lower[2]-.5-gt[2])*SPACING[2],(upper[2]-.5-gt[2])*SPACING[2],
                    (lower[vertical]-.5-gt[vertical])*SPACING[vertical],(upper[vertical]-.5-gt[vertical])*SPACING[vertical]]
            ax.imshow(plane,origin='lower',extent=extent,cmap='gray',vmin=lo,vmax=hi)
            offsets=(centers-gt)*SPACING
            visible=(np.abs(centers[:,slab_axis]-gt[slab_axis])<=slab_half)&(np.linalg.norm(offsets,axis=1)<22)
            ax.scatter(offsets[visible,2],offsets[visible,vertical],marker='o',s=38,facecolors='none',edgecolors='#66c2ff',linewidths=1.1,label='Incumbent candidates in slab')
            other_offsets=(others-gt)*SPACING
            visible_gt=(np.abs(others[:,slab_axis]-gt[slab_axis])<=slab_half)&(np.linalg.norm(other_offsets,axis=1)<22)
            ax.scatter(other_offsets[visible_gt,2],other_offsets[visible_gt,vertical],marker='+',s=60,color='#b2df8a',linewidths=1.2,label='Annotated centers in slab')
            near=(nearest-gt)*SPACING
            ax.scatter([near[2]],[near[vertical]],marker='D',s=70,facecolors='none',edgecolors='#ffae42',linewidths=1.7,label='Nearest incumbent (projected)')
            ax.scatter([0],[0],marker='+',s=170,color='#8fff8f',linewidths=2,label='Missed GT center')
            ax.add_patch(plt.Circle((0,0),7,fill=False,ls='--',lw=1,color='#8fff8f'))
            ax.set_xlim(-15,15);ax.set_ylim(-15,15);ax.set_aspect('equal')
            ax.set_xlabel('ΔX (µm)');ax.set_ylabel('Δ'+('Y' if vertical==1 else 'Z')+' (µm)')
            ax.set_title(f'{key} · {title}\n3D nearest distance {case["nearest_incumbent_distance_um"]:.3f} µm',fontsize=10)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=2,fontsize=9)
    fig.suptitle('Incumbent errors near the 7 µm gate\nThin-slab maximum projections; dashed circles are projection guides, not 3D match tests',fontsize=13)
    out=ROOT/'evaluation/figures';out.mkdir(exist_ok=True)
    fig.savefig(out/'incumbent-near-gate.png',dpi=140)
    plt.close(fig)
    print(out/'incumbent-near-gate.png')


if __name__=='__main__':main()
