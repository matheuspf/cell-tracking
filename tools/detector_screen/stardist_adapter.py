"""Native 3D Xenopus checkpoint, preserving geometry and StarDist center scores."""
import argparse,fcntl,gc,hashlib,json,os,time
from pathlib import Path
os.environ.setdefault('TF_USE_LEGACY_KERAS','1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
import numpy as np
from scipy.ndimage import zoom
import tensorflow as tf
from csbdeep.utils import normalize
from stardist.models import StarDist3D
from stardist.geometry import polyhedron_to_label
from skimage.measure import regionprops_table

ROOT=Path(__file__).resolve().parents[2]/'work/detector-screen-20260914'
LOCK=Path('/kaggle/working/cell-tracking/detector-screen-20260914.gpu.lock')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--role',choices=['pilot','assessment','all'],default='all');ap.add_argument('--limit',type=int);ap.add_argument('--default-only',action='store_true');ap.add_argument('--n-tiles',nargs=3,type=int,default=[4,2,2]);args=ap.parse_args()
    tf.config.threading.set_intra_op_parallelism_threads(4);tf.config.threading.set_inter_op_parallelism_threads(2)
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.set_logical_device_configuration(gpu,[tf.config.LogicalDeviceConfiguration(memory_limit=4000)])
    folder=ROOT/'assets/xenopus';model=StarDist3D(None,name=folder.name,basedir=str(folder.parent));threshold=float(model.thresholds.prob)
    cfg=dict(checkpoint_sha256=hashlib.sha256((folder/'weights_best.h5').read_bytes()).hexdigest(),
        desired_zoom_zyx=[4/float(model.config.anisotropy[0]),1,1],resampling='scipy zoom order1 grid_mode=True mode=nearest; voxel-center inverse',
        normalization='1/99.8 percentiles',n_tiles=args.n_tiles,probability_threshold=threshold,loose_threshold=.1,nms=.3,
        center_definition='StarDist predicted polyhedron origin / centerness point; no instance rasterization',
        scope='Match documented training anisotropy while preserving XY scale; absolute training spacing absent. Frozen without GEFF access.')
    (ROOT/'xenopus').mkdir(exist_ok=True);(ROOT/'xenopus/config.json').write_text(json.dumps(cfg,indent=2))
    rows=[r for r in json.loads((ROOT/'panel.json').read_text())['frames'] if args.role=='all' or r['role']==args.role]
    if args.limit:rows=rows[:args.limit]
    for row in rows:
        methods=['xenopus'] if args.default_only else ['xenopus','xenopus_loose']
        cuts=[threshold] if args.default_only else [threshold,.1]
        paths=[ROOT/'predictions'/m/(row['key']+'.npz') for m in methods]
        centroid_path=ROOT/'predictions/xenopus_centroid'/(row['key']+'.npz')
        if all(p.exists() for p in paths) and centroid_path.exists():continue
        start=time.monotonic();raw=np.load(row['image_path']).astype(np.float32)
        image=zoom(raw,cfg['desired_zoom_zyx'],order=1,grid_mode=True,mode='nearest')
        actual_zoom=np.array(image.shape)/np.array(raw.shape);image=normalize(image,1,99.8,axis=(0,1,2));prep=time.monotonic()-start
        with LOCK.open('a') as lock:
            waited=time.monotonic();fcntl.flock(lock,fcntl.LOCK_EX);waited=time.monotonic()-waited;start=time.monotonic()
            # The upstream generator exposes the exact neural/NMS boundary.
            # Release the GPU while its unchanged CPU polyhedron NMS runs.
            generator=model._predict_instances_generator(image,axes='ZYX',n_tiles=tuple(args.n_tiles),prob_thresh=min(cuts),nms_thresh=.3,return_labels=False,show_tile_progress=False)
            for stage in generator:
                if isinstance(stage,str) and stage=='nms':break
            neural_seconds=time.monotonic()-start;fcntl.flock(lock,fcntl.LOCK_UN)
        nms_start=time.monotonic()
        for result in generator:pass
        _,details=result
        nms_seconds=time.monotonic()-nms_start;elapsed=neural_seconds+nms_seconds
        centers=(np.asarray(details['points'],dtype=np.float64).reshape(-1,3)+.5)/actual_zoom-.5
        scores=np.asarray(details['prob'],dtype=np.float64)
        render_start=time.monotonic()
        default_keep=scores>=threshold
        labels=polyhedron_to_label(np.asarray(details['dist'])[default_keep],
            np.asarray(details['points'])[default_keep],details['rays'],shape=image.shape,
            prob=scores[default_keep],verbose=False)
        props=regionprops_table(labels,properties=('label','centroid'))
        geometric=np.column_stack([props['centroid-'+str(axis)] for axis in range(3)])
        geometric=(geometric+.5)/actual_zoom-.5
        geometric_scores=scores[default_keep][np.asarray(props['label'],dtype=int)-1]
        render_seconds=time.monotonic()-render_start
        centroid_path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(centroid_path,centers_zyx=geometric,scores=geometric_scores)
        centroid_path.with_suffix('.json').write_text(json.dumps(dict(key=row['key'],
            seconds=elapsed+prep+render_seconds,neural_seconds=neural_seconds,nms_seconds=nms_seconds,
            mask_centroid_seconds=render_seconds,preprocess_seconds=prep,gpu_wait_seconds=waited,
            threshold=threshold,candidates=len(geometric),input_shape=list(image.shape),
            actual_zoom_zyx=actual_zoom.tolist(),n_tiles=args.n_tiles,
            center_definition='Geometric centroid of source rasterized polyhedron; original label ID maps to source confidence. Voxel-center inverse to native ZYX.'),indent=2))
        for dest,cut in zip(paths,cuts):
            keep=scores>=cut;dest.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest,centers_zyx=centers[keep],scores=scores[keep])
            dest.with_suffix('.json').write_text(json.dumps(dict(key=row['key'],seconds=elapsed+prep,gpu_wait_seconds=waited,
                inference_seconds=elapsed,neural_seconds=neural_seconds,nms_seconds=nms_seconds,preprocess_seconds=prep,candidates=int(keep.sum()),threshold=cut,
                input_shape=list(image.shape),actual_zoom_zyx=actual_zoom.tolist(),n_tiles=args.n_tiles,
                execution='Upstream neural/NMS pass with proposal threshold '+str(min(cuts))+'; no temporal assignment.'),indent=2))
        del details,labels;gc.collect()
        print(row['key'],len(centers),int((scores>=threshold).sum()),round(elapsed+prep,3),flush=True)

if __name__=='__main__':main()
