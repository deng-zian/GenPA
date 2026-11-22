"""Export a portable RGB/current-depth subset from aligned replay demonstrations.

The export preserves RGB videos, action/state arrays, and metric-depth samples
without recomputing or quantizing them. Other perception targets are not copied.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import h5py

CAMERAS=("agentview","robot0_eye_in_hand")
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args(); src=args.source; out=args.output
    if out.exists() and any(out.iterdir()): raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    for folder in ('data','videos','meta'):
        for f in sorted((src/folder).rglob('*')):
            if f.is_file() and f.suffix in ('.parquet','.mp4','.json','.jsonl') and 'episode_stats' not in f.parts:
                dest=out/f.relative_to(src); dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(f,dest)
    rows=[json.loads(x) for x in (src/'meta/episodes.jsonl').read_text().splitlines()]
    n=int(json.loads((src/'meta/info.json').read_text()).get('chunks_size',1000))
    def export(row):
        i=int(row['episode_index']); rel=Path(f'chunk-{i//n:03d}/episode_{i:06d}.hdf5')
        dest=out/'depth'/rel; dest.parent.mkdir(parents=True,exist_ok=True)
        with h5py.File(src/'visual_gt'/rel) as h,h5py.File(dest,'w') as d:
            for cam in CAMERAS:
                group=d.create_group(cam); h.copy(h[cam]['depth_m'],group,name='depth_m',without_attrs=True)
                if group['depth_m'].shape[0]!=int(row['length']): raise ValueError(f'Unaligned episode {i}')
        return i
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for count,_ in enumerate(pool.map(export,rows),1):
            if count%100==0: print(f'depth episodes {count}/{len(rows)}',flush=True)
    # Retain demonstration provenance without machine-specific paths.
    if (src/'manifests/source_map.jsonl').exists():
        (out/'manifests').mkdir(exist_ok=True)
        records=[]
        for line in (src/'manifests/source_map.jsonl').read_text().splitlines():
            r=json.loads(line);s=r['source']
            records.append({'episode_index':r['view_episode_index'],'suite':s['suite'],'task':s['task'],
                'source_hdf5':s['source_hdf5_relative'],'source_demo':s['source_demo'],
                'frames':r['view_length']})
        (out/'manifests/demonstrations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    # Dataset metadata contains public tasks/stats, not local paths.
    for f in (out/'meta').rglob('*.json*'):
        text=f.read_text()
        if '/root/' in text or '/home/' in text: raise ValueError(f'Absolute path in {f}')
    manifest={'format':'genpa_libero_rgbd_v1','episodes':len(rows),'frames':sum(int(r['length']) for r in rows),
        'depth_units':'meters','depth_encoding':'lossless source values; log-inverse mapping in loader',
        'cameras':list(CAMERAS),'state':'complete'}
    (out/'dataset.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest),flush=True)
if __name__=='__main__': main()
