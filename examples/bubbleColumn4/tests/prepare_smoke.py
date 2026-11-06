#!/usr/bin/env python3
"""Prepare short native cases outside the source case; never overwrite input."""
from pathlib import Path
import argparse,re,shutil
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('output',type=Path)
p.add_argument('--steps',type=int,default=20)
p.add_argument('--mesh',type=Path,help='Existing column mesh; IDs 1=inlet, 2=outlet, 3=wall')
a=p.parse_args();root=Path(__file__).resolve().parents[1]
mesh=a.mesh or root.parent/'bubbleColumn2/bubbleColumn2.re2'
if not mesh.is_file():p.error('Column mesh is not in the repository. Supply --mesh /path/to/column.re2')
profiles={
 'plume':{},
 'plume_no_drag':{'dragEnabled':'0'},
 'uniform_mixture':{'initialPlumeHeight':'1e6'},
 'hydrostatic_liquid':{'alphaInlet':'0','gasInletVelocity':'0','liquidInletVelocity':'0','dragEnabled':'0'},
}
for name,changes in profiles.items():
    dest=a.output/name
    dest.mkdir(parents=True,exist_ok=False)
    for f in root.iterdir():
        if f.is_file() and f.suffix in {'.udf','.oudf','.usr','.okl','.hpp'}:shutil.copy2(f,dest/f.name)
    text=(root/'bubbleColumn4.par').read_text()
    changes.update({'numSteps':str(a.steps),'monitorInterval':'1','checkpointInterval':str(a.steps),'file':str(mesh.resolve())})
    for key,value in changes.items():text=re.sub(r'(?mi)^'+re.escape(key)+r'\s*=.*$',key+' = '+value,text)
    (dest/'bubbleColumn4.par').write_text(text)
    print(dest.resolve())
