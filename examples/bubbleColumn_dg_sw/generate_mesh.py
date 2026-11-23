#!/usr/bin/env python3
"""Generate a 0.1 x 0.1 x 1 m straight all-hex column, centred in x/y.
Boundary IDs: bottom=1, top=2, wallx=3, wally=4. Python stdlib only.
Use an existing bubbleColumn mesh with these IDs for controlled comparisons.
"""
import argparse
from pathlib import Path
import struct

DOMAIN = (0.10, 0.10, 1.0)
BASE_COUNTS = (8, 8, 40)
# Standard Nek .re2 vertex order (cyclic, NOT tensor order).
VERTICES = ((0,0,0),(1,0,0),(1,1,0),(0,1,0),
            (0,0,1),(1,0,1),(1,1,1),(0,1,1))
FACES = ((0,1,5,4),(1,2,6,5),(2,3,7,6),
         (3,0,4,7),(0,3,2,1),(4,5,6,7))


def generate(path, refinement=1):
    if not isinstance(refinement, int) or refinement < 1:
        raise ValueError("refinement must be a positive integer")
    nx, ny, nz = (v*refinement for v in BASE_COUNTS)
    spacing = tuple(d/n for d,n in zip(DOMAIN,(nx,ny,nz)))
    count = nx*ny*nz
    header = f"#v003{count:9d}{3:3d}{count:9d} Method1 DG alpha SI units"
    boundary = []
    with Path(path).open("wb") as out:
        out.write(header.ljust(80).encode("ascii"))
        out.write(struct.pack("<f", 6.54321))
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    eid = 1+i+nx*(j+ny*k)
                    xyz = [tuple((v[c]+(i,j,k)[c])*spacing[c]+(-.05,-.05,0.)[c] for c in range(3))
                           for v in VERTICES]
                    out.write(struct.pack("<q24d", 0,
                        *(xyz[v][c] for c in range(3) for v in range(8))))
                    for face,at_boundary,tag in (
                        (1,j==0,4),(2,i==nx-1,3),(3,j==ny-1,4),
                        (4,i==0,3),(5,k==0,1),(6,k==nz-1,2)):
                        if at_boundary:
                            boundary.append((eid,face,tag))
        out.write(struct.pack("<d",0))  # no curved sides
        out.write(struct.pack("<d",len(boundary)))
        for eid,face,tag in boundary:
            code = b"O       " if tag==2 else b"v       "
            out.write(struct.pack("<qq5d8s",eid,face,0,0,0,0,tag,code))
    return count, len(boundary), spacing


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refine",type=int,default=1,
                        help="integer refinement factor in all three directions")
    parser.add_argument("--output",type=Path,
                        default=Path(__file__).with_name("bubbleColumn_dg_sw.re2"))
    args=parser.parse_args()
    elements,boundary,spacing=generate(args.output,args.refine)
    print(f"{args.output}: {elements} hexes, {boundary} boundary faces, spacing={spacing} m")


if __name__=="__main__":
    main()
