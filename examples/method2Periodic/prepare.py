#!/usr/bin/env python3
"""Generate deterministic periodic .re2 meshes and stage current method2 sources.

No third-party Python packages or Nek mesh tools are required. Run from anywhere.
"""
from pathlib import Path
import math, struct, shutil, argparse
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'bubbleColumn2'
FACES=((0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7),(0,3,2,1),(4,5,6,7))

def write_re2(path,elements,boundaries,curves=()):
    with path.open('wb') as f:
        f.write(("#v004%16d%3d%16d%4d hdr"%(len(elements),3,len(elements),1)).ljust(80).encode())
        f.write(struct.pack('<f',6.54321))
        for e in elements:
            f.write(struct.pack('<25d',0,*[v[a] for a in range(3) for v in e]))
        f.write(struct.pack('<d',len(curves)))
        for e,edge,radius in curves:
            f.write(struct.pack('<7d',e,edge,radius,0,0,0,0)+b'C       ')
        f.write(struct.pack('<d',len(boundaries)))
        for e,face,code,bid,pe,pf in boundaries:
            f.write(struct.pack('<7d',e,face,pe,pf,0,0,bid)+code.ljust(8).encode())

def hex_from_quad(q,z0,z1):
    return [(x,y,z) for z in (z0,z1) for x,y in q]

def box(nx=4,length=.1):
    elements=[];b=[]
    def eid(i,j,k):return 1+i+nx*(j+nx*k)
    for k in range(nx):
        for j in range(nx):
            for i in range(nx):
                x0=i*length/nx;x1=(i+1)*length/nx
                y0=j*length/nx;y1=(j+1)*length/nx
                elements.append(hex_from_quad([(x0,y0),(x1,y0),(x1,y1),(x0,y1)],k*length/nx,(k+1)*length/nx))
                e=eid(i,j,k)
                for cond,face,partner,pf in ((j==0,1,eid(i,nx-1,k),3),
                    (i==nx-1,2,eid(0,j,k),4),(j==nx-1,3,eid(i,0,k),1),
                    (i==0,4,eid(nx-1,j,k),2),(k==0,5,eid(i,j,nx-1),6),
                    (k==nx-1,6,eid(i,j,0),5)):
                    if cond:b.append((e,face,'P',0,partner,pf))
    return elements,b,[]

def tube(n=4,nr=3,nz=8,radius=.075,length=.3):
    a=.4*radius;quads=[];wall=[]
    for j in range(n):
        for i in range(n):
            x0=-a+2*a*i/n;x1=-a+2*a*(i+1)/n
            y0=-a+2*a*j/n;y1=-a+2*a*(j+1)/n
            quads.append([(x0,y0),(x1,y0),(x1,y1),(x0,y1)])
    def rotate(x,y,side):
        for _ in range(side):x,y=-y,x
        return x,y
    def point(t,j,side):
        x,y=a,-a+2*a*j/n;norm=math.hypot(x,y)
        x,y=((1-t)*x+t*radius*x/norm,(1-t)*y+t*radius*y/norm)
        return rotate(x,y,side)
    for side in range(4):
        for r in range(nr):
            for j in range(n):
                quads.append([point(r/nr,j,side),point((r+1)/nr,j,side),
                              point((r+1)/nr,j+1,side),point(r/nr,j+1,side)])
                if r==nr-1:wall.append(len(quads)-1)
    elements=[];b=[];curves=[];nq=len(quads)
    for k in range(nz):
        for qi,q in enumerate(quads):
            e=1+qi+k*nq;elements.append(hex_from_quad(q,k*length/nz,(k+1)*length/nz))
            if k==0:b.append((e,5,'P',0,1+qi+(nz-1)*nq,6))
            if k==nz-1:b.append((e,6,'P',0,1+qi,5))
            if qi in wall:
                b.append((e,2,'W',1,0,0))
                # Native circular arcs on both horizontal edges of each wall face.
                curves.extend(((e,2,radius),(e,6,radius)))
    return elements,b,curves

def stage(case):
    # Copies make each run directory portable and avoid symlink handling on HPC.
    # These are generated files, not independent forks of method2.
    for name in ('bubbleColumn2Equations.okl','bubbleColumn2Transport.okl',
                 'explicitFilter.okl'):
        shutil.copy2(BASE/name,case/name)
    shutil.copy2(ROOT/'method2Periodic.okl',case/'method2Periodic.okl')
    shutil.copy2(ROOT/'periodicBoundary.oudf',case/'periodicBoundary.oudf')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=['1','2','3','all'],default='all')
    args=parser.parse_args()
    for c in (1,2,3):
        if args.case not in ('all',str(c)):continue
        case=ROOT/f'case{c}';stage(case)
        elements,b,curves=tube() if c==3 else box()
        write_re2(case/f'case{c}.re2',elements,b,curves)
        print(f'case{c}: {len(elements)} hex, {sum(x[2]=="P" for x in b)} periodic faces, '
              f'{sum(x[2]=="W" for x in b)} wall faces, {len(curves)} circular edges')
if __name__=='__main__':main()
