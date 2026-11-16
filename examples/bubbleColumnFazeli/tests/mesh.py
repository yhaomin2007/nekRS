"""Independently decode RE2 and check geometry, connectivity and boundary IDs."""
from collections import Counter
import importlib.util
import math
from pathlib import Path
import struct
import tempfile

case=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("generate_mesh",case/"generate_mesh.py")
generator=importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


def check(path):
    with path.open("rb") as f:
        header=f.read(80).decode()
        assert header[:5]=="#v003"
        count=int(header[5:14]); assert int(header[14:17])==3
        assert int(header[17:26])==count
        assert math.isclose(struct.unpack("<f",f.read(4))[0],6.54321,rel_tol=1e-7)
        faces=Counter(); exposed={}; volumes=0.; source=0.
        for e in range(1,count+1):
            values=struct.unpack("<q24d",f.read(200)); assert values[0]==0
            xyz=[tuple(values[1+c*8+v] for c in range(3)) for v in range(8)]
            # Positive affine Jacobian with Nek's vertex convention.
            dx=xyz[1][0]-xyz[0][0];dy=xyz[3][1]-xyz[0][1];dz=xyz[4][2]-xyz[0][2]
            assert dx>0 and dy>0 and dz>0
            for v,(ix,iy,iz) in enumerate(generator.VERTICES):
                expected=(xyz[0][0]+ix*dx,xyz[0][1]+iy*dy,xyz[0][2]+iz*dz)
                assert all(math.isclose(a,b,abs_tol=1e-14) for a,b in zip(xyz[v],expected))
            vol=dx*dy*dz;volumes+=vol
            xc,yc,zc=(sum(p[c] for p in xyz)/8 for c in range(3))
            if .08<xc<.12 and .01<yc<.04 and .01<zc<.04: source+=vol
            for face,vs in enumerate(generator.FACES,1):
                points=tuple(sorted(tuple(round(c,12) for c in xyz[v]) for v in vs))
                faces[points]+=1;exposed[(e,face)]=points
        assert struct.unpack("<d",f.read(8))[0]==0
        nb=int(struct.unpack("<d",f.read(8))[0]); seen=set();area=Counter()
        for _ in range(nb):
            e,face,*rest=struct.unpack("<qq5d8s",f.read(64))
            *params,code=rest; tag=int(params[4]); key=(e,face)
            assert key not in seen;seen.add(key)
            points=exposed[key]; assert faces[points]==1
            if tag==1: assert face==5 and all(p[2]==0 for p in points)
            elif tag==2: assert face==6 and all(p[2]==.45 for p in points)
            else: assert tag==3 and face in (1,2,3,4)
            assert code[:3]==(b"O  " if tag==2 else b"v  ")
            extents=[max(p[c] for p in points)-min(p[c] for p in points) for c in range(3)]
            area[tag]+=math.prod(d for d in extents if d>0)
        assert not f.read()
        assert sum(n==1 for n in faces.values())==nb
        assert all(n in (1,2) for n in faces.values())
        assert math.isclose(volumes,.2*.05*.45,rel_tol=1e-10)
        assert math.isclose(source,3.6e-5,rel_tol=1e-10)
        for tag,expected in ((1,.01),(2,.01),(3,2*(.20+.05)*.45)):
            assert math.isclose(area[tag],expected,rel_tol=1e-10)
        return count


with tempfile.TemporaryDirectory() as t:
    path=Path(t)/"mesh.re2"
    generator.generate(path,1)
    assert check(path)==9000
    generator.generate(path,2)
    assert check(path)==72000
if (case/"bubbleColumnFazeli.re2").exists(): check(case/"bubbleColumnFazeli.re2")
print("RE2 decoding, positive Jacobians, paired interior faces, boundary tags/areas and source volume passed (9k/72k hexes).")
