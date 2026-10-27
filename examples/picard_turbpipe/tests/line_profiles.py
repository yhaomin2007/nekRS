"""Exercise production line exporter with real OCCA memory and interpolation/MPI doubles."""
import ast
import csv
import configparser
import os
from pathlib import Path
import subprocess
import tempfile

case = Path(__file__).resolve().parents[1]
repo = case.parents[1]
occa = Path(os.environ.get('OCCA_DIR', repo.parent/'occa-cub-parser'))
# Numerical settings must match the native case; only the mesh filename differs.
configs=[]
for name in ('turbPipe/turbPipe.par','picard_turbpipe/picard_turbpipe.par'):
    config=configparser.ConfigParser()
    config.read_string('[DEFAULT]\n'+(repo/'examples'/name).read_text())
    configs.append(config)
for section in ('GENERAL','FLUID PRESSURE','FLUID VELOCITY','SCALAR S00'):
    assert dict(configs[0][section])==dict(configs[1][section]),section
for key in ('lineprofilesenabled','lineprofileinterval','lineprofilepoints'):
    assert configs[0]['CASEDATA'][key]==configs[1]['CASEDATA'][key]
assert 'if (tstep == 0 || nrs->timeStepConverged) pipeLines::write(time, tstep);' in (repo/'examples/turbPipe/turbPipe.udf').read_text()
# Reuse the Serial memory/types harness, without executing its tests.
module = ast.parse((case/'tests/frozen_inner.py').read_text())
prefix = next(ast.literal_eval(node.value) for node in module.body
              if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'prefix' for t in node.targets))
prefix = prefix.replace('struct Platform {', 'struct Par {template<class T>void extract(const char*,const char*,T&) {}} par;\nstruct Platform {Par *par=&::par;')
prefix = prefix.replace('struct Mesh {', 'struct Mesh {occa::memory o_x,o_y,o_z;')
prefix = prefix.replace('struct Scalar {', 'struct Scalar {occa::memory o_solution(std::string){return o_S;}')
prefix = prefix.replace('int outerCorrector=0,', 'bool lastStep=false;int outerCorrector=0,')
prefix += r'''
constexpr int MPI_INT=0,MPI_SUM=0;
void MPI_Allreduce(const int *in,int *out,int,int,int,int){*out=*in;}
'''
interpolation = r'''
#pragma once
static bool mockMissing=false;
struct pointInterpolation_t {
 static constexpr int CODE_NOT_FOUND=2;
 enum class VerbosityLevel {Detailed};
 struct Data {std::vector<int> code;} d;
 int np=0;
 pointInterpolation_t(Mesh*,int){}
 void setPoints(const std::vector<double>&x,const std::vector<double>&y,const std::vector<double>&z){assert(x.size()==y.size()&&y.size()==z.size());np=x.size();d.code.resize(np);}
 void find(VerbosityLevel){if(mockMissing&&!d.code.empty())d.code[0]=CODE_NOT_FOUND;}
 Data &data(){return d;}
 void eval(long fields,long stride,const occa::memory &in,long outStride,occa::memory &out){
  assert(fields==5&&stride==5&&outStride==np);
  for(int f=0;f<fields;++f)for(int i=0;i<np;++i)ptr(out)[f*np+i]=ptr(in)[f*stride+i%3];
 }
};
'''
main = r'''
#include "lineProfiles.hpp"
int main(int argc,char**argv){
 occa::properties props;props["mode"]="Serial";platform->device.setup(props);
 Mesh mesh;mesh.Nlocal=3;double x[3]={-.5,0,.5},y[3]={-.5,0,.5},z[3]={0,10,20};
 mesh.o_x=platform->device.malloc<double>(3,x);mesh.o_y=platform->device.malloc<double>(3,y);mesh.o_z=platform->device.malloc<double>(3,z);
 Fluid f;Scalar s;f.o_U=alloc(15);f.o_P=alloc(5);s.o_S=alloc(7);
 for(int c=0;c<3;++c)for(int i=0;i<3;++i)ptr(f.o_U)[5*c+i]=10*c+i;
 for(int i=0;i<3;++i){ptr(f.o_P)[i]=30+i;ptr(s.o_S)[i]=40+i;}
 nrs->meshV=&mesh;nrs->fluid=&f;nrs->scalar=&s;
 pipeLines::configure();
 if(argc>1 && std::string(argv[1])=="disabled"){pipeLines::enabled=0;pipeLines::write(0,0);return 0;}
 if(argc>1)mockMissing=true;
 for(int step=0;step<=250;++step){nrs->lastStep=step==250;pipeLines::write(step*.001,step);}
 // Duplicate callback must not overwrite an accepted profile.
 ptr(f.o_U)[0]=999;pipeLines::write(.25,250);

}
'''
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory)
    (path/'nekrsSys.hpp').write_text('#pragma once\n')
    (path/'pointInterpolation.hpp').write_text(interpolation)
    (path/'test.cpp').write_text(prefix+main)
    command = ['g++', '-std=c++17', str(path/'test.cpp'), '-I'+str(path),
               '-I'+str(repo/'examples/turbPipe'), '-I'+str(repo/'src/platform'),
               '-I'+str(occa/'include'), '-L'+str(occa/'lib'), '-Wl,-rpath,'+str(occa/'lib'),
               '-locca', '-o', str(path/'test')]
    subprocess.run(command, check=True)
    subprocess.run([str(path/'test')], cwd=path, check=True, capture_output=True)
    files = sorted(path.glob('lineProfiles_step*.csv'))
    assert [int(p.stem.split('step')[1]) for p in files] == [0, 100, 200, 250]
    for file in files:
        rows = list(csv.DictReader(file.open()))
        assert len(rows) == 5*201
        assert {r['line'] for r in rows} == {'x_z25','x_z50','x_z75','y_z50','axial'}
        for i, row in enumerate(rows):
            assert int(row['point']) == i % 201
            for f, field in enumerate(('ux','uy','uz','p','s00')):
                assert float(row[field]) == 10*f+i%3
            assert -.5 < float(row['x']) < .5 and -.5 < float(row['y']) < .5
            assert 0 < float(row['z']) < 20
    disabled = path/'disabled'; disabled.mkdir()
    subprocess.run([str(path/'test'), 'disabled'], cwd=disabled, check=True)
    assert not list(disabled.glob('*.csv'))
    missing=path/'missing';missing.mkdir()
    result=subprocess.run([str(path/'test'),'missing'],cwd=missing,capture_output=True)
    assert result.returncode != 0 and not list(missing.glob('*.csv'))
print('PASS: production exporter, 5 lines, 201 points, field packing, cadence, final step, duplicate guard and disabled mode.')
print('Interpolation and MPI are test doubles; distributed point location requires the full solver run.')
