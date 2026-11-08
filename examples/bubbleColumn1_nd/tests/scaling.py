"""Check parameters and compare the production SI and ND pointwise sources."""
from configparser import ConfigParser
from math import isclose, sqrt
from pathlib import Path
import re
import subprocess
import tempfile

case = Path(__file__).resolve().parents[1]
baseline = case.parent / 'bubbleColumn'


def read_par(path):
    result = ConfigParser(interpolation=None)
    result.read_string('[TOP]\n' + path.read_text())
    return result


si = read_par(baseline / 'bubbleColumn.par')
nd = read_par(case / 'bubbleColumn1_nd.par')
L, U, rho = 0.1, float(si['CASEDATA']['gasInletVelocity']), float(si['CASEDATA']['rhoLiquid'])


def check(section, key, factor):
    assert isclose(float(nd[section][key]), factor * float(si[section][key]), rel_tol=1e-14), (section, key)


check('GENERAL', 'dt', U/L)
check('FLUID VELOCITY', 'rho', 1/rho)
check('FLUID VELOCITY', 'viscosity', 1/(rho*U*L))
for field, factor in [('FLUID PRESSURE', sqrt(L)/U**2),
                      ('FLUID VELOCITY', 1/(sqrt(L)*rho*U**2)),
                      ('SCALAR ALPHA', 1/(sqrt(L)*U)),
                      # UG trial tightens the SI-equivalent ND tolerance by 100.
                      *[('SCALAR UG'+axis, 0.01/(sqrt(L)*U**2)) for axis in 'XYZ']]:
    check(field, 'residualTol', factor)
    if field.startswith('SCALAR'):
        check(field, 'diffusionCoeff', 1/(U*L))
    if field != 'FLUID PRESSURE':
        rate = lambda p: float(p[field]['regularization'].split('scalingCoeff=')[1])
        assert isclose(rate(nd), rate(si)*L/U)
for keys, factor in [(['rhoLiquid','rhoGas'], 1/rho), (['muLiquid','muGas'], 1/(rho*U*L)),
                     (['gasInletVelocity','gasInitialVelocity','gasVelocityMaximum','mixtureVelocityMaximum'], 1/U),
                     (['initialPlumeHeight','initialPlumeThickness','zRampBottom','zRampTop',
                       'bubbleDiameter','divergenceInletThickness'], 1/L),
                     (['gravityX','gravityY','gravityZ'], L/U**2)]:
    for key in keys:
        check('CASEDATA', key, factor)


keys = ('rhoLiquid','rhoGas','muLiquid','muGas','alphaInlet','gasInletVelocity',
        'gasInitialVelocity','initialPlumeHeight','initialPlumeThickness','gravityZ','bubbleDiameter')
source = '''#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <string_view>
using dfloat = double; using dlong = long;
struct bcData {
 int id, idxVol;
 double uxFluid, uyFluid, uzFluid, nx, ny, nz, pFluid, sScalar;
 double* usrwrk;
};
std::string_view currentField;
bool isField(std::string_view field) { return field == currentField; }
struct Input {double rl,rg,ml,mg,alpha,ug,ugInitial,height,thickness,g,d;};
'''
source += f'constexpr double L={L}, U={U}, R={rho};\n'
for label, parent, p in [('si',baseline,si), ('nd',case,nd)]:
    values = ','.join(p['CASEDATA'][key] for key in keys)
    source += f'namespace {label} {{\nconstexpr Input in{{{values}}};\n'
    for macro, key in [('RHO_LIQUID','rhoLiquid'),('RHO_GAS','rhoGas'),('ALPHA_INLET','alphaInlet'),('UG_INLET','gasInletVelocity')]:
        source += f'constexpr double p_{macro} = {p["CASEDATA"][key]};\n'
    if label == 'nd':
        source += 'constexpr double p_ND_INV_DENSITY_REF=1/R, p_ND_OUTLET_INV_VELOCITY=20*U;\n'
    source += (parent/'bubbleColumnBoundary.oudf').read_text() + '}\n'

okl = (case/'bubbleColumnEquations.okl').read_text().replace('@kernel ', '').replace('@ restrict ', '')
source += re.sub(r'; @tile\(p_blockSize, @outer, @inner\)', '', okl)
source += r'''
void same(double a, double b) {
  assert(std::abs(a-b) < 2e-12*std::max({1.,std::abs(a),std::abs(b)}));
}
struct Result { double a, ug[3], um[3], r, mu, drift[9], stress[9], sa, su[3], lambda, al[3]; };
Result evaluate(Input p, double x, double v, double t, double rho, int force) {
  Result o{};
  double z=.055/x;
  initializePlume(1,1,p.rl,p.rg,p.alpha,p.ugInitial,p.height,p.thickness,&z,
                  &o.a,o.ug,o.ug+1,o.ug+2,o.um);
  double ul[3]={.05/v,.02/v,-.05/v}, prev[3]={.04/v,.021/v,-.049/v};
  double gu[9]={.25*t,-.03*t,0,0,-.11*t,0,0,0,-.16*t};
  updateVirtualMassHistory(1,1,t/1e-3,ul,prev,gu,o.al);
  double gp[3]={1*x/(rho*v*v),-2*x/(rho*v*v),-9810*x/(rho*v*v)};
  double ga[3]={},dq[1]={};
  buildEquationTerms(1,1,p.rl,p.rg,p.ml,p.mg,1e-6,force==0,force==1,p.d,
                     force==2,.5,0,0,force==3?p.g:0,&o.a,o.um,o.ug,ul,ga,dq,gu,o.al,gp,
                     &o.r,&o.mu,o.drift,o.stress,&o.sa,o.su,&o.lambda);
  return o;
}
int main() {
  for(int force=0;force<4;++force) {
    const Result a=evaluate(si::in,1,1,1,1,force), b=evaluate(nd::in,L,U,L/U,R,force);
    same(a.a,b.a); same(a.r,b.r*R); same(a.mu,b.mu*R*U*L);
    same(a.sa,b.sa*U/L); same(a.lambda,b.lambda*U/L);
    for(int i=0;i<3;++i) {
      same(a.ug[i],b.ug[i]*U); same(a.um[i],b.um[i]*U);
      same(a.su[i],b.su[i]*U*U/L); same(a.al[i],b.al[i]*U*U/L);
    }
    for(int i=0;i<9;++i) {same(a.drift[i],b.drift[i]*R*U*U); same(a.stress[i],b.stress[i]*R*U*U);}
  }
  double weight=1;
  bcData a{1,0,0,0,0,0,0,1,0,0,&weight}, b=a;
  currentField="fluid velocity"; si::udfDirichlet(&a); nd::udfDirichlet(&b);
  same(a.uzFluid,b.uzFluid*U);
  currentField="scalar ugz"; si::udfDirichlet(&a); nd::udfDirichlet(&b);
  same(a.sScalar,b.sScalar*U); same(b.sScalar,1);
  currentField="fluid pressure";
  a.id=b.id=2; a.uzFluid=-.08; b.uzFluid=a.uzFluid/U;
  si::udfDirichlet(&a); nd::udfDirichlet(&b);
  same(a.pFluid,b.pFluid*R*U*U);
}
'''
with tempfile.TemporaryDirectory() as d:
    src = Path(d) / 'scaling.cpp'
    src.write_text(source)
    exe = src.with_suffix('')
    subprocess.run(['c++','-std=c++17',str(src),'-o',str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
print('SI/ND input, inlet and outlet BC, plume, drag, VM, pressure, gravity and stress scaling passed.')
