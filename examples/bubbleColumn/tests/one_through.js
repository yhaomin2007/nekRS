// Run with: node examples/bubbleColumn/tests/one_through.js
// Executes the actual C++ orchestration translated to JS over CPU test doubles.
// Also executes native sumMakef arithmetic; this is not a C++ compile or CFD run.
import {readFileSync} from 'node:fs';
import assert from 'node:assert/strict';
const root = new URL('../', import.meta.url);
const read = p => readFileSync(new URL(p, root), 'utf8');
const moduleSource = read('bubbleColumnOneThrough.hpp');
const terms = read('bubbleColumnTerms.hpp');
const udf = read('bubbleColumn.udf');
const par = read('bubbleColumn.par');
const native = read('../../../src/solver/scalar/kernels/sumMakef.okl');
const driver = read('../../../src/app/nrs/nrs.cpp');
const scalarDriver = read('../../../src/solver/scalar/scalarSolver.cpp');

function body(source, marker) {
  const a = source.indexOf(marker);
  assert(a >= 0, marker);
  let i = source.indexOf('{', a), depth = 1, j = i + 1;
  while (depth) { depth += (source[j] === '{') - (source[j] === '}'); ++j; }
  return source.slice(i + 1, j - 1);
}
function orchestration(name, args) {
  let s = body(moduleSource, 'inline void ' + name + '(');
  s = s.replace(/\/\/[^\n]*/g, '')
    .replace(/->/g, '.')
    .replace(/auto &scalar/g, 'const scalar')
    .replace(/const (?:dlong|dfloat|int) /g, 'const ')
    .replace(/auto (\w+) =/g, 'let $1 =')
    .replace(/for \(int /g, 'for (let ')
    .replace(/for \(const char \*name : \{([^}]+)\}\)/g, 'for (const name of [$1])')
    .replace(/const char \*names\[\] = \{([^}]+)\};/g, 'const names = [$1];')
    .replace(/scalar.compute = enabled;/g, 'scalar.compute = enabled.slice();');
  return 'function ' + name + '(' + args + '){' + s + '}';
}
function preprocess(s, defines) {
  const stack = []; let active = true, out = [];
  for (const line of s.split('\n')) {
    let m = line.match(/^\s*#if\s+(\w+)/);
    if (m) { stack.push([active, !!defines[m[1]]]); active = active && !!defines[m[1]]; }
    else if (/^\s*#else/.test(line)) { const [parent, truth] = stack.at(-1); active = parent && !truth; }
    else if (/^\s*#endif/.test(line)) { active = stack.pop()[0]; }
    else if (active && !/^\s*#pragma/.test(line)) out.push(line);
  }
  return out.join('\n');
}
function forcing(order) {
  let b = preprocess(body(native, '@kernel void sumMakef('),
      {p_SUBCYCLING:0, p_MovingMesh:0, p_ADVECTION:1});
  b = b.replace(/;\s*@tile\(p_blockSize, @outer, @inner\)/g, '')
    .replace(/const (?:dfloat|dlong) /g, 'const ')
    .replace(/\b(?:dfloat|dlong|int) /g, 'let ')
    .replace(/p_nBDF/g, String(order)).replace(/p_nEXT/g, '3');
  return new Function('N','massMatrix','idt','coeffEXT','coeffBDF',
      'fieldOffsetScan','fieldSumOffset','fieldOffsetJW','RHO','U','ADV','FU','BF', b);
}
class Mem {
  constructor(n, data, start=0) { this.data = data || Array(n).fill(0); this.start=start; this.count=n; }
  size() { return this.count; }
  resize(n) { this.data=Array(n).fill(0); this.start=0; this.count=n; }
  slice(i,n) { return new Mem(n,this.data,this.start+i); }
  copyFrom(b,n,dst=0,src=0) {
    const values=b.data.slice(b.start+src,b.start+src+n);
    for(let i=0;i<n;++i) this.data[this.start+dst+i]=values[i];
  }
  values() { return this.data.slice(this.start,this.start+this.count); }
}
const translated = orchestration('beginStep','time') + orchestration('solveGas','time,tstep');
let total = 0;
for(const order of [1,2,3]) for(const vm of [0,1]) for(const pressureEnabled of [0,1])
for(const temporal of [0,1]) {
  const N=2, offset=4, stride=16, states=Math.max(3,order), mass=new Mem(N); mass.data=[2,7];
  const BDF=order===1?[1]:order===2?[2,-.5]:[3,-1.5,1/3];
  const sumMakef=forcing(order);
  const enabled=[1,1,1,1], names=['alpha','ugx','ugy','ugz'];
  let preparedStep=-1, solvedStep=-1, temporalUpdates=0, postprocessed=0;
  let enabledAtBC=[], solveCounts=[0,0,0,0], hpfCounts=[0,0,0,0], lagCount=0, extShift=0;
  const pressureAtSource=new Mem(12), dragAtSource=new Mem(4), pressureDelta=new Mem(12);
  const zeroAdvection=new Mem(3*stride), o_gradP=new Mem(12), o_dragLambda=new Mem(4);
  const p={rhoGas:1.2,rhoLiquid:998.2,virtualMassEnabled:vm,virtualMassCoefficient:.5,gasPressureEnabled:pressureEnabled};
  const rhoEff=p.rhoGas+vm*.5*p.rhoLiquid;
  let rawPressure=-12, filterAnchor=-8, pressureFilterStep=-1;
  const nonpressure=names.map((_,c)=>[c+2,-c-3]);
  const hpf=names.map((_,c)=>[.01*c,-.02*c]);
  const nrs={tstep:0,dt:[.125],fieldOffset:offset,timeStepConverged:false,meshV:{Nlocal:N,fieldOffset:offset,o_LMM:mass}};
  const scalar=nrs.scalar={
    compute:enabled.slice(), nameToIndex:new Map(names.map((x,i)=>[x,i])),
    fieldOffsetScan:[0,4,8,12],fieldOffsetSum:stride,
    o_coeffEXT:new Mem(3),o_coeffBDF:new Mem(order),
    o_S:new Mem(states*stride),o_EXT:new Mem(3*stride),o_JwF:new Mem(stride),
    rho:new Mem(stride),
    o_transportCoeff(){return this.rho;},
    makeExplicit(is){
      ++hpfCounts[is];
      for(let i=0;i<N;++i)this.o_EXT.data[is*offset+i]+=hpf[is][i];
    },
    applyDirichlet(){enabledAtBC.push(this.compute.slice());},
    solve(time,stage){
      assert.equal(stage,1);
      for(let c=0;c<4;++c)if(this.compute[c]){
        ++solveCounts[c];
        for(let i=0;i<N;++i){
          const diag=(order===1?1:order===2?1.5:11/6)/nrs.dt[0]+(c?o_dragLambda.data[i]:0);
          this.o_S.data[c*offset+i]=this.o_JwF.data[c*offset+i]/mass.data[i]/diag;
        }
      }
    }
  };
  scalar.nameToIndex.at=x=>scalar.nameToIndex.get(x); // C++ std::map::at
  scalar.o_coeffEXT.data=[1,0,0]; scalar.o_coeffBDF.data=BDF;
  scalar.rho.data.fill(1); scalar.o_JwF.data.fill(999);
  for(let h=0;h<states;++h)for(let c=0;c<4;++c)for(let i=0;i<N;++i)
    scalar.o_S.data[h*stride+c*offset+i]=.1+.01*c+.001*i+.002*h;
  const platform={comm:{mpiComm(){return 0;}},linAlg:{
    axpby(n,a,x,b,y){for(let i=0;i<n;++i)y.data[y.start+i]=a*x.data[x.start+i]+b*y.data[y.start+i];},
    axmy(n,a,x,y){for(let i=0;i<n;++i)y.data[y.start+i]*=a*x.data[x.start+i];}
  }};
  const EXIT_FAILURE=1;
  function nekrsCheck(f){if(f)throw Error('order/state guard');}
  function evaluateGasPressureGradient(advance) {
    if(temporal&&advance&&pressureFilterStep!==nrs.tstep){
      filterAnchor=.8*filterAnchor+.2*rawPressure;pressureFilterStep=nrs.tstep;++temporalUpdates;
    }
    for(let c=0;c<3;++c)for(let i=0;i<N;++i)
      o_gradP.data[c*offset+i]=i===0?0:(temporal?filterAnchor:rawPressure)*(c+1);
  }
  function addExplicitSources(){
    evaluateGasPressureGradient(false);
    for(let c=0;c<4;++c)for(let i=0;i<N;++i){
      const pressure=c?-pressureEnabled*o_gradP.data[(c-1)*offset+i]/rhoEff:0;
      scalar.o_EXT.data[c*offset+i]=nonpressure[c][i]+pressure;
    }
    o_dragLambda.data[0]=2;o_dragLambda.data[1]=3;
  }
  function launchKernel(name,...args){
    assert.equal(name,'scalar_t::sumMakef');
    const values=args.map(a=>a instanceof Mem?a.values():a);
    sumMakef(...values);
    args.at(-1).data.splice(0,stride,...values.at(-1));
  }
  function postProcessGasFlux(){++postprocessed;}
  // Evaluate actual orchestration in the mock lexical environment.
  const api=eval('('+ 'function(){'+translated+';return {beginStep,solveGas};}' +')')();
  for(let step=1;step<=2;++step){
    nrs.tstep=step; nrs.timeStepConverged=false;
    scalar.o_EXT.data.fill(0);const old=scalar.o_S.values();
    api.beginStep(step-1);
    assert.deepEqual(scalar.compute,[1,0,0,0]);
    assert.deepEqual(solveCounts,Array(4).fill(step-1));
    // Native makeForcing: only alpha remains enabled, but ALL histories shift.
    scalar.makeExplicit(0);launchKernel('scalar_t::sumMakef',N,mass,1/nrs.dt[0],
      scalar.o_coeffEXT,scalar.o_coeffBDF,0,stride,offset,scalar.rho,
      scalar.o_S,zeroAdvection,scalar.o_EXT,scalar.o_JwF);
    for(let h=2;h>0;--h)scalar.o_EXT.copyFrom(scalar.o_EXT,stride,h*stride,(h-1)*stride);
    ++extShift;
    for(let h=states-1;h>0;--h)scalar.o_S.copyFrom(scalar.o_S,stride,h*stride,(h-1)*stride);
    ++lagCount;
    scalar.applyDirichlet();scalar.solve(step,1);
    assert.deepEqual(solveCounts,[step,step-1,step-1,step-1]);
    const alpha=scalar.o_S.values().slice(0,offset),alphaRhs=scalar.o_JwF.values().slice(0,offset);
    const solutionHistory=scalar.o_S.values().slice(stride),extHistory=scalar.o_EXT.values();
    // UM properties may refresh scratch drag, but must not change the frozen UG coefficient.
    o_dragLambda.data[0]=1000;o_dragLambda.data[1]=2000;
    rawPressure=-21-step;nrs.timeStepConverged=true; // one native UM/pressure solve
    api.solveGas(step,step);
    assert.deepEqual(scalar.compute,enabled);assert.deepEqual(solveCounts,Array(4).fill(step));
    assert.deepEqual(hpfCounts,Array(4).fill(step));assert.equal(postprocessed,step);
    assert.deepEqual(enabledAtBC.at(-1),[0,1,1,1]);
    assert.deepEqual(o_dragLambda.values().slice(0,N),[2,3]);
    assert.deepEqual(scalar.o_S.values().slice(0,offset),alpha);
    assert.deepEqual(scalar.o_JwF.values().slice(0,offset),alphaRhs);
    assert.deepEqual(scalar.o_S.values().slice(stride),solutionHistory);
    assert.deepEqual(scalar.o_EXT.values(),extHistory);
    for(let c=1;c<4;++c)for(let i=0;i<N;++i){
      const bdf=BDF.reduce((v,b,h)=>v+b*old[h*stride+c*offset+i]/nrs.dt[0],0);
      const expected=mass.data[i]*(bdf+nonpressure[c][i]+hpf[c][i]
        -pressureEnabled*o_gradP.data[(c-1)*offset+i]/rhoEff);
      assert(Math.abs(scalar.o_JwF.data[c*offset+i]-expected)<1e-10);
      assert.equal(scalar.o_JwF.data[c*offset+N],999); // padded tail untouched
    }
    assert.throws(()=>api.solveGas(step,step));
  }
  assert.equal(temporalUpdates,temporal?2:0);assert.equal(lagCount,2);assert.equal(extShift,2);
  ++total;
}
assert(!/picard/i.test(moduleSource+terms+udf+par));
assert(udf.indexOf('oneThrough::solveGas(time, tstep)')<udf.indexOf('diagnoseDivergenceCoupling'));
assert(udf.indexOf('oneThrough::solveGas(time, tstep)')<udf.indexOf('updateVirtualMassHistory'));
assert(!body(terms,'inline void updateProperties(').includes('postProcessGasFlux'));
assert(terms.includes('advanceTemporalFilter && tstep != pressureGradientFilterStep'));
assert(moduleSource.includes('p.scalarExtrapolationEnabled'));
assert(moduleSource.includes('nrs->userConvergenceCheck = &singlePass'));
const run=body(driver,'bool nrs_t::runInnerStep(');
assert(run.indexOf('scalar->solve(')<run.indexOf('fluid->solve('));
assert(run.indexOf('fluid->solve(')<run.indexOf('udf.executeStep('));
assert(!body(scalarDriver,'void scalar_t::lagSolution(').includes('compute['));
const forcingDriver=body(scalarDriver,'void scalar_t::makeForcing(');
assert(!forcingDriver.slice(forcingDriver.indexOf('const auto n =')).includes('compute['));
console.log(total+' actual-module orchestration/forcing cases passed: BDF1/2/3, VM, pressure on/off, filtering, two steps, ordering, masks, drag pairing, HPF/BDF/EXT histories, padding and duplicate guards.');
