// Arithmetic audit of production kernel bodies; not an OCCA compilation test.
function audit(source) {
  const assert = (ok, message) => { if (!ok) throw new Error(message); };
  function kernel(name, args) {
    const start=source.indexOf("@kernel void "+name+"(");
    const open=source.indexOf("{",start);
    let end=open+1,depth=1;
    for(;depth;end++) {if(source[end]==="{")depth++;if(source[end]==="}")depth--;}
    let body=source.slice(open+1,end-1)
      .replace(/;\s*@tile\(p_blockSize,@outer,@inner\)/g,"")
      .replace(/\bconst (dfloat|dlong|int)\b/g,"const")
      .replace(/\b(dfloat|dlong|int)\b/g,"let");
    return new Function(...args,body);
  }
  const rk=kernel("dgRK",["N","stage","h","old","rhs","increment","a"]);
  const sf=kernel("dgSurfaceFlux",["NF","inletAlpha","backflowAlpha","fluxType","bid","sgeo","local","sum","surface","boundary","p_Nfp","p_Nsgeo","p_WSJID"]);
  function flux(type,b,am,ap,um,up){
    let face=[0],boundary=[0];
    sf(1,.7,.4,type,[b],[1],[am,um],[am+ap,um+up],face,boundary,1,1,0);
    return face[0];
  }
  for(let type=0;type<3;type++){
    for(let i=0;i<100;i++){
      let a=.01*i,b=.3,u=Math.sin(i),v=Math.cos(i);
      assert(Math.abs(flux(type,0,a,b,u,v)+flux(type,0,b,a,v,u))<1e-14,"paired flux");
    }
    assert(flux(type,3,.2,.8,3,-2)===0,"wall flux");
    assert(Math.abs(flux(type,1,.2,0,2,0)-.4)<1e-14,"outflow extrapolation");
    assert(Math.abs(flux(type,1,.2,0,-2,0)+(type===0?.9:1.4))<1e-14,"inflow");
    assert(Math.abs(flux(type,2,.2,0,-2,0)+(type===0?.6:.8))<1e-14,"backflow");
  }
  let errors=[];
  for(let steps of [10,20,40]){
    let a=[1],increment=[0],h=1/steps;
    for(let i=0;i<steps;i++){let old=[a[0]];for(let stage=0;stage<4;stage++)rk(1,stage,h,old,[a[0]],increment,a);}
    errors.push(Math.abs(a[0]-Math.E));
  }
  assert(errors[0]/errors[1]>15 && errors[1]/errors[2]>15,"fourth order");
  // Nonautonomous volume loss with solution-dependent boundary flux.
  let a=[1],old=[1],inc=[0],h=.2,integral=0;
  for(let stage=0;stage<4;stage++){
    let t=h*[0,.5,.5,1][stage],b=a[0]+t*t*t;
    integral+=h*[1/6,1/3,1/3,1/6][stage]*b;
    rk(1,stage,h,old,[-b],inc,a);
  }
  assert(Math.abs(a[0]-old[0]+integral)<1e-15,"RK4 boundary balance");
  return {errors,orderRatios:[errors[0]/errors[1],errors[1]/errors[2]],balance:a[0]-old[0]+integral};
}

const fs=require("node:fs");
const path=require("node:path");
console.log(audit(fs.readFileSync(path.join(__dirname,"../alphaDG.okl"),"utf8")));
