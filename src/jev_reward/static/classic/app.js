"use strict";
const $ = id => document.getElementById(id);
const colors = {jev:"#267c56",native:"#5279bc",rules:"#c48148"};
const tasks = [
  {slug:"cartpole",name:"CartPole",env_id:"CartPole-v1",steps:60000,objective:"Balance the pole for 500 steps.",success_definition:"Success = all 500 steps completed without the pole falling."},
  {slug:"mountaincar",name:"MountainCar",env_id:"MountainCar-v0",steps:120000,objective:"Build momentum and reach the flag on the right hill.",success_definition:"Success = reaching the right-hand flag within 200 steps."},
  {slug:"acrobot",name:"Acrobot",env_id:"Acrobot-v1",steps:120000,objective:"Swing the two-link pendulum above the target height.",success_definition:"Success = reaching the target height within 500 steps."},
  {slug:"frozenlake",name:"FrozenLake",env_id:"FrozenLake-v1",steps:60000,objective:"Cross a slippery frozen lake without falling into a hole.",success_definition:"Success = reaching the goal within 100 steps. Slipping is enabled."}
];
let manifest = {runs:[]}, task = tasks[0], run = null, taskRuns = [], frameIndex = 0;
let playing = true, lastFrame = 0, backend = false, activeRun = null, shownCheckpoint = -1, requestVersion = 0;
const loaded = new Map();
const format = (n, digits=1) => Number.isFinite(n) ? n.toLocaleString("en-US",{maximumFractionDigits:digits}) : "—";
const pct = n => Number.isFinite(n) ? `${format(n*100,1)}%` : "—";
const point = () => run?.checkpoints?.[Number($("checkpoint").value)];
const replay = () => point()?.evaluation?.replay;

function buildTabs(){
  $("games").replaceChildren();
  tasks.forEach((item,i)=>{
    const button=document.createElement("button");button.className="game-tab";button.dataset.game=item.slug;
    button.setAttribute("role","tab");button.setAttribute("aria-selected",String(item.slug===task.slug));
    const number=document.createElement("span");number.className="number";number.textContent=`0${i+1}`;
    const text=document.createElement("span"),title=document.createElement("b"),sub=document.createElement("small");
    title.textContent=item.name;sub.textContent=item.slug==="frozenlake"?"STOCHASTIC GRID WORLD":"CLASSIC CONTROL";
    text.append(title,sub);button.append(number,text);button.onclick=()=>selectTask(item.slug);$("games").append(button);
  });
}

async function getRun(entry){
  if(!loaded.has(entry.file)) loaded.set(entry.file,fetch(`/static/classic/${entry.file}`).then(r=>{if(!r.ok)throw Error("Recorded run could not be loaded.");return r.json();}));
  return loaded.get(entry.file);
}
async function selectTask(slug){
  const version=++requestVersion;
  task=tasks.find(t=>t.slug===slug);buildTabs();run=null;taskRuns=[];frameIndex=0;
  $("game-title").textContent=task.name;$("env-id").textContent=task.env_id;
  $("objective").textContent=task.objective;$("success-definition").textContent=task.success_definition;
  $("train-steps").value=task.steps;$("error").hidden=true;
  $("success").textContent="—";$("native-return").textContent="—";$("replay-label").textContent="LOADING RECORDED EXPERIMENT";
  try{
    const entries=manifest.runs.filter(r=>r.task===slug);
    const values=await Promise.all(entries.map(getRun));
    if(version!==requestVersion)return;
    taskRuns=values;selectRecorded();drawCharts();
  }catch(error){if(version===requestVersion)showError(error.message);}
}
function selectRecorded(){
  const selected=taskRuns.find(r=>r.provider===$("recorded-provider").value && r.config.seed===Number($("recorded-seed").value));
  if(!selected){run=null;$("replay-label").textContent="NO RECORDED RUN — TRAIN ONE BELOW";$("checkpoint").max=0;$("checkpoint").value=0;return;}
  showRun(selected,true);
}
function showRun(value,latest=false){
  run=value;$("checkpoint").max=Math.max(0,run.checkpoints.length-1);
  if(latest)$("checkpoint").value=$("checkpoint").max;
  $("provider-badge").textContent=run.provider==="jev"?"OFFICIAL JEV REWARD":run.provider==="rules"?"HUMAN DESIGN REWARD":"NATIVE REWARD";
  $("success").textContent=pct(run.test?.success_rate);$("native-return").textContent=format(run.test?.native_return_mean);
  $("test-label").textContent=run.test?`FINAL TEST · SEED ${run.config.seed} · ${run.test.episodes} EPISODES`:"TRAINING · FINAL TEST PENDING";
  $("steps-budget").textContent=format(run.steps,0);
  const model=run.reward?.served_models?.[0];
  $("score-source").textContent=run.provider==="jev"?`${model||"Official JEV"}. Stored answers are reused only for identical abstract observations.`:run.provider==="rules"?"A deterministic implementation of the same human-written scoring rubric. No model calls.":"The original reward from the official Gymnasium environment. No model calls.";
  selectCheckpoint();drawCharts();
}
function selectCheckpoint(){
  frameIndex=0;const p=point();
  if(!p)return;
  $("checkpoint-label").textContent=`Step ${format(p.steps,0)} · episode ${format(p.episode,0)}`;
  $("checkpoint-metric").textContent=`${pct(p.evaluation.success_rate)} success · ${format(p.evaluation.native_return_mean)} total score`;
  $("replay-label").textContent=p.steps===0?"UNTRAINED POLICY":`SAVED POLICY · ${format(p.steps,0)} STEPS`;
  $("replay-seed").textContent=`FIXED REPLAY SEED ${p.evaluation.replay?.seed||10000}`;
  $("replay-return").textContent=`Replay total score ${format(p.evaluation.replay?.native_return)}`;
}
function showError(message){$("error").textContent=message;$("error").hidden=false;}

const ctx=$("game-canvas").getContext("2d");
function line(x1,y1,x2,y2,color,width=2){ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.strokeStyle=color;ctx.lineWidth=width;ctx.lineCap="round";ctx.stroke();}
function circle(x,y,r,fill){ctx.beginPath();ctx.arc(x,y,r,0,2*Math.PI);ctx.fillStyle=fill;ctx.fill();}
function rect(x,y,w,h,fill,r=4){ctx.beginPath();ctx.roundRect(x,y,w,h,r);ctx.fillStyle=fill;ctx.fill();}
function text(value,x,y,color="#a6b9ac",size=12,align="center"){ctx.fillStyle=color;ctx.font=`${size}px ui-monospace,monospace`;ctx.textAlign=align;ctx.fillText(value,x,y);}
function drawGame(){
  const frames=replay()?.frames, f=frames?.[frameIndex], o=f?.observation;
  ctx.clearRect(0,0,900,460);ctx.fillStyle="#142a23";ctx.fillRect(0,0,900,460);
  ctx.strokeStyle="#213a2e";ctx.lineWidth=.6;
  for(let x=30;x<900;x+=40)line(x,50,x,410,"#20392e",.6);
  for(let y=60;y<430;y+=40)line(25,y,875,y,"#20392e",.6);
  if(task.slug==="cartpole"){
    const v=Array.isArray(o)?o:[0,0,.025,0],x=450+v[0]*125, y=310;
    line(90,y+28,810,y+28,"#718779",2);line(150,y+22,150,y+34,"#a8bead");line(750,y+22,750,y+34,"#a8bead");
    text("−2.4",150,365);text("+2.4",750,365);
    rect(x-40,y-15,80,35,"#c8ed81",7);circle(x-23,y+26,10,"#dce9d9");circle(x+23,y+26,10,"#dce9d9");
    const tx=x+Math.sin(v[2])*165,ty=y-Math.cos(v[2])*165;
    line(x,y,tx,ty,"#91baa0",12);circle(x,y,9,"#192d24");circle(tx,ty,8,"#dce9d9");
    text(`ANGLE  ${(v[2]*180/Math.PI).toFixed(1)}°`,450,90,"#abc6b2",11);
  }else if(task.slug==="mountaincar"){
    const v=Array.isArray(o)?o:[-.5,0],sx=p=>110+(p+1.2)/1.8*680,sy=p=>300-Math.sin(3*p)*100;
    ctx.beginPath();for(let i=0;i<=180;i++){let p=-1.2+i/100;i?ctx.lineTo(sx(p),sy(p)):ctx.moveTo(sx(p),sy(p));}ctx.strokeStyle="#91baa0";ctx.lineWidth=3;ctx.stroke();
    line(sx(.5),sy(.5),sx(.5),sy(.5)-60,"#c8ed81",3);ctx.beginPath();ctx.moveTo(sx(.5),sy(.5)-60);ctx.lineTo(sx(.5)+32,sy(.5)-48);ctx.lineTo(sx(.5),sy(.5)-36);ctx.fillStyle="#c8ed81";ctx.fill();
    const x=sx(v[0]),y=sy(v[0]);ctx.save();ctx.translate(x,y-12);ctx.rotate(Math.atan2(-300*Math.cos(3*v[0]),680/1.8));rect(-23,-12,46,20,"#c8ed81",6);circle(-13,11,7,"#dce9d9");circle(13,11,7,"#dce9d9");ctx.restore();
    text(`POSITION ${v[0].toFixed(2)}    VELOCITY ${v[1].toFixed(3)}`,450,90,"#abc6b2",11);
  }else if(task.slug==="acrobot"){
    const v=Array.isArray(o)?o:[1,0,1,0,0,0],scale=77,x=450,y=230;
    const x1=x+scale*v[1],y1=y+scale*v[0],x2=x1+scale*(v[1]*v[2]+v[0]*v[3]),y2=y1+scale*(v[0]*v[2]-v[1]*v[3]);
    ctx.setLineDash([5,7]);line(250,y-scale,650,y-scale,"#809b72",1.5);ctx.setLineDash([]);text("TARGET HEIGHT",650,y-scale-12,"#a8be9e",10,"right");
    line(x,y,x1,y1,"#91baa0",13);line(x1,y1,x2,y2,"#c8ed81",13);circle(x,y,10,"#e8efde");circle(x1,y1,10,"#e8efde");circle(x2,y2,9,"#c8ed81");
    text(`TIP HEIGHT ${((y-y2)/scale).toFixed(2)}`,450,80,"#abc6b2",11);
  }else{
    const map=["SFFF","FHFH","FFFH","HFFG"],s=Number.isInteger(o)?o:0,size=78,left=294,top=70;
    map.forEach((row,y)=>[...row].forEach((tile,x)=>{
      const px=left+x*size,py=top+y*size;rect(px+3,py+3,size-6,size-6,tile==="H"?"#102018":tile==="G"?"#506940":"#456b5c",6);
      if(tile==="H"){circle(px+size/2,py+size/2,21,"#091c14");}
      else if(tile==="G"){text("GOAL",px+size/2,py+size/2+5,"#dbefb0",13);}
      else {line(px+15,py+20,px+33,py+16,"#668c7b",1);if(tile==="S")text("START",px+size/2,py+size/2+5,"#87ad98",10);}
    }));
    const x=left+(s%4)*size+size/2,y=top+Math.floor(s/4)*size+size/2;
    circle(x,y,16,"#c8ed81");circle(x-5,y-2,2,"#183122");circle(x+5,y-2,2,"#183122");line(x-4,y+6,x+4,y+6,"#183122",2);
    text("SLIPPERY",170,220,"#a8be9e",12);text("4 × 4",720,220,"#a8be9e",13);
  }
  $("frame-number").textContent=`STEP ${frameIndex}`;
  $("outcome").textContent=frames&&frameIndex===frames.length-1?(replay().success?"SUCCESS":"EPISODE ENDED"):"POLICY REPLAY";
}
function animate(timestamp){
  const frames=replay()?.frames;
  if(playing&&frames&&timestamp-lastFrame>1000/(30*Number($("speed").value))){
    if(frameIndex<frames.length-1){frameIndex++;lastFrame=timestamp;}else if(timestamp-lastFrame>1600){frameIndex=0;lastFrame=timestamp;}
  }
  drawGame();requestAnimationFrame(animate);
}

function chart(id,series,percent=false){
  const c=$(id),g=c.getContext("2d"),w=c.width,h=c.height,L=65,R=20,T=18,B=42;
  g.clearRect(0,0,w,h);g.font="16px system-ui";
  const all=series.flatMap(s=>s.points),xmax=Math.max(1,...all.map(p=>p.x));
  let ymin=percent?0:Math.min(0,...all.map(p=>p.y-(p.sd||0))),ymax=percent?1:Math.max(1,...all.map(p=>p.y+(p.sd||0)));
  if(ymax===ymin)ymax++;
  const x=n=>L+n/xmax*(w-L-R),y=n=>T+(1-(n-ymin)/(ymax-ymin))*(h-T-B);
  g.lineWidth=1;g.textAlign="right";g.fillStyle="#748073";
  for(let i=0;i<=4;i++){const value=ymin+(ymax-ymin)*i/4,py=y(value);g.strokeStyle="#e0e5da";g.beginPath();g.moveTo(L,py);g.lineTo(w-R,py);g.stroke();g.fillText(percent?`${Math.round(value*100)}%`:format(value,0),L-10,py+5);}
  g.textAlign="center";for(let i=0;i<=4;i++){let value=xmax*i/4;g.fillText(value>=1000?`${format(value/1000,0)}k`:format(value,0),x(value),h-14);}
  series.forEach(s=>{
    if(!s.points.length)return;
    if(s.points.some(p=>p.sd)){g.fillStyle=s.color;g.globalAlpha=.10;g.beginPath();s.points.forEach((p,i)=>i?g.lineTo(x(p.x),y(Math.min(ymax,p.y+p.sd))):g.moveTo(x(p.x),y(Math.min(ymax,p.y+p.sd))));[...s.points].reverse().forEach(p=>g.lineTo(x(p.x),y(Math.max(ymin,p.y-p.sd))));g.closePath();g.fill();g.globalAlpha=1;}
    g.strokeStyle=s.color;g.globalAlpha=s.alpha||1;g.lineWidth=s.width||3;g.beginPath();s.points.forEach((p,i)=>i?g.lineTo(x(p.x),y(p.y)):g.moveTo(x(p.x),y(p.y)));g.stroke();g.globalAlpha=1;
  });
  if(!all.length){g.fillStyle="#7c887a";g.fillText("No measured data loaded",w/2,h/2);}
}
function drawCharts(){
  const aggregate=key=>["jev","native","rules"].map(provider=>{
    const rs=taskRuns.filter(r=>r.provider===provider),points=[];
    if(rs.length)rs[0].checkpoints.forEach(p=>{const values=rs.map(r=>r.checkpoints.find(c=>c.steps===p.steps)?.evaluation[key]).filter(Number.isFinite);if(!values.length)return;const mean=values.reduce((a,b)=>a+b,0)/values.length;points.push({x:p.steps,y:mean,sd:values.length>1?Math.sqrt(values.reduce((a,b)=>a+(b-mean)**2,0)/(values.length-1)):0});});
    return {color:colors[provider],points};
  });
  chart("return-chart",aggregate("native_return_mean"));chart("success-chart",aggregate("success_rate"),true);
  const hs=run?.history||[],smooth=hs.map((p,i)=>({x:p.steps,y:hs.slice(Math.max(0,i-19),i+1).reduce((s,e)=>s+e.reward_return,0)/Math.min(i+1,20)}));
  chart("reward-chart",[{color:colors[run?.provider]||colors.jev,alpha:.22,width:1,points:hs.map(p=>({x:p.steps,y:p.reward_return}))},{color:colors[run?.provider]||colors.jev,points:smooth}]);
  $("reward-caption").textContent=`Selected ${({jev:"JEV",rules:"Human design",native:"Native"})[run?.provider]||"—"} run · seed ${run?.config.seed??"—"} · per-episode reward and 20-episode moving mean · x-axis: environment steps`;
}

async function startTraining(){
  $("error").hidden=true;const source=$("train-provider").value;
  const body={task:task.slug,provider:["recorded","live"].includes(source)?"jev":source,recorded_scores:source!=="live",total_steps:Number($("train-steps").value),checkpoint_steps:Number($("save-every").value),seed:Number($("train-seed").value),max_calls:Number($("max-calls").value)};
  try{const response=await fetch("/api/classic/train",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)}),data=await response.json();if(!response.ok)throw Error(typeof data.detail==="string"?data.detail:"Check the training settings.");activeRun=data.id;shownCheckpoint=-1;$("train").disabled=true;$("stop").hidden=false;await poll();}catch(error){showError(error.message);}
}
async function poll(){
  if(!backend)return;
  try{const response=await fetch("/api/classic/state");if(!response.ok)return;const state=await response.json();if(state.status==="idle")return;
    $("train").disabled=state.status==="running";$("stop").hidden=state.status!=="running";
    $("progress").value=state.steps/Math.max(1,state.total_steps);$("training-status").textContent=`${state.status} · ${format(state.steps,0)} / ${format(state.total_steps,0)} steps`;
    if(state.error)showError(state.error);
    if(activeRun===state.id&&task.slug===state.task){
      if(state.result){showRun(state.result,true);activeRun=null;}
      else if(state.checkpoints.length && state.checkpoints.length!==shownCheckpoint){shownCheckpoint=state.checkpoints.length;showRun({provider:state.provider,config:{seed:state.seed},steps:state.steps,checkpoints:state.checkpoints,history:state.history},true);}
    }
  }catch{/* Keep saved playback usable when the local server is unavailable. */}
}
$("recorded-provider").onchange=selectRecorded;$("recorded-seed").onchange=selectRecorded;
$("checkpoint").oninput=selectCheckpoint;$("before").onclick=()=>{$("checkpoint").value=0;selectCheckpoint();};$("after").onclick=()=>{$("checkpoint").value=$("checkpoint").max;selectCheckpoint();};
$("play-pause").onclick=()=>{playing=!playing;$("play-pause").textContent=playing?"Pause":"Play";$("play-pause").setAttribute("aria-label",playing?"Pause playback":"Resume playback");};$("restart").onclick=()=>{frameIndex=0;};
$("download").onclick=()=>{if(!run)return;const url=URL.createObjectURL(new Blob([JSON.stringify(run,null,2)],{type:"application/json"})),a=document.createElement("a");a.href=url;a.download=`jevrl-${task.slug}-${run.provider}-seed${run.config.seed}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
$("train").onclick=startTraining;$("stop").onclick=async()=>{await fetch("/api/classic/stop",{method:"POST"});$("training-status").textContent="Stopping and saving policy…";};
async function init(){
  buildTabs();requestAnimationFrame(animate);
  try{const r=await fetch("/static/classic/manifest.json");if(r.ok)manifest=await r.json();}catch{/* No bundled results yet. */}
  await selectTask(task.slug);
  try{const r=await fetch("/api/classic/config");if(r.ok){const config=await r.json();backend=config.available;$("backend-status").textContent=backend?"LOCAL TRAINING":"INSTALL CLASSIC EXTRA";if(!config.jev_available)$("train-provider").querySelector('[value="live"]').disabled=true;}}
  catch{/* A static export contains playback only. */}
  if(!backend){$("train").disabled=true;$("backend-status").textContent="REPLAY MODE";$("training-status").textContent="For local training: uv run --extra classic jev-arcade serve";}
  setInterval(poll,700);
}
init();
