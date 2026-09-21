/* Shared by the local training dashboard and the portable static replay. */
'use strict';
const $ = id => document.getElementById(id);
const MAP = ['#########','#...K...#','#.#.L.#.#','#.#...#.#','#.LL#...#','#......E#','#########'];
const DELTAS = [[0,-1],[1,0],[0,1],[-1,0]];
const LABELS = ['Lava','Blocked','Farther','Closer','Key','Win'];
const WEIGHTS = [-1,-.15,-.12,.08,.6,1];
let result = null, online = false, running = false, view = 'trained', frames = [], frameIndex = 0;
let paused = false, lastTick = 0, version = -1, activeRunId = null, liveFrames = [];
let manual = {x:1,y:5,has_key:false,event:'start',action:null}, manualSteps = 0;
let currentFrame = manual, trail = [], exportData = null, gateway = 'typesafe';
let checkpoints = [], checkpointIndex = -1;
const canvas = $('game'), ctx = canvas.getContext('2d');

function roundRect(x,y,w,h,r,fill,stroke) {
  ctx.beginPath(); ctx.roundRect(x,y,w,h,r);
  if(fill){ctx.fillStyle=fill;ctx.fill();}
  if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=1;ctx.stroke();}
}

function draw(frame, now = 0) {
  const w=canvas.width,h=canvas.height,t=88,ox=(w-9*t)/2,oy=(h-7*t)/2;
  ctx.clearRect(0,0,w,h);ctx.fillStyle='#111b13';ctx.fillRect(0,0,w,h);
  const gradient=ctx.createRadialGradient(w*.55,h*.5,30,w*.5,h*.5,500);
  gradient.addColorStop(0,'#243222');gradient.addColorStop(1,'#111a13');ctx.fillStyle=gradient;ctx.fillRect(0,0,w,h);
  for(let y=0;y<7;y++) for(let x=0;x<9;x++) {
    const px=ox+x*t,py=oy+y*t,tile=MAP[y][x];
    if(tile==='#') {
      roundRect(px+3,py+3,t-6,t-6,8,'#263723','#3a4b30');
      ctx.fillStyle='#34472b';ctx.fillRect(px+12,py+12,t-24,3);
      ctx.fillStyle='#182515';ctx.fillRect(px+11,py+t-15,t-22,3);
      ctx.fillStyle='#415137';ctx.fillRect(px+13,py+20,4,4);
    } else {
      roundRect(px+3,py+3,t-6,t-6,7,'#1a2719','#2a3924');
      ctx.fillStyle='#35482a';ctx.fillRect(px+18,py+24,3,3);ctx.fillRect(px+58,py+62,3,3);
      if(tile==='L') {
        roundRect(px+7,py+7,t-14,t-14,7,'#522f22','#935139');
        ctx.fillStyle='#d56c48';
        ctx.beginPath();ctx.moveTo(px+15,py+46);ctx.lineTo(px+31,py+22);ctx.lineTo(px+42,py+46);ctx.lineTo(px+62,py+25);ctx.lineTo(px+75,py+67);ctx.lineTo(px+18,py+70);ctx.fill();
        ctx.fillStyle='#ffb66a';ctx.fillRect(px+31,py+49,7,7);ctx.fillRect(px+60,py+53,5,5);
      }
      if(tile==='K' && !frame.has_key) {
        ctx.save();ctx.translate(px+t/2,py+t/2);ctx.rotate(-Math.PI/4);
        ctx.shadowBlur=19;ctx.shadowColor='#d4af51';ctx.strokeStyle='#f3d27d';ctx.lineWidth=8;
        ctx.beginPath();ctx.arc(0,-12,12,0,Math.PI*2);ctx.stroke();ctx.shadowBlur=0;
        ctx.fillStyle='#f3d27d';ctx.fillRect(-4,0,8,29);ctx.fillRect(0,17,13,7);ctx.restore();
      }
      if(tile==='E') {
        roundRect(px+19,py+10,50,68,6,frame.has_key?'#244d3b':'#263c31',frame.has_key?'#a2e7a2':'#629083');
        roundRect(px+27,py+18,34,52,3,'#101d17',null);
        ctx.fillStyle=frame.has_key?'#bffb84':'#779887';ctx.fillRect(px+53,py+43,4,7);
        if(frame.has_key){ctx.shadowColor='#b2f985';ctx.shadowBlur=20;ctx.fillStyle='#c4f76d';ctx.fillRect(px+27,py+18,34,3);ctx.shadowBlur=0;}
      }
    }
  }
  // The trail is observed movement, never an oracle route.
  if(trail.length>1){ctx.strokeStyle='#c4f743';ctx.lineWidth=4;ctx.lineCap='round';ctx.setLineDash([3,12]);ctx.beginPath();trail.forEach((p,i)=>{const x=ox+(p.x+.5)*t,y=oy+(p.y+.5)*t;i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();ctx.setLineDash([]);}
  const ax=ox+(frame.x+.5)*t,ay=oy+(frame.y+.5)*t;
  ctx.fillStyle='#0005';ctx.beginPath();ctx.ellipse(ax,ay+26,24,8,0,0,Math.PI*2);ctx.fill();
  ctx.shadowColor='#c4f76d';ctx.shadowBlur=18;roundRect(ax-24,ay-27,48,48,12,frame.event==='lava'?'#e58c6d':'#c4f76d',null);ctx.shadowBlur=0;
  roundRect(ax-17,ay-13,34,19,5,'#243a1b',null);
  ctx.fillStyle='#e9ffd0';ctx.fillRect(ax-11,ay-8,6,7);ctx.fillRect(ax+5,ay-8,6,7);
  ctx.fillStyle='#83ad43';ctx.fillRect(ax-15,ay+21,9,7);ctx.fillRect(ax+6,ay+21,9,7);
  if(frame.has_key){ctx.fillStyle='#f3d27d';ctx.beginPath();ctx.arc(ax+25,ay-24,7,0,Math.PI*2);ctx.fill();}
}

function renderVerdict(verdict) {
  const reward=verdict?.reward;
  $('reward').textContent=reward==null?'—':`${reward>=0?'+':''}${reward.toFixed(2)}`;
  $('reward').style.color=reward<0?'#ef987e':'var(--lime)';
  $('verdict-label').textContent=verdict?.label || (view==='play'?'Your move. No judge calls in manual play.':'No recorded judgment for this frame.');
  $('confidence').textContent=verdict ? `${Math.round(verdict.confidence*100)}%`:'—';
  $('confidence-bar').style.width=verdict ? `${verdict.confidence*100}%`:'0%';
  $('verdict-source').textContent=verdict ? `${verdict.provider==='jev'?'JEV':'RULES'}${verdict.cached?' · CACHED':''}`:'NO JUDGMENT';
  $('distribution').replaceChildren(...LABELS.map((name,i)=>{
    const row=document.createElement('div');row.className='dist-row';
    const label=document.createElement('span');label.className='dist-name';label.textContent=name;
    const track=document.createElement('div');track.className='dist-track';
    const bar=document.createElement('div');bar.className='dist-fill';bar.style.width=`${(verdict?.probabilities?.[String(i)] || 0)*100}%`;track.append(bar);
    const value=document.createElement('span');value.className='dist-value';value.textContent=`${Math.round((verdict?.probabilities?.[String(i)] || 0)*100)}%`;
    row.append(label,track,value);return row;
  }));
}

function showFrame(frame, index) {
  currentFrame=frame;
  $('step-label').textContent=`STEP ${String(index).padStart(2,'0')}`;
  $('objective-label').textContent=frame.has_key?'REACH THE EXIT':'FIND THE KEY';
  const message={win:'QUEST COMPLETE  +',lava:'LAVA HIT · TRY AGAIN',key:'KEY COLLECTED → EXIT UNLOCKED',locked:'EXIT LOCKED · FIND THE KEY'}[frame.event];
  $('game-toast').textContent=message||'';$('game-toast').classList.toggle('visible',!!message);
  renderVerdict(frame.verdict);draw(frame);
}

function setView(next) {
  view=next;frameIndex=0;lastTick=0;trail=[];paused=false;
  if(checkpoints.length&&next==='trained')checkpointIndex=checkpoints.length-1;
  if(checkpoints.length&&next==='before')checkpointIndex=0;
  checkpointLabel();
  document.querySelectorAll('[data-view]').forEach(b=>b.classList.toggle('active',b.dataset.view===next));
  $('dpad').hidden=next!=='play';$('pause').disabled=next==='play';$('speed').disabled=next==='play';
  $('pause').textContent='Ⅱ';$('pause').setAttribute('aria-label','Pause playback');
  $('view-label').textContent=next==='play'?'YOUR TURN':next==='before'?'UNTRAINED POLICY':running?'LIVE TRAINING':'TRAINED POLICY';
  $('game-tip').textContent=next==='play'?'Arrow keys / WASD · R to restart':'Collect the key. Avoid lava. Reach the exit.';
  if(next==='play'){manual={x:1,y:5,has_key:false,event:'start',action:null};manualSteps=0;showFrame(manual,0);canvas.focus({preventScroll:true});return;}
  frames=(next==='before'?result?.before_replay?.frames:running?liveFrames:result?.replay?.frames)||[];
  if(frames.length)showFrame(frames[0],0);
}

function manualMove(action) {
  if(view!=='play'||['win','lava'].includes(manual.event))return;
  const [dx,dy]=DELTAS[action],x=manual.x+dx,y=manual.y+dy,tile=MAP[y]?.[x]||'#';
  const before={...manual};manualSteps++;
  if(tile==='#')manual={...manual,event:'wall',action};
  else if(tile==='E'&&!manual.has_key)manual={...manual,event:'locked',action};
  else manual={x,y,has_key:manual.has_key||tile==='K',action,event:tile==='L'?'lava':tile==='E'?'win':tile==='K'&&!manual.has_key?'key':'move'};
  trail.push(before);trail=trail.slice(-25);showFrame(manual,manualSteps);
}

function checkpointLabel() {
  const point=checkpoints[checkpointIndex];
  if(!point)return;
  $('checkpoint-slider').value=checkpointIndex;
  $('checkpoint-title').textContent=`EPISODE ${String(point.episode).padStart(3,'0')}${point.partial_episode?' · PARTIAL':''}`;
  $('checkpoint-meta').textContent=`${Math.round(point.evaluation.success_rate*100)}% win rate · ${point.replay.steps} steps in replay`;
  $('checkpoint-prev').disabled=checkpointIndex<=0;
  $('checkpoint-next').disabled=checkpointIndex>=checkpoints.length-1;
}

function updateCheckpoints(points, latest=false) {
  checkpoints=points||[];
  $('checkpoint-panel').hidden=!checkpoints.length;
  $('checkpoint-slider').max=Math.max(0,checkpoints.length-1);
  if(latest||checkpointIndex<0)checkpointIndex=checkpoints.length-1;
  checkpointIndex=Math.min(checkpointIndex,checkpoints.length-1);
  checkpointLabel();
}

function selectCheckpoint(index) {
  if(!checkpoints.length)return;
  checkpointIndex=Math.max(0,Math.min(Number(index),checkpoints.length-1));
  const point=checkpoints[checkpointIndex];
  view='checkpoint';paused=false;frameIndex=0;lastTick=0;trail=[];
  frames=point.replay.frames;
  document.querySelectorAll('[data-view]').forEach(b=>b.classList.toggle('active',b.dataset.view==='trained'));
  $('dpad').hidden=true;$('pause').disabled=false;$('speed').disabled=false;
  $('pause').textContent='Ⅱ';$('pause').setAttribute('aria-label','Pause playback');
  $('view-label').textContent=`SAVED POLICY · EP ${point.episode}`;
  $('game-tip').textContent='A replay from this saved policy. Same evaluation seed.';
  checkpointLabel();showFrame(frames[0],0);
}

function chart(history=[]) {
  const w=700,h=165,left=35,right=10,top=20,bottom=28;
  const points=history.map((row,i)=>{const window=history.slice(Math.max(0,i-19),i+1);return window.filter(r=>r.success).length/window.length;});
  const path=points.map((p,i)=>`${i?'L':'M'}${left+i/Math.max(1,points.length-1)*(w-left-right)},${top+(1-p)*(h-top-bottom)}`).join(' ');
  const lines=[0,.5,1].map(p=>{const y=top+(1-p)*(h-top-bottom);return `<line x1="${left}" y1="${y}" x2="${w-right}" y2="${y}" stroke="#303a29" stroke-dasharray="3 5"/><text x="0" y="${y+3}" fill="#819174" font-size="9">${p*100}%</text>`;}).join('');
  $('chart').innerHTML=`<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" aria-hidden="true"><defs><linearGradient id="chartFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#c4f76d" stop-opacity=".18"/><stop offset="100%" stop-color="#c4f76d" stop-opacity="0"/></linearGradient></defs>${lines}${points.length?`<path d="${path} L${w-right},${h-bottom} L${left},${h-bottom} Z" fill="url(#chartFill)"/><path d="${path}" fill="none" stroke="#c4f76d" stroke-width="2"/>`:''}<text x="${left}" y="${h-7}" fill="#819174" font-size="9">1</text><text x="${w-right}" y="${h-7}" fill="#819174" font-size="9" text-anchor="end">${history.length || 0} episodes</text></svg>`;
  $('chart-episodes').textContent=`${history.length} EPISODES`;
  $('epsilon').textContent=`ε ${history.at(-1)?.epsilon?.toFixed(2) || '1.00'}`;
}

function stats(judge={}) {
  $('api-calls').textContent=judge.api_calls||0;$('cache-hits').textContent=judge.cache_hits||0;
  $('api-cost').textContent=judge.cost_usd==null?'—':`$${judge.cost_usd.toFixed(5)}`;
  if(judge.gateway&&judge.gateway!=='local')gateway=judge.gateway;
}

function provenance(provider, live=false) {
  $('provider-badge').textContent=provider==='jev'?'JEV REWARDS':'RULES DEMO';
  $('source-note').textContent=provider==='jev'?`Official JEV via ${gateway==='openrouter'?'OpenRouter':'TypeSafe'} · structured state → Score → reward`:`${live?'Live':'Recorded'} CPU training · deterministic judge · no JEV API calls`;
}

function loadResult(data) {
  result=data;exportData=data;
  chart(data.history);stats(data.judge);provenance(data.provider);
  $('before-rate').textContent=`${Math.round(data.before.success_rate*100)}%`;
  $('after-rate').textContent=`${Math.round(data.after.success_rate*100)}%`;
  $('session-status').textContent=`${data.status.toUpperCase()} / SEED ${data.config.seed}`;
  $('progress-bar').style.width=`${100*data.history.length/data.config.episodes}%`;
  $('progress-count').textContent=`${data.history.length} / ${data.config.episodes}`;
  $('progress-caption').textContent=data.status==='completed'?'Training complete':data.status==='stopped'?'Stopped · progress saved':'Failed · progress saved';
  $('export').disabled=false;setView('trained');
  updateCheckpoints(data.checkpoints,true);
  if(data.error)showError(data.error);
}

function showError(message) { $('error').textContent=message;$('error').hidden=false; }
function controls() {
  ['provider','episodes','seed','budget','checkpoint-every'].forEach(id=>$(id).disabled=running||!online);
  $('train').disabled=running||!online;$('stop').hidden=!running;
  $('train').firstChild.textContent=running?'Learning… ':online?'Start learning ':'Replay mode ';
}

async function api(path,body) {
  const response=await fetch(path,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  let data;try{data=await response.json();}catch{throw new Error('Training server returned an unreadable response.');}
  if(!response.ok)throw new Error(typeof data.detail==='string'?data.detail:'Invalid run settings. Check episodes, seed and budget.');
  return data;
}

async function poll() {
  if(!online)return;
  try {
    const state=await api('api/state');
    if(state.status==='idle')return;
    if(activeRunId!==state.id){activeRunId=state.id;version=-1;}
    if(state.version===version&&running===(state.status==='running'))return;
    version=state.version;running=state.status==='running';controls();
    if(running){
      provenance(state.provider,true);chart(state.history);stats(state.judge);
      const count=state.history.length;$('progress-bar').style.width=`${count/state.total_episodes*100}%`;
      $('progress-count').textContent=`${count} / ${state.total_episodes}`;
      $('progress-caption').textContent='Exploring, judging, learning';$('session-status').textContent='TRAINING IN PROGRESS';
      $('before-rate').textContent=state.before?`${Math.round(state.before.success_rate*100)}%`:'—';$('after-rate').textContent='…';
      liveFrames=state.frames||[];
      updateCheckpoints(state.checkpoints);
      if(view==='trained'&&liveFrames.length){frames=liveFrames;frameIndex=0;trail=[];$('view-label').textContent=`TRAINING · EP ${count}`;}
    } else if(state.result)loadResult(state.result);
    else if(state.error)showError(state.error);
  } catch(error){showError(`Server connection interrupted: ${error.message}`);}
}

$('train').addEventListener('click',async()=>{
  $('error').hidden=true;
  const request={provider:$('provider').value,episodes:Number($('episodes').value),seed:Number($('seed').value),max_calls:Number($('budget').value),checkpoint_every:Number($('checkpoint-every').value)};
  if(!Number.isInteger(request.checkpoint_every)||request.checkpoint_every<1||request.checkpoint_every>5000){showError('Save interval must be an integer from 1 to 5000.');return;}
  if(!Number.isInteger(request.episodes)||request.episodes<1||request.episodes>5000||!Number.isInteger(request.seed)||request.seed<0||request.seed>4294967295||!Number.isInteger(request.max_calls)||request.max_calls<1||request.max_calls>2000){showError('Use 1–5000 episodes, a nonnegative integer seed, and 1–2000 API attempts.');return;}
  $('train').disabled=true;
  try{const start=await api('api/train',request);activeRunId=start.id;running=true;version=-1;liveFrames=[];controls();chart([]);setView('trained');await poll();}catch(error){showError(error.message);controls();}
});
$('stop').addEventListener('click',async()=>{try{await api('api/stop',{});$('progress-caption').textContent='Stopping after current judgment…';}catch(error){showError(error.message);}});
$('provider').addEventListener('change',()=>{$('budget-field').hidden=$('provider').value!=='jev';});
document.querySelectorAll('[data-view]').forEach(button=>button.addEventListener('click',()=>setView(button.dataset.view)));
document.querySelectorAll('[data-action]').forEach(button=>button.addEventListener('click',()=>manualMove(Number(button.dataset.action))));
$('reset-game').addEventListener('click',()=>setView('play'));
$('checkpoint-slider').addEventListener('input',event=>selectCheckpoint(event.target.value));
$('checkpoint-prev').addEventListener('click',()=>selectCheckpoint(checkpointIndex-1));
$('checkpoint-next').addEventListener('click',()=>selectCheckpoint(checkpointIndex+1));
$('pause').addEventListener('click',()=>{paused=!paused;$('pause').textContent=paused?'▶':'Ⅱ';$('pause').setAttribute('aria-label',paused?'Resume playback':'Pause playback');});
$('export').addEventListener('click',()=>{if(!exportData)return;const url=URL.createObjectURL(new Blob([JSON.stringify(exportData,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=`jev-${exportData.provider}-seed${exportData.config.seed}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
document.addEventListener('keydown',event=>{if(view!=='play'||['INPUT','SELECT','TEXTAREA','BUTTON'].includes(document.activeElement.tagName))return;const keys={ArrowUp:0,w:0,ArrowRight:1,d:1,ArrowDown:2,s:2,ArrowLeft:3,a:3};if(event.key in keys){event.preventDefault();manualMove(keys[event.key]);}if(event.key.toLowerCase()==='r')setView('play');});

function animate(now) {
  if(view!=='play'&&!paused&&frames.length&&now-lastTick>Number($('speed').value)) {
    lastTick=now;
    if(frameIndex>=frames.length+6){frameIndex=0;trail=[];}
    if(frameIndex<frames.length){trail.push(frames[frameIndex]);trail=trail.slice(-30);showFrame(frames[frameIndex],frameIndex);}
    frameIndex++;
  }
  requestAnimationFrame(animate);
}

async function init() {
  renderVerdict(null);draw(manual);chart([]);$('export').disabled=true;
  for(const path of ['static/jev.json','static/demo.json']){try{const response=await fetch(path);if(response.ok){loadResult(await response.json());break;}}catch{ /* Try the next recorded example. */ }}
  try{
    const config=await api('api/config');online=true;gateway=config.gateway;
    $('provider').querySelector('[value=jev]').textContent=`JEV · ${gateway==='openrouter'?'OpenRouter':'TypeSafe direct'}`;
    $('key-state').textContent=config.jev_available?`JEV via ${gateway==='openrouter'?'OpenRouter':'TypeSafe'}`:'JEV key not set';
    $('server-hint').textContent=config.jev_available?`Official JEV ready via ${gateway==='openrouter'?'OpenRouter':'TypeSafe'}. Repeated judgments are cached.`:'Rules demo is ready. For JEV, set OPENROUTER_API_KEY or TYPESAFE_API_KEY in .env and restart.';
  }catch{
    online=false;$('key-state').textContent='Static replay';
    $('server-hint').innerHTML='This is a recorded demo. To train an agent, run locally:<br><code>uv run jev-arcade serve</code>';
  }
  controls();await poll();setInterval(poll,550);requestAnimationFrame(animate);
}
init();
