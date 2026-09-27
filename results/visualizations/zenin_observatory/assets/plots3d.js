/* =====================================================================
   ZENIN Observatory — 3D Plots Module
   Plotly WebGL: Phase-Space Attractor, Stokes S², Decision Surface
   Per-plot independent playback with requestAnimationFrame
   ===================================================================== */

// ── Scene defaults ──────────────────────────────────────────────────
function scn(title) {
  return {
    backgroundcolor:'#06090f',
    gridcolor:'#1e2f44',
    zerolinecolor:'#243050',
    showbackground:true, showgrid:true,
    showspikes:false,
    title:{ text:title, font:{ color:'#7a8fa6', size:9 } },
  };
}
function make3dLayout(xt, yt, zt, eye, extra) {
  return Object.assign({
    paper_bgcolor:'#0e1623',
    font:{ family:"'JetBrains Mono',monospace", color:'#7a8fa6', size:10 },
    margin:{ l:0, r:0, t:20, b:20 },
    scene:{
      xaxis:{ ...scn(xt), tickfont:{ size:8,color:'#4a6280' }, nticks:5 },
      yaxis:{ ...scn(yt), tickfont:{ size:8,color:'#4a6280' }, nticks:5 },
      zaxis:{ ...scn(zt), tickfont:{ size:8,color:'#4a6280' }, nticks:5 },
      camera:{ eye: eye || { x:1.6,y:1.6,z:1.1 } },
      aspectmode:'cube',
      dragmode:'orbit',
    },
    legend:{ font:{size:9}, bgcolor:'rgba(14,22,35,.85)', bordercolor:'#1e2d42', borderwidth:1 },
    hovermode:'closest',
    hoverlabel:{ bgcolor:'#0d1724', bordercolor:'#2a4060', font:{color:'#e8edf5',size:11} },
  }, extra || {});
}

// ── Color mapping for action code ──────────────────────────────────
function actionColor(code) {
  if (code ===  1) return '#10b981';
  if (code === -1) return '#f43f5e';
  return '#f59e0b';
}

// ── Scatter3d factory ───────────────────────────────────────────────
function sc3(idx, xs, ys, zs, col, sz, sym, name, hover) {
  return {
    type:'scatter3d', mode:'markers',
    x:idx.map(i=>xs[i]), y:idx.map(i=>ys[i]), z:idx.map(i=>zs[i]),
    marker:{ color:col, size:sz, symbol:sym||'circle', opacity:.88,
             line:{ width: sz>=7 ? 1.5 : 0, color:'rgba(255,255,255,.25)' } },
    name, hovertemplate: hover || `${name}<br>t=%{customdata:.2f}s<extra></extra>`,
    customdata: idx.map(i=>window.Z.times[i]),
  };
}

// ── Per-plot playback state ─────────────────────────────────────────
const P3 = {
  ph:{ idx:0, playing:false, raf:null, spd:1.0 },
  st:{ idx:0, playing:false, raf:null, spd:1.0 },
  sf:{ idx:0, playing:false, raf:null, spd:1.0 },
};
const ID3 = { ph:'p3d-ph', st:'p3d-st', sf:'p3d-sf' };
// Cursor trace index inside each plot's data array
const CTRACE = { ph:4, st:4, sf:2 };

// ── Auto-rotate state ───────────────────────────────────────────────
const ROT = { ph:0.8, st:0.8, sf:0.8 };
let rotTimers = {};
let globalRot = false;
const presets = {
  iso:  { x:1.6, y:1.6, z:1.1 },
  top:  { x:.01, y:.01, z:2.5 },
  side: { x:.01, y:2.5, z:.3 },
  north:{ x:.01, y:.01, z:2.8 },
  eq:   { x:2.2, y:2.2, z:.1 },
  front:{ x:2.5, y:.01, z:.3 },
};

// ── Build 3D plots ──────────────────────────────────────────────────
window.init3D = function() {
  const Z = window.Z;
  const { times,xs,vels,accs,s1n,s2n,s3n,pds,dvs,dests,
          execIdx,holdIdx,flushIdx,sphereMesh,surfData } = Z;

  // ── A. Phase Space: M³(x, ẋ, ẍ) ─────────────────────────────────
  // Color trajectory by D(t) for richer information
  const phOrbit = {
    type:'scatter3d', mode:'lines',
    x:xs, y:vels, z:accs,
    line:{
      color: Z.dests,           // colour by D(t) value
      colorscale:'Plasma',
      width:3,
      cmin:0, cmax:1,
      colorbar:{ title:'D(t)', len:.5, thickness:8, x:0.0,
                 tickfont:{color:'#7a8fa6',size:8} }
    },
    name:'Orbita M³', hovertemplate:'x=%{x:.3f}<br>ẋ=%{y:.4f}<br>ẍ=%{z:.4f}<extra></extra>',
    showlegend:true,
  };
  const phExec  = sc3(execIdx,  xs,vels,accs, '#10b981',5,'circle',  'EXECUTE', 'EXECUTE t=%{customdata:.2f}s<br>x=%{x:.3f}<extra></extra>');
  const phHold  = sc3(holdIdx,  xs,vels,accs, '#f59e0b',3,'circle',  'HOLD');
  const phFlush = sc3(flushIdx, xs,vels,accs, '#f43f5e',8,'cross',   'FLUSH', 'FLUSH t=%{customdata:.2f}s<br>x=%{x:.3f}<extra></extra>');
  const phCur   = {
    type:'scatter3d', mode:'markers',
    x:[xs[0]], y:[vels[0]], z:[accs[0]],
    marker:{ color:'#38bdf8', size:12, symbol:'diamond',
             line:{ color:'#fff', width:1 } },
    name:'t actual', hovertemplate:'t=%{customdata:.2f}s<extra></extra>',
    customdata:[times[0]],
  };
  Plotly.newPlot(ID3.ph, [phOrbit,phExec,phHold,phFlush,phCur],
    make3dLayout(
      'x(t)  — Estado del sistema',
      'ẋ(t) = dx/dt  — Velocidad',
      'ẍ(t) = d²x/dt²  — Aceleracion',
      presets.iso
    ),
    { responsive:true, displayModeBar:true, displaylogo:false,
      modeBarButtonsToRemove:['toImage'] });

  // ── B. Stokes Sphere S² ──────────────────────────────────────────
  const sphSurf = {
    type:'surface',
    x:sphereMesh.x, y:sphereMesh.y, z:sphereMesh.z,
    opacity:.14,
    colorscale:[[0,'#0d1830'],[0.5,'#1a3050'],[1,'#2a4870']],
    showscale:false, hoverinfo:'none',
    contours:{ x:{show:false}, y:{show:false}, z:{show:false} },
  };
  // Reference circles (latitude lines)
  const circle = (lat, col) => {
    const t = Array.from({length:65},(_,i)=>i*2*Math.PI/64);
    const r = Math.cos(lat);
    return { type:'scatter3d', mode:'lines',
      x:t.map(a=>r*Math.cos(a)), y:t.map(a=>r*Math.sin(a)),
      z:new Array(65).fill(Math.sin(lat)),
      line:{color:col,width:1,dash:'dot'}, showlegend:false, hoverinfo:'none' };
  };
  const sphSpinor = {
    type:'scatter3d', mode:'lines',
    x:s1n, y:s2n, z:s3n,
    line:{
      color: Z.dests,
      colorscale:'Turbo',
      width:5, cmin:0, cmax:1,
      colorbar:{ title:'D(t)', len:.5, thickness:8, x:1.0,
                 tickfont:{color:'#7a8fa6',size:8} }
    },
    name:'Espinor Hopf (S1,S2,S3)/S0',
    hovertemplate:'S1/S0=%{x:.4f}<br>S2/S0=%{y:.4f}<br>S3/S0=%{z:.4f}<extra></extra>',
  };
  const sphPoles = {
    type:'scatter3d', mode:'markers+text',
    x:[0,0], y:[0,0], z:[1.06,-1.06],
    marker:{ size:10, color:['#ec4899','#f59e0b'],
             line:{color:'#fff',width:1} },
    text:['z₁ Rosa Roja','z₂ MRT'],
    textposition:['top center','bottom center'],
    textfont:{ size:9, color:'#94a3b8' },
    name:'Polos Hopf',
  };
  const sphCur = {
    type:'scatter3d', mode:'markers',
    x:[s1n[0]], y:[s2n[0]], z:[s3n[0]],
    marker:{ color:'#f43f5e', size:12, symbol:'circle',
             line:{color:'#fff',width:1.5} },
    name:'t actual',
    hovertemplate:'S1/S0=%{x:.4f}<br>S2/S0=%{y:.4f}<br>S3/S0=%{z:.4f}<extra></extra>',
  };
  const stLay = make3dLayout(
    'S₁/S₀ — Polarizacion lineal',
    'S₂/S₀ — Polarizacion cruzada',
    'S₃/S₀ — Quiralidad / Asimetria',
    { x:1.4, y:1.4, z:0.9 }
  );
  stLay.scene.xaxis.range = [-1.12,1.12];
  stLay.scene.yaxis.range = [-1.12,1.12];
  stLay.scene.zaxis.range = [-1.12,1.12];
  Plotly.newPlot(ID3.st,
    [sphSurf, circle(0,'#334566'), circle(Math.PI/6,'#243050'), circle(-Math.PI/6,'#243050'),
     circle(Math.PI/3,'#1a2840'), sphSpinor, sphPoles, sphCur],
    stLay,
    { responsive:true, displayModeBar:true, displaylogo:false,
      modeBarButtonsToRemove:['toImage'] });

  // ── C. Decision Surface D(θ,∇·F) ────────────────────────────────
  const sfSurf = {
    type:'surface',
    x:surfData.theta, y:surfData.div, z:surfData.z,
    colorscale:'Viridis',
    opacity:.9,
    contours:{
      z:{ show:true, usecolormap:true, highlightcolor:'#fff', project:{z:false}, width:1 }
    },
    colorbar:{
      title:{ text:'D(θ,∇·F)', font:{color:'#7a8fa6',size:9} },
      len:.7, thickness:12,
      tickfont:{color:'#7a8fa6',size:8},
    },
    hovertemplate:'θ=%{x:.3f} rad<br>∇·F=%{y:.3f}<br>D=%{z:.4f}<extra></extra>',
    name:'Superficie D(θ,∇·F)',
  };
  const sfPts = {
    type:'scatter3d', mode:'markers',
    x:pds, y:dvs, z:dests,
    marker:{ size:3.5, color:Z.dests, colorscale:'Plasma',
             cmin:0, cmax:1, opacity:.8 },
    name:'Trayectoria ZENIN',
    hovertemplate:'θ=%{x:.3f}<br>∇·F=%{y:.3f}<br>D=%{z:.4f}<extra></extra>',
  };
  const sfCur = {
    type:'scatter3d', mode:'markers',
    x:[pds[0]], y:[dvs[0]], z:[dests[0]],
    marker:{ color:'#f8f8f8', size:11, symbol:'diamond',
             line:{color:'#f59e0b',width:2} },
    name:'t actual',
  };
  Plotly.newPlot(ID3.sf, [sfSurf, sfPts, sfCur],
    make3dLayout(
      'θ (rad)  — Fase delta entre motores',
      '∇·F  — Divergencia del flujo',
      'D(t)  — Certeza Variable Destino',
      { x:1.7, y:-1.4, z:1.2 }
    ),
    { responsive:true, displayModeBar:true, displaylogo:false,
      modeBarButtonsToRemove:['toImage'] });
};

// ── Per-plot cursor update (no full redraw, only restyle cursor) ────
window.update3dCursor = function(idx) {
  const Z = window.Z;
  const r = Z.records[idx];
  // Batch all 3 restyle calls at once
  requestAnimationFrame(() => {
    try { Plotly.restyle(ID3.ph,{ x:[[r.x]], y:[[r.v]], z:[[r.a]] },[CTRACE.ph]); } catch(e){}
    try { Plotly.restyle(ID3.st,{ x:[[Z.s1n[idx]]], y:[[Z.s2n[idx]]], z:[[Z.s3n[idx]]] },[CTRACE.st]); } catch(e){}
    try { Plotly.restyle(ID3.sf,{ x:[[r.phase_delta]], y:[[r.div_flow]], z:[[r.dest_var]] },[CTRACE.sf]); } catch(e){}
  });
};

// ── Per-plot independent playback ───────────────────────────────────
function plotStep(name, delta) {
  const s = P3[name];
  s.idx = Math.max(0, Math.min(window.Z.N-1, s.idx + delta));
  syncPlotSlider(name, s.idx);
}

function syncPlotSlider(name, idx) {
  const s = P3[name];
  s.idx = idx;
  const r = window.Z.records[idx];
  const el = document.getElementById(`pslider-${name}`);
  if (el) el.value = idx;
  const stepEl = document.getElementById(`pstep-${name}`);
  if (stepEl) stepEl.textContent = `#${String(idx).padStart(3,'0')} / ${window.Z.N}`;
  const regEl = document.getElementById(`pregime-${name}`);
  if (regEl) regEl.textContent = r.regime;
  // Update 3D cursor only for this plot
  requestAnimationFrame(() => {
    if (name === 'ph') {
      try { Plotly.restyle(ID3.ph,{ x:[[r.x]], y:[[r.v]], z:[[r.a]] },[CTRACE.ph]); } catch(e){}
    } else if (name === 'st') {
      try { Plotly.restyle(ID3.st,{ x:[[window.Z.s1n[idx]]], y:[[window.Z.s2n[idx]]], z:[[window.Z.s3n[idx]]] },[CTRACE.st]); } catch(e){}
    } else if (name === 'sf') {
      try { Plotly.restyle(ID3.sf,{ x:[[r.phase_delta]], y:[[r.div_flow]], z:[[r.dest_var]] },[CTRACE.sf]); } catch(e){}
    }
  });
}

window.onPlotSlider = function(name, val) { syncPlotSlider(name, parseInt(val)); };

window.togglePlotPlay = function(name) {
  const s = P3[name];
  s.playing = !s.playing;
  const btn = document.getElementById(`ppbtn-${name}`);
  if (s.playing) {
    btn.textContent = '⏸';
    btn.classList.add('btn-p');
    let last = 0;
    function frame(ts) {
      if (!P3[name].playing) return;
      const interval = Math.max(16, 60 / s.spd);
      if (ts - last >= interval) {
        last = ts;
        const next = (s.idx + 1) >= window.Z.N ? 0 : s.idx + 1;
        syncPlotSlider(name, next);
      }
      P3[name].raf = requestAnimationFrame(frame);
    }
    s.raf = requestAnimationFrame(frame);
  } else {
    btn.textContent = '▶';
    btn.classList.remove('btn-p');
    if (s.raf) { cancelAnimationFrame(s.raf); s.raf = null; }
  }
};

// Called by master playback to sync all independent plot sliders
window.syncAll3dSliders = function(idx) {
  ['ph','st','sf'].forEach(n => syncPlotSlider(n, idx));
};

// ── Camera controls ─────────────────────────────────────────────────
function stopRotate(n) {
  if (rotTimers[n]) { clearInterval(rotTimers[n]); delete rotTimers[n]; }
}
window.toggleRotate = function(n) {
  if (rotTimers[n]) { stopRotate(n); return; }
  rotTimers[n] = setInterval(() => {
    const spd = parseInt(document.getElementById('rotSpd').value) * .0025;
    ROT[n] = (ROT[n] || 0) + spd;
    const radius = 2.1;
    const eye = { x:radius*Math.cos(ROT[n]), y:radius*Math.sin(ROT[n]), z: n==='sf' ? 1.4 : 1.0 };
    try { Plotly.relayout(ID3[n], { 'scene.camera.eye': eye }); } catch(e){}
  }, 50);
};
window.setView = function(n, preset) {
  stopRotate(n);
  try { Plotly.relayout(ID3[n], { 'scene.camera.eye': presets[preset] || presets.iso }); } catch(e){}
};
window.toggleAllRotate = function() {
  globalRot = !globalRot;
  const btn = document.getElementById('btnGrot');
  if (globalRot) {
    btn.textContent = 'Auto-giro: ON'; btn.classList.add('btn-p');
    ['ph','st','sf'].forEach(n => { if (!rotTimers[n]) window.toggleRotate(n); });
  } else {
    btn.textContent = 'Auto-giro: OFF'; btn.classList.remove('btn-p');
    ['ph','st','sf'].forEach(n => stopRotate(n));
  }
};

window.resize3D = function() {
  ['p3d-ph','p3d-st','p3d-sf'].forEach(p => { try{Plotly.Plots.resize(p);}catch(e){} });
};
