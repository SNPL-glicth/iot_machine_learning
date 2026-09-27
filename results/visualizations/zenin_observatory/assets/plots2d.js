/* =====================================================================
   ZENIN Observatory — 2D Plots Module
   All 6 synchronized diagnostic panels with RAF-based cursor
   No lag: cursor updates only on the active tab
   ===================================================================== */

const DB = {
  paper_bgcolor:'#0e1623', plot_bgcolor:'#070b14',
  font:{ family:"'JetBrains Mono',monospace", color:'#7a8fa6', size:10 },
  margin:{ l:54, r:16, t:28, b:42 },
  xaxis:{
    gridcolor:'#1a2740', zerolinecolor:'#243050',
    tickcolor:'#3a5070', linecolor:'#1e2d42', tickfont:{size:9},
    title:{ font:{color:'#94a3b8',size:10}, text:'Tiempo (s)' },
  },
  yaxis:{
    gridcolor:'#1a2740', zerolinecolor:'#243050',
    tickcolor:'#3a5070', linecolor:'#1e2d42', tickfont:{size:9},
  },
  legend:{ orientation:'h', y:1.14, x:0, font:{size:9}, bgcolor:'transparent' },
  hovermode:'x unified',
  hoverlabel:{ bgcolor:'#0d1724', bordercolor:'#2a4060', font:{color:'#e8edf5',size:11} },
  selectdirection:'h',
};

function regimeShapes() {
  const colors = [
    'rgba(56,189,248,.055)', 'rgba(168,85,247,.075)',
    'rgba(244,63,94,.085)',  'rgba(245,158,11,.065)',
    'rgba(16,185,129,.055)',
  ];
  return window.Z.regimeBands.map((b,i) => ({
    type:'rect', xref:'x', yref:'paper',
    x0:b.x0, x1:b.x1, y0:0, y1:1,
    fillcolor:colors[i], line:{width:0}, layer:'below',
  }));
}

function regimeAnnots() {
  return window.Z.regimeBands.map(b => ({
    xref:'x', yref:'paper',
    x:(b.x0+b.x1)/2, y:0.98,
    text:b.label, showarrow:false,
    font:{size:7.5, color:'rgba(100,140,180,.7)'},
    xanchor:'center', yanchor:'top',
  }));
}

function mkCursor(yMin, yMax) {
  return {
    x:[window.Z.times[0], window.Z.times[0]],
    y:[yMin, yMax],
    mode:'lines',
    line:{ color:'#38bdf8', width:1.8, dash:'dot' },
    hoverinfo:'none', showlegend:false, name:'t',
  };
}

function baseLayout(ytitle) {
  return {
    ...DB,
    yaxis:{ ...DB.yaxis, title:{ font:{color:'#94a3b8',size:10}, text:ytitle } },
    shapes: regimeShapes(),
    annotations: regimeAnnots(),
  };
}

// Cursor trace indices (last trace in each plot's data array)
const CSLOT = { 'p2d-1':5, 'p2d-2':3, 'p2d-3':6, 'p2d-4':3, 'p2d-5':5, 'p2d-6':5 };

window.init2D = function() {
  const Z = window.Z;
  const { times,xs,vels,accs,dMs,rets,rTs,rTsN,cvars,lams,phis,
          s0arr,s1arr,s2arr,s3arr,dests,liovs,cSovs,
          execIdx,holdIdx,flushIdx,outIdx,vetoIdx, N } = Z;

  const sh = regimeShapes(), an = regimeAnnots();
  const tEnd = times[N-1];

  // ── P1: Trayectoria x(t) ─────────────────────────────────────────
  const yMin1 = Math.min(...xs)-4, yMax1 = Math.max(...xs)+4;
  Plotly.newPlot('p2d-1', [
    { x:times, y:xs, mode:'lines', line:{color:'#e2e8f0',width:1.8},
      name:'x(t)', hovertemplate:'t=%{x:.2f}s<br>x=%{y:.4f}<extra></extra>' },
    { x:execIdx.map(i=>times[i]), y:execIdx.map(i=>xs[i]), mode:'markers',
      marker:{color:'#10b981',size:6,opacity:.9}, name:'EXECUTE',
      hovertemplate:'EXECUTE t=%{x:.2f}s<br>x=%{y:.4f}<extra></extra>' },
    { x:holdIdx.map(i=>times[i]),  y:holdIdx.map(i=>xs[i]),  mode:'markers',
      marker:{color:'#f59e0b',size:4,opacity:.65}, name:'HOLD' },
    { x:flushIdx.map(i=>times[i]), y:flushIdx.map(i=>xs[i]), mode:'markers',
      marker:{color:'#f43f5e',size:11,symbol:'x',opacity:1}, name:'FLUSH',
      hovertemplate:'FLUSH t=%{x:.2f}s<extra></extra>' },
    { x:outIdx.map(i=>times[i]),   y:outIdx.map(i=>xs[i]),   mode:'markers',
      marker:{color:'#a855f7',size:8,symbol:'diamond',opacity:.9}, name:'Outlier bloqueado' },
    mkCursor(yMin1, yMax1),
  ], { ...baseLayout('Estado del sistema x(t)'),
       yaxis:{ ...DB.yaxis, title:{text:'x(t)',font:{color:'#94a3b8',size:10}} } },
  { responsive:true, displaylogo:false, modeBarButtonsToRemove:['toImage'] });

  // ── P2: Distancia Mahalanobis ─────────────────────────────────────
  const yMax2 = Math.max(...dMs, 4.5) + 0.4;
  Plotly.newPlot('p2d-2', [
    { x:times, y:dMs, mode:'lines', line:{color:'#38bdf8',width:1.6},
      fill:'tozeroy', fillcolor:'rgba(56,189,248,.05)', name:'d Mahal(t)',
      hovertemplate:'t=%{x:.2f}s<br>d=%{y:.5f}<extra></extra>' },
    { x:[0,tEnd], y:[3,3], mode:'lines', line:{color:'#f43f5e',width:2,dash:'dash'},
      name:'Umbral τ = 3.0' },
    { x:[0,tEnd], y:[2,2], mode:'lines', line:{color:'#f59e0b',width:1,dash:'dot'},
      name:'Zona alerta (d=2)', showlegend:true },
    mkCursor(0, yMax2),
  ], { ...baseLayout('Distancia de Mahalanobis d(t)') },
  { responsive:true, displaylogo:false });

  // ── P3: Riesgo CVaR ───────────────────────────────────────────────
  const yMin3 = Math.min(...rTsN,...rets)*1.3, yMax3 = Math.max(...cvars,...rTs)*1.4;
  Plotly.newPlot('p2d-3', [
    { x:times, y:rTs,  mode:'lines', line:{color:'#38bdf8',width:.6}, showlegend:false, hoverinfo:'none' },
    { x:times, y:rTsN, mode:'lines', line:{color:'#38bdf8',width:.6},
      fill:'tonexty', fillcolor:'rgba(56,189,248,.10)', name:'Tubo ±R_t' },
    { x:times, y:rets, mode:'lines', line:{color:'#94a3b8',width:1.4},
      name:'Retorno r(t)', hovertemplate:'t=%{x:.2f}s<br>r=%{y:.6f}<extra></extra>' },
    { x:times, y:cvars, mode:'lines', line:{color:'#f59e0b',width:2.2},
      name:'CVaR(t) Student-t', hovertemplate:'t=%{x:.2f}s<br>CVaR=%{y:.6f}<extra></extra>' },
    { x:[0,tEnd], y:[.02,.02], mode:'lines', line:{color:'#f43f5e',width:2,dash:'dot'},
      name:'L_max = 0.020' },
    { x:vetoIdx.map(i=>times[i]), y:vetoIdx.map(i=>cvars[i]), mode:'markers',
      marker:{color:'#f43f5e',size:9,symbol:'x',line:{width:2,color:'#f43f5e'}},
      name:'Veto CVaR activo' },
    mkCursor(yMin3, yMax3),
  ], { ...baseLayout('Escala de retorno') },
  { responsive:true, displaylogo:false });

  // ── P4: Lambda + MoE ─────────────────────────────────────────────
  Plotly.newPlot('p2d-4', [
    { x:times, y:lams, mode:'lines', line:{color:'#a855f7',width:2.2},
      fill:'tozeroy', fillcolor:'rgba(168,85,247,.055)', name:'Lambda(t) — Sincronia cronica',
      hovertemplate:'t=%{x:.2f}s<br>Λ=%{y:.5f}<extra></extra>' },
    { x:times, y:phis, mode:'lines', line:{color:'#34d399',width:1.8,dash:'dashdot'},
      name:'Φ_MoE — Consenso jurado', hovertemplate:'t=%{x:.2f}s<br>Φ=%{y:.5f}<extra></extra>' },
    { x:[0,tEnd], y:[.6,.6], mode:'lines', line:{color:'#f59e0b',width:1.5,dash:'dot'},
      name:'Umbral MoE = 0.60' },
    mkCursor(0, 1.05),
  ], { ...baseLayout('Sincronia / Consenso [0..1]'),
       yaxis:{ ...DB.yaxis, title:{text:'Indice [0..1]',font:{color:'#94a3b8',size:10}}, range:[0,1.12] } },
  { responsive:true, displaylogo:false });

  // ── P5: Stokes S0, S1, S2, S3 ────────────────────────────────────
  const yMin5 = Math.min(...s3arr)-.12, yMax5 = Math.max(...s0arr)+.08;
  Plotly.newPlot('p2d-5', [
    { x:times, y:s0arr, mode:'lines', line:{color:'#f1f5f9',width:2.0},
      name:'S₀  (intensidad total)', hovertemplate:'S0=%{y:.6f}<extra></extra>' },
    { x:times, y:s1arr, mode:'lines', line:{color:'#38bdf8',width:1.8},
      name:'S₁  (polarizacion lineal)', hovertemplate:'S1=%{y:.6f}<extra></extra>' },
    { x:times, y:s2arr, mode:'lines', line:{color:'#34d399',width:1.4,dash:'dot'},
      name:'S₂  (polarizacion cruzada)', hovertemplate:'S2=%{y:.6f}<extra></extra>' },
    { x:times, y:s3arr, mode:'lines', line:{color:'#ec4899',width:2.0},
      name:'S₃  (quiralidad / asimetria)', hovertemplate:'S3=%{y:.6f}<extra></extra>' },
    { x:[0,tEnd], y:[0,0], mode:'lines', line:{color:'#374151',width:.8}, showlegend:false, hoverinfo:'none' },
    mkCursor(yMin5, yMax5),
  ], { ...baseLayout('Parametros de Stokes') },
  { responsive:true, displaylogo:false });

  // ── P6: D(t) + Liouville ─────────────────────────────────────────
  const yMax6 = Math.max(...dests,...liovs)*1.18;
  Plotly.newPlot('p2d-6', [
    { x:times, y:dests, mode:'lines', line:{color:'#10b981',width:2.6},
      fill:'tozeroy', fillcolor:'rgba(16,185,129,.055)', name:'D(t) — Variable Destino',
      hovertemplate:'t=%{x:.2f}s<br>D=%{y:.6f}<extra></extra>' },
    { x:times, y:liovs, mode:'lines', line:{color:'#f59e0b',width:1.8,dash:'dash'},
      name:'Factor Liouville exp(-div F)', hovertemplate:'Liouv=%{y:.6f}<extra></extra>' },
    { x:times, y:cSovs, mode:'lines', line:{color:'#60a5fa',width:1.5,dash:'dashdot'},
      name:'|S_sov| — Acople coherente' },
    { x:[0,tEnd], y:[.35,.35], mode:'lines', line:{color:'#f43f5e',width:2,dash:'dot'},
      name:'Umbral EXECUTE D = 0.35' },
    { x:execIdx.map(i=>times[i]), y:execIdx.map(i=>dests[i]), mode:'markers',
      marker:{color:'#10b981',size:5,opacity:.85}, name:'Puntos EXECUTE autorizados' },
    mkCursor(0, yMax6),
  ], { ...baseLayout('Certeza Consolidada D(t)') },
  { responsive:true, displaylogo:false });
};

// ── Cursor update — lag-free with RAF throttle ────────────────────
let _cursorRaf = null;
let _pendingCursorT = null;

window.update2dCursor = function(idx) {
  _pendingCursorT = window.Z.times[idx];
  if (!_cursorRaf) {
    _cursorRaf = requestAnimationFrame(() => {
      _cursorRaf = null;
      if (_pendingCursorT === null) return;
      const tv = [_pendingCursorT, _pendingCursorT];
      Object.entries(CSLOT).forEach(([pid, ti]) => {
        try { Plotly.restyle(pid, { x:[tv] }, [ti]); } catch(e){}
      });
      _pendingCursorT = null;
    });
  }
};

window.resize2D = function() {
  for (let i=1; i<=6; i++) { try{Plotly.Plots.resize('p2d-'+i);}catch(e){} }
};
