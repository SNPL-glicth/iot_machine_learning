/* =====================================================================
   ZENIN Observatory — Master Playback & HUD Module
   requestAnimationFrame loop — no setInterval lag
   Tab-aware updates: only refresh visible tab's plots
   ===================================================================== */

let masterIdx   = 0;
let isPlaying   = false;
let masterRaf   = null;
let masterSpeed = 1.0;
let lastFrameTs = 0;

// ── Compute active tab ──────────────────────────────────────────────
function activeTab() {
  const el = document.querySelector('.tpane.active');
  return el ? el.id.replace('pane-','') : 'geo3d';
}

// ── HUD update ──────────────────────────────────────────────────────
window.updateStep = function(idx) {
  masterIdx = Math.max(0, Math.min(window.Z.N-1, idx|0));
  const r = window.Z.records[masterIdx];
  const t = window.Z.times[masterIdx];

  // Slider + label
  document.getElementById('timeSlider').value = masterIdx;
  document.getElementById('timeLabel').textContent = `t = ${t.toFixed(2)} s`;

  // HUD cells
  document.getElementById('hudStep').textContent   = `#${String(masterIdx).padStart(3,'0')}`;
  document.getElementById('hudRegime').textContent  = r.regime;
  document.getElementById('hudX').textContent       = r.x.toFixed(4);
  document.getElementById('hudDx').textContent      = `dx = ${r.dx>=0?'+':''}${r.dx.toFixed(5)}`;
  document.getElementById('hudVA').textContent      = `v:${r.v.toFixed(3)}  a:${r.a.toFixed(3)}`;

  const mEl = document.getElementById('hudMahal');
  mEl.textContent = r.d_mahal.toFixed(4);
  mEl.className   = `hv ${r.is_outlier?'tr':r.d_mahal>2?'ta':'tg'}`;
  document.getElementById('hudMS').textContent = r.is_outlier ? 'BLOQUEADO — Outlier' : 'Normal (tau=3.0)';

  document.getElementById('hudCvar').textContent = `${r.cvar_t.toFixed(5)} / ${r.l_max.toFixed(3)}`;
  document.getElementById('hudCS').textContent   = r.veto_riesgo===0 ? 'VETO activo — riesgo alto' : 'Sin veto';
  document.getElementById('hudLambda').textContent = r.lambda_crono.toFixed(4);
  document.getElementById('hudSt').textContent    = `${r.s0.toFixed(3)} / ${r.s1.toFixed(3)} / ${r.s3.toFixed(3)}`;
  document.getElementById('hudS2').textContent    = `S₂ = ${r.s2.toFixed(5)}`;
  document.getElementById('hudD').textContent     = r.dest_var.toFixed(5);
  document.getElementById('hudLv').textContent    = `Liouville = ${r.liouville_factor.toFixed(4)}`;

  const aEl = document.getElementById('hudAction');
  const rEl = document.getElementById('hudAR');
  if (r.action_code ===  1) { aEl.innerHTML='<span class="b-exec">EXECUTE</span>';       rEl.textContent='Alta certeza — todo verde'; }
  else if (r.action_code===-1){ aEl.innerHTML='<span class="b-flush">EMERGENCY_FLUSH</span>'; rEl.textContent='Zona de caida libre'; }
  else {                        aEl.innerHTML='<span class="b-hold">HOLD</span>';
    rEl.textContent = r.is_outlier?'Outlier detectado':r.veto_riesgo===0?'Veto CVaR':'Certeza insuficiente'; }

  // Sync 3D per-plot sliders
  if (typeof window.syncAll3dSliders === 'function') window.syncAll3dSliders(masterIdx);

  // Tab-aware 3D / 2D cursor
  const tab = activeTab();
  if (tab === 'geo3d' && typeof window.update3dCursor === 'function') {
    window.update3dCursor(masterIdx);
  } else if (tab === 'signals' && typeof window.update2dCursor === 'function') {
    window.update2dCursor(masterIdx);
  }

  // Table highlight (lightweight: only when bitacora tab is active)
  if (tab === 'bitacora') {
    document.querySelectorAll('#auditTable tbody tr').forEach(tr => tr.classList.remove('ractive'));
    const row = document.getElementById(`row-${masterIdx}`);
    if (row) { row.classList.add('ractive'); row.scrollIntoView({block:'nearest'}); }
  }
};

// ── RAF-based master playback (no setInterval) ──────────────────────
function masterLoop(ts) {
  if (!isPlaying) return;
  const msBetween = Math.max(16, 60 / masterSpeed);
  if (ts - lastFrameTs >= msBetween) {
    lastFrameTs = ts;
    const next = (masterIdx + 1) >= window.Z.N ? 0 : masterIdx + 1;
    window.updateStep(next);
  }
  masterRaf = requestAnimationFrame(masterLoop);
}

window.togglePlayback = function() {
  isPlaying = !isPlaying;
  const btn = document.getElementById('btnPlay');
  if (isPlaying) {
    btn.textContent = '⏸ Pause'; btn.classList.add('btn-p');
    lastFrameTs = 0;
    masterRaf = requestAnimationFrame(masterLoop);
  } else {
    btn.innerHTML = '&#9654; Play'; btn.classList.remove('btn-p');
    if (masterRaf) { cancelAnimationFrame(masterRaf); masterRaf = null; }
  }
};

window.changeSpeed = function(v) {
  masterSpeed = parseFloat(v);
  // Restart loop if playing to pick up new speed
  if (isPlaying) { cancelAnimationFrame(masterRaf); lastFrameTs=0; masterRaf=requestAnimationFrame(masterLoop); }
};

window.onSlider = function(v)   { window.updateStep(parseInt(v)); };
window.stepSim  = function(d)   { window.updateStep(masterIdx + d); };

// ── Tab switching ────────────────────────────────────────────────────
window.switchTab = function(id) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tpane').forEach(p  => p.classList.remove('active'));
  document.getElementById('pane-'+id).classList.add('active');
  const btn = Array.from(document.querySelectorAll('.tab-btn')).find(b=>b.getAttribute('onclick').includes(id));
  if (btn) btn.classList.add('active');

  // Force-update cursor on newly visible tab
  setTimeout(() => {
    if (id === 'geo3d')   { window.resize3D(); window.update3dCursor(masterIdx); }
    if (id === 'signals') { window.resize2D(); window.update2dCursor(masterIdx); }
  }, 80);
};

window.addEventListener('resize', () => {
  window.resize3D();
  window.resize2D();
});
