/* =====================================================================
   ZENIN Observatory — Audit Table Module
   ===================================================================== */

window.buildTable = function() {
  const Z = window.Z;
  const tbody = document.querySelector('#auditTable tbody');
  const frag = document.createDocumentFragment();

  Z.records.forEach((r, i) => {
    const tr = document.createElement('tr');
    tr.id = `row-${i}`;
    tr.onclick = () => window.updateStep(i);

    let cls = 't-hold';
    if (r.action_code ===  1) cls = 't-exec';
    if (r.action_code === -1) cls = 't-flush';

    const oMark = r.is_outlier              ? ' <span class="tag t-out">OTL</span>' : '';
    const vMark = (r.veto_riesgo===0 && !r.is_outlier) ? ' <span class="tag t-flush">VTO</span>' : '';
    const dCls  = r.d_mahal > 3 ? 'tr' : r.d_mahal > 2 ? 'ta' : '';
    const cCls  = r.cvar_t > r.l_max ? 'tr' : '';
    const lCls  = r.lambda_crono < 0.5 ? 'ta' : '';
    const dxCls = Math.abs(r.dx) > 0.5 ? 'ta' : '';

    tr.innerHTML = `
      <td class="mono">#${String(i).padStart(3,'0')}</td>
      <td class="mono">${r.time.toFixed(2)}</td>
      <td style="font-size:.66rem;max-width:110px;overflow:hidden;white-space:nowrap;">${r.regime}</td>
      <td class="mono">${r.x.toFixed(3)}</td>
      <td class="mono ${dxCls}">${r.dx>=0?'+':''}${r.dx.toFixed(4)}</td>
      <td class="mono">${r.v.toFixed(3)}</td>
      <td class="mono">${r.a.toFixed(3)}</td>
      <td class="mono ${dCls}">${r.d_mahal.toFixed(4)}${oMark}</td>
      <td class="mono ${cCls}">${r.cvar_t.toFixed(5)}${vMark}</td>
      <td class="mono ${lCls}">${r.lambda_crono.toFixed(4)}</td>
      <td class="mono">${r.s0.toFixed(4)}</td>
      <td class="mono">${r.s1.toFixed(4)}</td>
      <td class="mono">${r.s2.toFixed(4)}</td>
      <td class="mono">${r.s3.toFixed(4)}</td>
      <td class="mono tg">${r.dest_var.toFixed(5)}</td>
      <td class="mono">${r.liouville_factor.toFixed(4)}</td>
      <td><span class="tag ${cls}">${r.action_code===1?'EXECUTE':r.action_code===-1?'FLUSH':'HOLD'}</span></td>
    `.trim();
    frag.appendChild(tr);
  });
  tbody.appendChild(frag);
};

window.filterTable = function(f, btn) {
  document.querySelectorAll('.fbtn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.querySelectorAll('#auditTable tbody tr').forEach((tr, i) => {
    const r = window.Z.records[i]; let show = true;
    if (f==='EXECUTE'          && r.action_code !== 1)  show = false;
    if (f==='HOLD'             && r.action_code !== 0)  show = false;
    if (f==='EMERGENCY_FLUSH'  && r.action_code !== -1) show = false;
    if (f==='OUTLIER'          && !r.is_outlier)         show = false;
    if (f==='VETO'             && !(r.veto_riesgo===0 && !r.is_outlier)) show = false;
    tr.style.display = show ? '' : 'none';
  });
};
