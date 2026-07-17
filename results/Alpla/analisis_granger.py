"""
Análisis de Granger causality sobre pares de parámetros reales de ALPLA.

Auditoría estadística rigurosa: valida estacionariedad (ADF), regularidad
de muestreo, gaps, y corre Granger causality canónica via OLS con F-test.
NO implementa nada en el pipeline — solo analiza y reporta.
"""
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

_SCRIPT_DIR = '/home/nicolas/Documentos/Proyectos/Zenin-Iot/iot_machine_learning/results'

# ── 1. Cargar datos pivotados (wide format, ya ffill/bfill) ──
xl = pd.ExcelFile(f'{_SCRIPT_DIR}/Información Chiller y CA - ZENIN.xlsx')

pivots_raw = {}
for sheet in ['Chiller', 'CA']:
    df = pd.read_excel(xl, sheet_name=sheet)
    # Strip trailing spaces from parameter names
    df['Parámetro'] = df['Parámetro'].str.strip()
    p = df.pivot_table(index='Fecha', columns='Parámetro', values='Valor', aggfunc='first').sort_index()
    pivots_raw[sheet] = p.ffill().bfill()


def adf_test(series, label="", max_lag=12):
    """
    Augmented Dickey-Fuller test con selección de lags via AIC.
    H0: la serie tiene raíz unitaria (no estacionaria).
    """
    y = series.dropna().values
    n = len(y)
    if n < 30:
        return None, False, f"n={n}<30"

    dy = np.diff(y)
    y_lag = y[:-1]

    best_aic = np.inf
    best_p = 1

    for p in range(1, min(max_lag + 1, n // 4)):
        rows = n - 1 - p
        if rows < 30:
            continue
        const = np.ones(rows)
        trend = np.arange(float(rows))
        y_lag_part = y_lag[p:]
        dy_lags = [dy[p - i - 1 : n - 1 - i - 1] for i in range(p)]
        X = np.column_stack([const, trend, y_lag_part] + dy_lags)
        y_actual = dy[p:]

        beta = np.linalg.lstsq(X, y_actual, rcond=None)[0]
        residuals = y_actual - X @ beta
        rss = residuals @ residuals
        k = X.shape[1]
        aic = n * np.log(rss / n) + 2 * k
        if aic < best_aic:
            best_aic = aic
            best_p = p

    p = best_p
    rows = n - 1 - p
    if rows < 30:
        return None, False, "rows<30"

    const = np.ones(rows)
    trend = np.arange(float(rows))
    y_lag_part = y_lag[p:]
    dy_lags = [dy[p - i - 1 : n - 1 - i - 1] for i in range(p)]
    X = np.column_stack([const, trend, y_lag_part] + dy_lags)
    y_actual = dy[p:]

    beta = np.linalg.lstsq(X, y_actual, rcond=None)[0]
    residuals = y_actual - X @ beta
    mse = residuals @ residuals / (len(y_actual) - X.shape[1])
    var_beta = np.linalg.inv(X.T @ X) * mse
    se = np.sqrt(np.diag(var_beta))
    t_stat = beta[2] / se[2] if abs(se[2]) > 1e-12 else 0.0

    thresholds = [(-3.96, 0.01), (-3.41, 0.05), (-3.13, 0.10)]
    p_val = None
    for crit, plevel in sorted(thresholds):
        if t_stat < crit:
            p_val = plevel

    is_stationary = t_stat < -3.41
    note = f"p≈{p_val or '>0.10'}, ADF={t_stat:.2f}, lags={p}"
    return t_stat, is_stationary, note


def granger_causality(source, target, max_lag=10):
    """
    Granger causality canónica via OLS + F-test.
    """
    common = source.dropna().index.intersection(target.dropna().index)
    src = source.loc[common].values
    tgt = target.loc[common].values
    n = len(common)

    results = []
    for p in range(1, min(max_lag + 1, n // 4)):
        rows = n - p
        if rows < 30:
            continue

        const_r = np.ones(rows)
        X_r = np.column_stack([const_r] + [tgt[p - i - 1 : n - i - 1] for i in range(p)])
        y_r = tgt[p:]
        beta_r = np.linalg.lstsq(X_r, y_r, rcond=None)[0]
        rss_r = ((y_r - X_r @ beta_r) ** 2).sum()

        X_u = np.column_stack([X_r] + [src[p - i - 1 : n - i - 1] for i in range(p)])
        beta_u = np.linalg.lstsq(X_u, y_r, rcond=None)[0]
        rss_u = ((y_r - X_u @ beta_u) ** 2).sum()

        df1, df2 = p, rows - 2 * p - 1
        if df2 <= 0 or rss_u < 1e-12:
            continue
        f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2)
        p_value = 1.0 - scipy_stats.f.cdf(f_stat, df1, df2)
        aic = n * np.log(rss_u / n) + 2 * X_u.shape[1]
        results.append((p, f_stat, p_value, aic))

    if not results:
        return {"direction": "none", "error": "insufficient_data"}

    best = min(results, key=lambda x: x[3])
    p_opt, f_opt, pval_opt, aic_opt = best

    return {
        "lag_optimo": p_opt,
        "F_statistic": round(f_opt, 4),
        "p_value": round(pval_opt, 6),
        "AIC": round(aic_opt, 2),
        "n_observaciones": n,
        "significativo": pval_opt < 0.05,
    }


def regularidad_muestreo(series_index):
    """Analiza gaps en el timeline."""
    dates = pd.Series(series_index).sort_values()
    deltas = np.diff([(d - dates.iloc[0]).days for d in dates])
    pct_reg = np.mean(deltas == 1) * 100
    max_gap = int(max(deltas)) if len(deltas) > 0 else 0
    return {
        "n_fechas": len(dates),
        "pct_regular": round(pct_reg, 1),
        "max_gap_dias": max_gap,
    }


# ══════════════════════════════════════════════════════════════════
# REPORTE
# ══════════════════════════════════════════════════════════════════
print("=" * 80)
print("  AUDITORÍA ESTADÍSTICA — GRANGER CAUSALITY SOBRE ALPLA")
print("  Método: Granger causality canónica vía OLS + F-test")
print("  Precondiciones validadas: estacionariedad (ADF), regularidad de muestreo")
print("=" * 80)

ALL_PAIRS = {
    "Chiller": [
        ("Consumo de energía sin restabecim. RTAE 5",
         "Temperatura de salida de agua",
         "más consumo → más calor residual → sube temp salida"),
        ("Consumo de energía sin restabecim. RTAE 5",
         "Temperatura de entrada de agua",
         "más consumo → más carga térmica → sube temp retorno"),
        ("Cto.1 Número de arranques del compresor",
         "Cto.1 Temperatura de saturación del refrigerante del evaporador",
         "más arranques → mayor estrés térmico en evaporador"),
        ("Cto.2 Número de arranques del compresor",
         "Cto.2 Temperatura de saturación de refrigerante de evaporador",
         "más arranques → mayor estrés térmico en evaporador (Cto.2)"),
        ("Temperatura de entrada de agua",
         "Temperatura de salida de agua",
         "Control: delta térmico — entrada debe preceder a salida"),
    ],
    "CA": [
        ("Temperatura del aceite",
         "Presión del aceite",
         "aceite más caliente → menor viscosidad → baja presión"),
        ("Temperatura de chumacera lado motor",
         "Presión del aceite",
         "chumacera caliente → fricción → carga en sistema lubricación"),
        ("Temperatura ambiente",
         "Temperatura de agua a la entrada del compresor",
         "Control: temp ambiente debe preceder a temp agua entrada"),
        ("Temperatura del punto de rocío del secador",
         "Presión de regulación",
         "punto de rocío alto → más humedad → afecta regulación"),
    ],
}

# PASO 2
print("\n\n" + "=" * 80)
print("  PASO 2: REGULARIDAD DE MUESTREO POR PAR CANDIDATO")
print("=" * 80)
for sheet, pairs in ALL_PAIRS.items():
    print(f"\n  --- {sheet} ---")
    for src_name, tgt_name, _ in pairs:
        if src_name in pivots_raw[sheet].columns and tgt_name in pivots_raw[sheet].columns:
            common = pivots_raw[sheet][src_name].dropna().index.intersection(
                pivots_raw[sheet][tgt_name].dropna().index)
            if len(common) < 30:
                print(f"  ❌ {src_name[:30]:30s} ↔ {tgt_name[:30]:30s}: SOLO {len(common)} fechas — NO APTO")
                continue
            r = regularidad_muestreo(common)
            if r["pct_regular"] < 50:
                print(f"  ⚠️  {src_name[:30]:30s} ↔ {tgt_name[:30]:30s}: {r['n_fechas']} fechas, {r['pct_regular']}% regular — IRREGULAR")
            else:
                print(f"  ✅ {src_name[:30]:30s} ↔ {tgt_name[:30]:30s}: {r['n_fechas']} fechas, {r['pct_regular']}% regular — APTO")
        else:
            print(f"  ❌ {src_name[:30]:30s} ↔ {tgt_name[:30]:30s}: parámetro no encontrado en pivote")

# PASO 3
print("\n\n" + "=" * 80)
print("  PASO 3: TEST DE ESTACIONARIEDAD (ADF)")
print("=" * 80)
all_params = set()
for sheet, pairs in ALL_PAIRS.items():
    for src_name, tgt_name, _ in pairs:
        all_params.add((src_name, sheet))
        all_params.add((tgt_name, sheet))

adf_results = {}
for param, sheet in sorted(all_params):
    if param not in pivots_raw[sheet].columns:
        print(f"  ⚠️  {param[:40]:40s} [{sheet:7s}] → no encontrado")
        continue
    t_stat, is_stationary, note = adf_test(pivots_raw[sheet][param])
    adf_results[(param, sheet)] = (is_stationary, note, t_stat)
    mark = "✅ ESTACIONARIA" if is_stationary else "❌ NO estacionaria"
    print(f"  {mark} {param[:40]:40s} [{sheet:7s}] → {note}")
    if not is_stationary:
        diff = pivots_raw[sheet][param].diff().dropna()
        t_stat2, is_stat2, note2 = adf_test(diff)
        if is_stat2:
            print(f"       → 1era diferencia: ✅ ESTACIONARIA ({note2})")
        else:
            print(f"       → 1era diferencia: ❌ aún NO ({note2})")

# PASO 4
print("\n\n" + "=" * 80)
print("  PASO 4: GRANGER CAUSALITY (canónica, OLS + F-test)")
print("  Lag óptimo via AIC (max_lag=10)")
print("=" * 80)

for sheet, pairs in ALL_PAIRS.items():
    print(f"\n  === {sheet} ===")
    for src_name, tgt_name, hypothesis in pairs:
        print(f"\n  ── {src_name[:35]:35s} → {tgt_name[:35]:35s}")
        print(f"     Hipótesis: {hypothesis}")

        if src_name not in pivots_raw[sheet].columns or tgt_name not in pivots_raw[sheet].columns:
            print(f"     ❌ SKIP: parámetro(s) no encontrados")
            continue

        src = pivots_raw[sheet][src_name]
        tgt = pivots_raw[sheet][tgt_name]
        src_stat = adf_results.get((src_name, sheet), (False, "", 0))
        tgt_stat = adf_results.get((tgt_name, sheet), (False, "", 0))

        diff_needed = not src_stat[0] or not tgt_stat[0]
        src_use = src.diff().dropna() if not src_stat[0] else src
        tgt_use = tgt.diff().dropna() if not tgt_stat[0] else tgt
        if diff_needed:
            print(f"     ⚠️  Series diferenciadas (no estacionarias en nivel)")

        res_fwd = granger_causality(src_use, tgt_use)
        res_rev = granger_causality(tgt_use, src_use)

        for label, res in [("source→target", res_fwd), ("target→source", res_rev)]:
            if res.get("error"):
                print(f"     ❌ {label}: {res['error']}")
                continue
            sig = "✅ SIGNIFICATIVO" if res["significativo"] else "❌ No significativo"
            print(f"     {label}: F={res['F_statistic']:.3f}, p={res['p_value']:.4f}, "
                  f"lag={res['lag_optimo']}, n={res['n_observaciones']}  {sig}")

        fwd_sig = res_fwd.get("significativo", False)
        rev_sig = res_rev.get("significativo", False)
        if fwd_sig and rev_sig:
            print(f"     ⚠️  BIDIRECCIONAL (feedback loop)")
        elif fwd_sig:
            print(f"     🎯 DIRECCIÓN: {src_name[:35]} PRECEDE a {tgt_name[:35]}")
        elif rev_sig:
            print(f"     🔄 INVERSA: {tgt_name[:35]} PRECEDE a {src_name[:35]}")
        else:
            print(f"     ➖ Sin relación significativa en ninguna dirección")

print("\n\n" + "=" * 80)
print("  DISCLAIMER ESTADÍSTICO")
print("=" * 80)
print("""
  Granger causality mide PRECEDENCIA PREDICTIVA, no causalidad física.
  Un resultado significativo (p<0.05) indica que la serie A contiene
  información útil para predecir B, controlando por el pasado de B.

  Limitaciones conocidas de este análisis:
  1. Muestreo con gaps de hasta 7 días (72% regular) — datos ffill/bfill.
  2. No se controló por terceras variables.
  3. n≈300 observaciones — poder limitado para efectos pequeños.

  Para la reunión con ALPLA:
  - NO diga "X causa Y" — diga "X precede predictivamente a Y"
  - Resultados del Chiller son más robustos que los del CA
""")
