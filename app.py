import io, contextlib
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
import streamlit as st
from matplotlib.patches import Ellipse
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage
from skbio.stats.distance import permanova, permdisp, DistanceMatrix
from skbio.stats.ordination import pcoa, rda
from adjustText import adjust_text
from grupos_columnas import columnas_grupos

st.set_page_config(page_title="Greener / BioRem — Analysis", layout="wide")
st.title("Soil microbiome analysis — Greener / BioRem")
st.caption(
    "Dataset: `datos_combinados.csv` — 54 samples × 664+ variables. "
    "6 treatments (BA, BS, CT, VCBA, VCBS, VCT) × 4 time points (days 2, 15, 60, 91) × 3 replicates. "
    "⚠️ `VCBS_15` and `VCBA_15` have no 16S data — they appear as gaps in bar plots and heatmaps."
)

# ── Data ──────────────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    df = pd.read_csv("datos_combinados.csv")
    cols_known   = set(sum(columnas_grupos.values(), []))
    cols_paprica = [c for c in df.columns if c not in cols_known]
    fam_cols = columnas_grupos["Datos brutos por familia"]
    gen_cols = columnas_grupos["Datos brutos por género"]
    fq_cols  = columnas_grupos["Físico-químicos"]
    return df, fam_cols, gen_cols, fq_cols, cols_paprica

df, fam_cols, gen_cols, fq_cols, cols_paprica = load_data()

levels = {"Family": fam_cols, "Genus": gen_cols}

ORDER = [
    "CT_2", "VCT_2",
    "BS_2", "BS_15", "BS_60", "BS_91",
    "VCBS_2", "VCBS_15", "VCBS_60", "VCBS_91",
    "BA_2", "BA_15", "BA_60", "BA_91",
    "VCBA_2", "VCBA_15", "VCBA_60", "VCBA_91",
]

# ── Shared helpers ────────────────────────────────────────────────────────────

def _samples(d):
    return d["Tratamiento"].astype(str) + "_" + d["Dia"].astype(str) + "_" + d["Replica"].astype(str)

def _palette(labels):
    labs = sorted(set(labels))
    return {l: c for l, c in zip(labs, plt.cm.tab20(np.linspace(0, 1, len(labs))))}

def _ellipse(ax, pts, color):
    if len(pts) < 3:
        return
    cov = np.cov(pts.T)
    mu  = pts.mean(axis=0)
    vals, vecs = np.linalg.eigh(cov)
    o = vals.argsort()[::-1]
    vals, vecs = vals[o], vecs[:, o]
    ang = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    ax.add_patch(Ellipse(
        mu, 2 * 2 * np.sqrt(vals[0]), 2 * 2 * np.sqrt(vals[1]),
        angle=ang, color=color, alpha=0.10, linewidth=0.8, linestyle="--", fill=True,
    ))

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Bar plots", "🔥 Heatmap", "🗺️ PCoA", "🔍 Forward selection", "➡️ RDA",
])

# ─────────────────────────────────────────────────────────────────────────────
# 1. Bar plots
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Relative abundance bar plots")
    st.write(
        "Each bar shows the fraction of the microbial community for each taxon in a given "
        "treatment × day combination (replicate means). Taxa below the threshold in every "
        "sample are grouped into **Other** (grey)."
    )
    ctrl, plot_area = st.columns([1, 4])
    with ctrl:
        bp_level = st.selectbox("Level", ["Family", "Genus"], key="bp_level")
        bp_thr   = st.slider("Threshold", 0.005, 0.10, 0.02, 0.005,
                              format="%.3f", key="bp_thr")

    cols = levels[bp_level]
    base = (
        df.dropna(subset=cols)
          .assign(td=lambda d: d["Tratamiento"].astype(str) + "_" + d["Dia"].astype(str))
          .groupby("td")[cols].mean()
    )
    rel   = base.div(base.sum(axis=1), axis=0)
    keep  = rel.columns[rel.max(axis=0) >= bp_thr]
    rel_r = rel[keep].copy()
    rel_r[f"Other (<{int(bp_thr * 100)}%)"] = rel[rel.columns.difference(keep)].sum(axis=1)
    rel_r = rel_r.reindex(ORDER)
    colors = list(sns.husl_palette(rel_r.shape[1] - 1, s=0.7, l=0.6)) + [(0.65, 0.65, 0.65)]
    fig_bp, ax_bp = plt.subplots(figsize=(14, 5))
    rel_r.plot(kind="bar", stacked=True, ax=ax_bp, color=colors, width=0.85,
               edgecolor="black", linewidth=0.3)
    ax_bp.set(title=f"Relative abundance — {bp_level}",
              xlabel="Treatment + Day", ylabel="Relative abundance")
    ax_bp.legend(title=bp_level, bbox_to_anchor=(1.01, 1), loc="upper left",
                 ncol=2, fontsize="small")
    plt.tight_layout()
    with plot_area:
        st.pyplot(fig_bp)
    plt.close(fig_bp)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Heatmap
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Heatmap + dendrograms (taxa × samples)")
    st.write(
        "Compact view of the full abundance matrix. "
        "Colour = relative abundance normalised per sample (red = above mean, blue = below). "
        "The **top dendrogram** can use 16S composition or the PAPRICA functional profile."
    )
    ctrl2, plot_area2 = st.columns([1, 4])
    with ctrl2:
        hm_level   = st.selectbox("Level", ["Family", "Genus"], key="hm_level")
        hm_thr     = st.slider("Threshold", 0.02, 0.10, 0.05, 0.005,
                                format="%.3f", key="hm_thr")
        hm_paprica = st.checkbox("Top tree: PAPRICA", value=True, key="hm_paprica")

    @st.cache_data
    def _heatmap_data(level, threshold, paprica):
        cols   = levels[level]
        td_grp = (
            df.dropna(subset=cols)
              .assign(td=lambda d: d["Tratamiento"].astype(str) + "_" + d["Dia"].astype(str))
              .groupby("td")[cols].mean()
        )
        rel    = td_grp.div(td_grp.sum(axis=1), axis=0)
        td_grp = td_grp[rel.columns[rel.max(axis=0) >= threshold]]
        mat    = td_grp.T.div(td_grp.T.sum(axis=0), axis=1)
        col_lnk = None
        if paprica and cols_paprica:
            pap = (
                df.assign(td=lambda d: d["Tratamiento"].astype(str) + "_" + d["Dia"].astype(str))
                  .groupby("td")[cols_paprica].mean()
                  .reindex(mat.columns).dropna()
            )
            mat     = mat[pap.index]
            col_lnk = linkage(pdist(pap.values, metric="braycurtis"), method="average")
        return mat, col_lnk, td_grp.shape[1]

    mat_hm, col_lnk_hm, n_taxa_hm = _heatmap_data(hm_level, hm_thr, hm_paprica)
    with ctrl2:
        st.caption(f"Taxa shown: **{n_taxa_hm}**")

    g_hm = sns.clustermap(
        mat_hm, method="average", metric="braycurtis", col_linkage=col_lnk_hm,
        cmap="RdBu_r", center=float(mat_hm.values.mean()),
        figsize=(12, 16), dendrogram_ratio=(0.12, 0.10),
        cbar_pos=(0.02, 0.83, 0.03, 0.13), yticklabels=True,
    )
    g_hm.ax_heatmap.tick_params(axis="y", labelsize=6)
    with plot_area2:
        st.pyplot(g_hm.fig)
    plt.close("all")

# ─────────────────────────────────────────────────────────────────────────────
# 3. PCoA + PERMANOVA + PERMDISP
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("PCoA + PERMANOVA + PERMDISP")
    st.write(
        "Projects Bray–Curtis distances between samples onto 2D. "
        "**PERMANOVA** tests whether groups differ in composition more than expected by chance "
        "(R² = fraction of total variation explained). "
        "**PERMDISP** checks that differences reflect composition, not just within-group spread. "
        "Results are cached — only recomputed when level, grouping, or metric changes."
    )
    ctrl3, plot_area3 = st.columns([1, 3])
    with ctrl3:
        pc_level   = st.selectbox("Level", ["Family", "Genus"], key="pc_level")
        pc_groupby = st.selectbox("Group by",
                                   ["Treatment", "Day", "Treatment+Day"], key="pc_groupby")
        pc_metric  = st.selectbox("Metric",
                                   ["braycurtis", "jaccard", "euclidean"], key="pc_metric")
        pc_ell     = st.checkbox("Ellipses", value=False, key="pc_ell")
        pc_labels  = st.checkbox("Sample labels", value=True, key="pc_labels")

    @st.cache_data(show_spinner="Running PERMANOVA (999 permutations)…")
    def _pcoa_stats(level, group_by, metric):
        cols = levels[level]
        d    = df.dropna(subset=cols).copy()
        d["sample"] = _samples(d)
        d["label"]  = d["Tratamiento"].astype(str) + "_" + d["Dia"].astype(str)
        if group_by == "Treatment+Day":
            g = d["Tratamiento"].astype(str) + "_" + d["Dia"].astype(str)
        elif group_by == "Day":
            g = d["Dia"].astype(str)
        else:
            g = d["Tratamiento"].astype(str)
        rel = d[cols].div(d[cols].sum(axis=1), axis=0)
        dm  = DistanceMatrix(squareform(pdist(rel.values, metric=metric)),
                              ids=d["sample"].tolist())
        res = pcoa(dm, number_of_dimensions=2)
        xy  = res.samples.iloc[:, :2].values
        pe  = res.proportion_explained.values[:2]
        pm  = permanova(dm, g.values, permutations=999)
        pd_ = permdisp(dm, g.values, permutations=999, test="centroid")
        F   = float(pm["test statistic"])
        k   = int(pm["number of groups"])
        n   = int(pm["sample size"])
        R2  = F * (k - 1) / (F * (k - 1) + (n - k))
        return (xy, pe, F, R2, float(pm["p-value"]),
                float(pd_["test statistic"]), float(pd_["p-value"]),
                g.values.tolist(), d["label"].values.tolist())

    xy_pc, pe_pc, F_pc, R2_pc, pm_p, pdF_pc, pdp_pc, g_pc, lbl_pc = _pcoa_stats(
        pc_level, pc_groupby, pc_metric
    )
    g_pc   = np.array(g_pc)
    lbl_pc = np.array(lbl_pc)

    cmap_pc = _palette(g_pc)
    fig_pc, ax_pc = plt.subplots(figsize=(9, 7))
    for lab in sorted(set(g_pc)):
        m   = g_pc == lab
        pts = xy_pc[m]
        c   = cmap_pc[lab]
        ax_pc.scatter(pts[:, 0], pts[:, 1], s=55, color=c, edgecolor="white",
                      linewidth=0.6, zorder=3, label=lab)
        if pc_ell:
            _ellipse(ax_pc, pts, c)
    if pc_labels:
        texts = []
        for lbl in sorted(set(lbl_pc)):
            mask = lbl_pc == lbl
            cx, cy = xy_pc[mask].mean(axis=0)
            grp = g_pc[mask][0]
            texts.append(ax_pc.text(cx, cy, lbl, fontsize=7.5, color=cmap_pc[grp],
                                     fontweight="bold", ha="center", va="center"))
        fig_pc.canvas.draw()
        with contextlib.redirect_stdout(io.StringIO()):
            adjust_text(texts, ax=ax_pc, arrowprops=dict(arrowstyle="-", color="#aaa", lw=0.6))
    ax_pc.axhline(0, color="gray", lw=0.5, ls="--")
    ax_pc.axvline(0, color="gray", lw=0.5, ls="--")
    ax_pc.set_xlabel(f"PCo1 ({pe_pc[0]*100:.1f}%)")
    ax_pc.set_ylabel(f"PCo2 ({pe_pc[1]*100:.1f}%)")
    ax_pc.set_title(f"PCoA ({pc_level}, {pc_metric})", fontsize=13, fontweight="bold")
    ax_pc.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize="small")
    disp_sig = pdp_pc < 0.05
    note  = "WARNING: heterogeneous dispersion" if disp_sig else "OK: homogeneous dispersion"
    stats = (f"PERMANOVA ({pc_groupby}): F={F_pc:.2f}, R²={R2_pc:.3f}, p={pm_p:.3f}"
             f"   |   PERMDISP: F={pdF_pc:.2f}, p={pdp_pc:.3f}   {note}")
    ax_pc.text(0.5, -0.10, stats, transform=ax_pc.transAxes, ha="center", fontsize=8.5,
               color="#c0392b" if disp_sig else "#27ae60",
               bbox=dict(boxstyle="round,pad=0.3",
                         facecolor="#fdf3f0" if disp_sig else "#f0fdf4",
                         edgecolor="#c0392b" if disp_sig else "#27ae60", alpha=0.8))
    plt.tight_layout()
    with plot_area3:
        st.pyplot(fig_pc)
    plt.close(fig_pc)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Forward selection
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Forward selection of physicochemical variables")
    st.info(
        "Identifies the physicochemical variables that explain microbial composition "
        "**significantly and independently** (equivalent to `ordiR2step` in R/vegan). "
        "Run this before the RDA. Results are stored in the session and pre-loaded in the RDA tab."
    )
    st.warning(
        "⏱️ This can take **several minutes** (each step evaluates ~14 candidates × n full RDA permutations)."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        fs_level  = st.selectbox("Level", ["Family", "Genus"], key="fs_level")
        fs_thr    = st.slider("Taxon threshold", 0.01, 0.10, 0.02, 0.01,
                               format="%.2f", key="fs_thr")
    with col_b:
        fs_alpha  = st.slider("Significance α", 0.01, 0.10, 0.05, 0.01,
                               format="%.2f", key="fs_alpha")
        fs_nperm  = st.select_slider("Permutations",
                                      options=[99, 199, 299, 499, 999], value=199, key="fs_nperm")

    if st.button("▶ Run forward selection", type="primary"):
        cols  = levels[fs_level]
        d     = df.dropna(subset=cols).copy()
        rel   = d[cols].div(d[cols].sum(axis=1), axis=0)
        Y     = np.sqrt(rel[rel.columns[rel.max(axis=0) >= fs_thr]])
        X_all = d[fq_cols].apply(pd.to_numeric, errors="coerce")
        X_all = ((X_all - X_all.mean()) / X_all.std()).dropna(axis=1)
        n     = len(Y)

        def _r2(Y_, X_sub):
            p = X_sub.shape[1]
            return rda(Y_, X_sub, scale_Y=True, scaling=2).proportion_explained.iloc[:p].sum()

        def _r2adj(r2, p):
            return 1 - (1 - r2) * (n - 1) / (n - p - 1)

        r2adj_scope = _r2adj(_r2(Y, X_all), X_all.shape[1])
        log_area = st.empty()
        lines = [f"R²adj full model (scope): **{r2adj_scope:.4f}**"]
        log_area.markdown("\n\n".join(lines))

        selected, remaining = [], list(X_all.columns)
        np.random.seed(42)
        with st.spinner("Running…"):
            while remaining:
                best = None
                for var in remaining:
                    X_test    = X_all[selected + [var]]
                    r2_obs    = _r2(Y, X_test)
                    r2adj_obs = _r2adj(r2_obs, X_test.shape[1])
                    if r2adj_obs > r2adj_scope:
                        continue
                    r2_null = [_r2(Y.sample(frac=1).set_axis(Y.index), X_test)
                               for _ in range(fs_nperm)]
                    p_val = (np.sum(np.array(r2_null) >= r2_obs) + 1) / (fs_nperm + 1)
                    if p_val <= fs_alpha and (best is None or r2adj_obs > best[2]):
                        best = (var, p_val, r2adj_obs)
                if best is None:
                    break
                var, p_val, r2adj_obs = best
                selected.append(var)
                remaining.remove(var)
                lines.append(f"Step {len(selected)}: **+{var}** → R²adj={r2adj_obs:.4f}, p={p_val:.3f}")
                log_area.markdown("\n\n".join(lines))

        st.session_state["vars_fs"] = selected
        st.success(f"**Selected ({len(selected)}):** {selected}  \nSwitch to the **RDA** tab to visualise.")

    elif "vars_fs" in st.session_state:
        st.success(
            f"**Last result — selected variables ({len(st.session_state['vars_fs'])}):** "
            f"{st.session_state['vars_fs']}"
        )

# ─────────────────────────────────────────────────────────────────────────────
# 5. RDA
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.subheader("RDA biplot")
    st.write(
        "Ordination constrained to the variation explained by the selected physicochemical variables. "
        "**Arrows** = physicochemical variables (direction = where variable increases, length = association strength). "
        "**Axis %** = fraction of community variation explained by the physicochemical variables. "
        "The RDA computation is cached — only recomputed when inputs change."
    )

    default_vars = st.session_state.get(
        "vars_fs", ["pH", "EC (dS/m)", "%OM", "N-NO3 (mg/kg)", "P-PO4 (mg/kg)"]
    )

    ctrl5, plot_area5 = st.columns([1, 3])
    with ctrl5:
        rda_level  = st.selectbox("Level", ["Family", "Genus"], key="rda_level")
        rda_vars   = st.multiselect("Physicochemical variables", fq_cols,
                                     default=default_vars, key="rda_vars")
        rda_thr    = st.slider("Taxon threshold", 0.01, 0.10, 0.02, 0.01,
                                format="%.2f", key="rda_thr")
        rda_nperm  = st.select_slider("Permutations",
                                       options=[99, 199, 299, 499, 999], value=199, key="rda_nperm")
        rda_ell    = st.checkbox("Ellipses", value=True, key="rda_ell")
        rda_labels = st.checkbox("Labels", value=False, key="rda_labels")

    if len(rda_vars) < 2:
        with plot_area5:
            st.warning("Select at least 2 physicochemical variables.")
    else:
        @st.cache_data(show_spinner="Running RDA + permutation test…")
        def _rda_stats(level, variables, threshold, n_perm):
            cols = levels[level]
            d    = df.dropna(subset=cols).copy()
            d["sample"] = _samples(d)
            d["label"]  = d["Tratamiento"].astype(str) + "_" + d["Dia"].astype(str)
            rel  = d[cols].div(d[cols].sum(axis=1), axis=0)
            Y    = np.sqrt(rel[rel.columns[rel.max(axis=0) >= threshold]])
            Y.index = d["sample"]
            X    = d[list(variables)].apply(pd.to_numeric, errors="coerce")
            X    = (X - X.mean()) / X.std()
            X.index = d["sample"]
            res    = rda(Y, X, scale_Y=True, scaling=2)
            pe     = res.proportion_explained.values[:2]
            sc     = res.samples.iloc[:, :2].values
            bp     = res.biplot_scores.iloc[:, :2].values
            p      = X.shape[1]
            R2_obs = float(res.proportion_explained.iloc[:p].sum())
            np.random.seed(42)
            R2_null = [
                rda(Y.sample(frac=1).set_axis(Y.index), X, scale_Y=True, scaling=2)
                .proportion_explained.iloc[:p].sum()
                for _ in range(n_perm)
            ]
            p_val = (np.sum(np.array(R2_null) >= R2_obs) + 1) / (n_perm + 1)
            return (sc, bp, pe, R2_obs, float(p_val),
                    d["Tratamiento"].values.tolist(), d["label"].values.tolist(), len(Y.columns))

        sc_r, bp_r, pe_r, R2_r, pval_r, g_r, lbl_r, n_taxa_r = _rda_stats(
            rda_level, tuple(rda_vars), rda_thr, rda_nperm
        )
        g_r   = np.array(g_r)
        lbl_r = np.array(lbl_r)

        cmap_r = _palette(g_r)
        fig_r, ax_r = plt.subplots(figsize=(9, 7))
        for lab in sorted(set(g_r)):
            m   = g_r == lab
            pts = sc_r[m]
            c   = cmap_r[lab]
            ax_r.scatter(pts[:, 0], pts[:, 1], s=55, color=c, edgecolor="white",
                          linewidth=0.6, zorder=3, label=lab)
            if rda_ell:
                _ellipse(ax_r, pts, c)
        if rda_labels:
            texts_r = [
                ax_r.text(sc_r[i, 0], sc_r[i, 1], lbl_r[i],
                           fontsize=7.5, color=cmap_r[g_r[i]], fontweight="bold")
                for i in range(len(sc_r))
            ]
            fig_r.canvas.draw()
            with contextlib.redirect_stdout(io.StringIO()):
                adjust_text(texts_r, ax=ax_r, arrowprops=dict(arrowstyle="-", color="#aaa", lw=0.6))
        k_scale = 2.8 * np.abs(sc_r).max() / max(np.abs(bp_r).max(), 1e-9)
        for i, v in enumerate(rda_vars):
            ax_r.arrow(0, 0, bp_r[i, 0]*k_scale, bp_r[i, 1]*k_scale,
                       color="#333", width=0.002, head_width=0.06,
                       length_includes_head=True, zorder=4)
            ax_r.text(bp_r[i, 0]*k_scale*1.13, bp_r[i, 1]*k_scale*1.13,
                      v, fontsize=8.5, color="#111", ha="center", fontweight="bold", zorder=5)
        lx = np.array([bp_r[i, 0]*k_scale*1.18 for i in range(len(rda_vars))])
        ly = np.array([bp_r[i, 1]*k_scale*1.18 for i in range(len(rda_vars))])
        all_x = np.concatenate([sc_r[:, 0], lx])
        all_y = np.concatenate([sc_r[:, 1], ly])
        px_ = (all_x.max() - all_x.min()) * 0.12
        py_ = (all_y.max() - all_y.min()) * 0.12
        ax_r.set_xlim(all_x.min() - px_, all_x.max() + px_)
        ax_r.set_ylim(all_y.min() - py_, all_y.max() + py_)
        ax_r.axhline(0, color="gray", lw=0.5, ls="--")
        ax_r.axvline(0, color="gray", lw=0.5, ls="--")
        ax_r.set_xlabel(f"RDA1 ({pe_r[0]*100:.1f}%)")
        ax_r.set_ylabel(f"RDA2 ({pe_r[1]*100:.1f}%)")
        ax_r.set_title(
            f"RDA ({rda_level}, {n_taxa_r} taxa ~ {len(rda_vars)} phys.-chem. vars)",
            fontsize=13, fontweight="bold",
        )
        ax_r.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize="small")
        ax_r.text(0.5, -0.10,
                  f"R²={R2_r:.4f}, p={pval_r:.3f} ({rda_nperm} perm.)   |   RDA2: {pe_r[1]*100:.1f}%",
                  transform=ax_r.transAxes, ha="center", fontsize=8.5,
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f5", edgecolor="#aaa", alpha=0.8))
        plt.tight_layout()
        with plot_area5:
            st.pyplot(fig_r)
        plt.close(fig_r)
