import io, contextlib
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
import networkx as nx
import streamlit as st
from matplotlib.patches import Ellipse
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import spearmanr, kruskal, t as tdist
from skbio.stats.distance import permanova, permdisp, DistanceMatrix
from skbio.stats.ordination import pcoa, rda
from adjustText import adjust_text
from grupos_columnas import columnas_grupos

st.set_page_config(page_title="Greener / BioRem — Analysis", layout="wide")
st.title("Soil microbiome analysis — Greener / BioRem")

st.markdown("""
This interactive tool accompanies the **Greener / BioRem bioremediation study**, which investigates
how different soil treatments reshape the microbial community over time in hydrocarbon-contaminated soil.

**Experimental design**

| Factor | Levels |
|--------|--------|
| Treatment | BA, BS, CT, VCBA, VCBS, VCT |
| Time point | Day 2, 15, 60, 91 |
| Replicates | 3 per treatment × time point |
| **Total** | **54 samples** |

The dataset (`datos_combinados.csv`) contains **664+ variables** per sample: physicochemical parameters
(pH, electrical conductivity, organic matter, nutrients, contaminants), soil texture,
and 16S rRNA amplicon sequencing data quantified at the **family** and **genus** level.
Functional trait predictions from **PAPRICA** are also available and can be used in the heatmap.

**Suggested workflow**

1. **Bar plots** — get a first impression of which taxa dominate and how community composition
   shifts across treatments and time.
2. **Heatmap** — inspect the full abundance matrix and see how samples cluster based on
   composition (or PAPRICA functional profile).
3. **PCoA** — test statistically whether groups differ in composition (PERMANOVA) and that
   the differences are not artefacts of unequal within-group variability (PERMDISP).
4. **Forward selection** — identify which physicochemical variables drive the community
   differences, avoiding over-fitting.
5. **RDA** — visualise the constrained ordination: how samples separate *along the gradients*
   defined by the selected physicochemical variables.
6. **Co-occurrence** — find groups of taxa that move together, and test each group against
   time and against treatment.
""")

st.warning(
    "⚠️ `VCBS_15` and `VCBA_15` have no 16S data — they are excluded from all microbiome analyses "
    "and appear as gaps in bar plots and heatmaps."
)

# ── Data ──────────────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    df = pd.read_csv("datos_combinados.csv")
    # grupos_columnas HAS a "Paprica" key, so deriving this by exclusion returned an empty
    # list and silently disabled the functional dendrogram in the heatmap tab.
    cols_paprica = columnas_grupos["Paprica"]
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

# PCoA (section 3): a PCoA is built from a distance matrix, so it has no loadings. What can be
# recovered is how strongly each taxon tracks an axis (correlation with the sample scores). Rare
# taxa can reach |rho| = 1 by luck on 16 independent points, so only taxa that get somewhere are
# ranked -- the ordination itself still uses every taxon.
PCO_MIN_ABUND = 0.02

# Co-occurrence network (section 6): fixed analysis choices, deliberately kept out of the
# controls. The two overlap almost entirely at n=16 (the FDR is what binds, not the floor),
# so exposing both was two sliders for one decision.
NET_ALPHA   = 0.05          # Benjamini-Hochberg FDR across all taxon pairs
NET_MIN_RHO = 0.6           # effect-size floor on |rho|

# ── Shared helpers ────────────────────────────────────────────────────────────

def _samples(d):
    return d["Tratamiento"].astype(str) + "_" + d["Dia"].astype(str) + "_" + d["Replica"].astype(str)

def _palette(labels):
    labs = sorted(set(labels))
    return {l: c for l, c in zip(labs, plt.cm.tab20(np.linspace(0, 1, len(labs))))}

def _star(p):
    """Significance marker (***/**/*) used by the co-occurrence network legend and panel."""
    if not np.isfinite(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

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


def _place_labels(ax, coords, labels, groups, cmap, avoid=None):
    """One label per unique value in `labels`, placed at the centroid of its points and
    then repelled from every data point (and from the other labels) so it stays readable.
    A white halo keeps it legible if it still lands near a point. Shared by PCoA and RDA
    for consistent, non-overlapping labelling.

    `coords` (n,2): point coordinates. `labels` (n,): label per point (one text per unique).
    `groups` (n,): grouping used for colour. `avoid`: optional (m,2) extra coordinates to
    keep clear (e.g. RDA arrow tips).
    """
    coords = np.asarray(coords)
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    texts = []
    for lb in sorted(set(labels)):
        m = labels == lb
        cx, cy = coords[m].mean(axis=0)
        texts.append(ax.text(cx, cy, lb, fontsize=7.5, color=cmap[groups[m][0]],
                             fontweight="bold", ha="center", va="center", zorder=6,
                             bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7)))
    ax_x, ax_y = coords[:, 0], coords[:, 1]
    if avoid is not None:
        a = np.asarray(avoid)
        ax_x = np.concatenate([ax_x, a[:, 0]])
        ax_y = np.concatenate([ax_y, a[:, 1]])
    with contextlib.redirect_stdout(io.StringIO()):
        adjust_text(texts, x=ax_x, y=ax_y, ax=ax,
                    force_text=(0.4, 0.6), force_static=(0.3, 0.5),
                    expand=(1.3, 1.6), min_arrow_len=8,
                    arrowprops=dict(arrowstyle="-", color="#888", lw=0.6))
    return texts

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Bar plots", "🔥 Heatmap", "🗺️ PCoA", "🔍 Forward selection", "➡️ RDA",
    "🕸️ Co-occurrence",
])

# ─────────────────────────────────────────────────────────────────────────────
# 1. Bar plots
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Relative abundance bar plots")
    st.markdown("""
Stacked bar charts showing the **relative abundance** of each taxon across all treatment × day
combinations. Values are averaged over the 3 replicates and normalised so that each bar sums to 1
(100% relative abundance).

**How to read the chart**

- Each colour represents one taxon (family or genus). Colour assignment is stable within a session.
- The grey segment labelled *Other* pools all taxa whose maximum relative abundance across
  all samples is below the chosen threshold — this keeps the legend readable without discarding
  any actual data from the statistics.
- Bars are ordered by treatment group to make temporal trends (left → right within each treatment)
  and cross-treatment comparisons easy to spot.

**Parameters**

| Parameter | Effect |
|-----------|--------|
| **Level** | Switch between family-level and genus-level taxonomy. |
| **Threshold** | Minimum relative abundance a taxon must reach in at least one sample to get its own colour. Lower = more taxa shown, busier legend. |
""")
    st.divider()
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
    st.markdown("""
A **clustered heatmap** that simultaneously shows the full abundance matrix and the hierarchical
relationships among samples (columns) and taxa (rows).

**Colour scale** — red/blue diverging palette centred on the mean relative abundance across the
matrix. Red cells indicate a taxon is more abundant than average in that sample; blue cells indicate
it is less abundant. This makes it easy to spot which taxa are characteristic of which treatments.

**Dendrograms** — both the row (taxa) and column (samples) trees are built by hierarchical
clustering using **Bray–Curtis dissimilarity** and average linkage. Samples that cluster together
have similar community composition.

**PAPRICA functional profile (optional top tree)** — instead of clustering samples by their
16S composition, you can use the functional trait predictions from PAPRICA
(Phylogenetic Assignment of Microbial Pathway Reconstruction and Interpretation with Assembly).
This answers a different but complementary question: do samples that look functionally similar
also look taxonomically similar?

**Parameters**

| Parameter | Effect |
|-----------|--------|
| **Level** | Family-level (fewer rows, easier to read) or genus-level (more detail). |
| **X axis** | `Treatment + Day` keeps the 16 conditions as measured; `Day` collapses to 4 columns; `Treatment` to 6. Collapsing **merges samples**. |
| **Threshold** | Exclude taxa that never exceed this relative abundance — reduces clutter in the row dendrogram. |
| **Balanced** | Restrict to the complete 4 treatments × 3 days block, so every column averages the same set. |
| **Top tree: PAPRICA** | If checked, the column dendrogram uses PAPRICA functional distances; otherwise it uses 16S Bray–Curtis. |

**Collapsing the X axis.** The design is unbalanced, so merged columns are not automatically
comparable — the labels therefore state what they contain (`Day 15 (2 trt)`, `CT (1 d)`).
Day 15 holds only BA and BS; day 2 includes the untreated controls (dropping them shifts that
column by Bray–Curtis 0.143); CT and VCT are single day-2 snapshots. **Balanced** restricts to
BA/BS/VCBA/VCBS × days 2/60/91, the largest complete block. Averaging is done on **relative
abundances**, giving each condition equal weight — summing raw reads would let deeply
sequenced samples dominate (depth varies 2.4× here). Expect **fewer taxa** when collapsing:
averaging flattens peaks, so at 5 % and family level 26 taxa become 15 by day.
""")
    st.divider()
    ctrl2, plot_area2 = st.columns([1, 4])
    with ctrl2:
        hm_level   = st.selectbox("Level", ["Family", "Genus"], key="hm_level")
        hm_xaxis   = st.selectbox("X axis", ["Treatment + Day", "Day", "Treatment"],
                                  key="hm_xaxis")
        hm_thr     = st.slider("Threshold", 0.02, 0.10, 0.05, 0.005,
                                format="%.3f", key="hm_thr")
        hm_balanced = st.checkbox("Balanced (only the complete 4 trt × 3 days block)",
                                  value=False, key="hm_balanced")
        hm_paprica = st.checkbox("Top tree: PAPRICA", value=True, key="hm_paprica")

    @st.cache_data
    def _heatmap_data(level, x_axis, threshold, balanced, paprica):
        cols = levels[level]
        d    = df.dropna(subset=cols).copy()
        key  = d["Tratamiento"].astype(str) + "_" + d["Dia"].astype(str)
        td   = d.groupby(key)[cols].mean()
        meta = d.groupby(key)[["Tratamiento", "Dia"]].first()
        pap  = d.groupby(key)[cols_paprica].mean()

        if balanced:
            # Largest complete treatment x day block: treatments seen on >= 3 days AND days
            # seen in >= 3 treatments. Here that is BA/BS/VCBA/VCBS x days 2/60/91, so every
            # column averages exactly the same set and the columns become comparable.
            t_ok = meta["Tratamiento"].map(meta.groupby("Tratamiento").size()) >= 3
            d_ok = meta["Dia"].map(meta.groupby("Dia").size()) >= 3
            sel  = (t_ok & d_ok).values
            td, meta, pap = td[sel], meta[sel], pap[sel]

        # Relative abundance FIRST, then average: summing raw reads would weight each
        # condition by its sequencing depth (varies 2.4x here) instead of weighting equally.
        rel = td.div(td.sum(axis=1), axis=0)
        if x_axis == "Day":
            grp   = meta["Dia"]
            names = {k: f"Day {k} ({n} trt)" for k, n in grp.value_counts().items()}
        elif x_axis == "Treatment":
            grp   = meta["Tratamiento"]
            names = {k: f"{k} ({n} d)" for k, n in grp.value_counts().items()}
        else:
            grp, names = None, None

        if grp is None:
            agg, pagg = rel, pap.div(pap.sum(axis=1), axis=0)
        else:
            # Column labels carry what they average, so the unbalanced design stays visible
            agg  = rel.groupby(grp.values).mean().rename(index=names)
            pagg = (pap.div(pap.sum(axis=1), axis=0)
                       .groupby(grp.values).mean().rename(index=names))

        # Threshold applied to the aggregated matrix, so taxa are chosen on what is drawn
        agg = agg[agg.columns[agg.max(axis=0) >= threshold]]
        mat = agg.T.div(agg.T.sum(axis=0), axis=1)

        col_lnk = None
        if paprica and len(cols_paprica) and len(mat.columns) > 2:
            # Same aggregation as the taxa, so the functional tree matches the columns drawn
            pagg    = pagg.reindex(mat.columns)
            col_lnk = linkage(pdist(pagg.values, metric="braycurtis"), method="average")
        return mat, col_lnk, agg.shape[1]

    mat_hm, col_lnk_hm, n_taxa_hm = _heatmap_data(hm_level, hm_xaxis, hm_thr,
                                                  hm_balanced, hm_paprica)
    with ctrl2:
        st.caption(f"Taxa shown: **{n_taxa_hm}**")

    # Figure scaled to the matrix: collapsing the X axis leaves 3-6 columns, which a fixed
    # 12-inch width would stretch into absurdly wide cells. The coefficients reproduce the
    # original (12, 16) for the default view (16 columns, 26 taxa).
    g_hm = sns.clustermap(
        mat_hm, method="average", metric="braycurtis", col_linkage=col_lnk_hm,
        cmap="RdBu_r", center=float(mat_hm.values.mean()),
        figsize=(2.0 + 0.62 * mat_hm.shape[1], min(24.0, 3.0 + 0.50 * mat_hm.shape[0])),
        dendrogram_ratio=(0.12, 0.10),
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
    st.markdown("""
**Principal Coordinates Analysis (PCoA)** is an unconstrained ordination method. It takes the
pairwise dissimilarity matrix between all 48 samples (with 16S data) and projects them onto a
2-dimensional plane so that samples that are most similar end up closest to each other.
The percentage on each axis label tells you how much of the total between-sample variation is
captured by that axis.

Unlike PCA, PCoA can work with any dissimilarity metric. **Bray–Curtis** (the default) is the
standard choice for microbial ecology: it is bounded between 0 (identical communities) and 1
(no shared taxa), and it ignores double-zeros (two samples sharing no taxa are not considered
"similar" just because both lack a taxon).

---

**PERMANOVA** (Permutational Multivariate Analysis of Variance) tests statistically whether
the group centroids differ more than expected by random chance. It permutes sample labels 999
times to build a null distribution and computes an empirical p-value.

- **R²** = proportion of total community variation explained by the grouping variable.
  An R² of 0.30 means 30% of variation is attributable to treatment/day differences.
- A significant p-value (< 0.05) means the groups are compositionally distinct.

**PERMDISP** (Permutational Multivariate Dispersion) checks a key assumption of PERMANOVA:
that within-group variability is homogeneous across groups. If PERMDISP is also significant
(p < 0.05), the PERMANOVA result may partly reflect differences in *spread* rather than
differences in *location*, and should be interpreted with caution (the plot flags this in red).

---

**Parameters**

| Parameter | Effect |
|-----------|--------|
| **Level** | Taxonomy level used to compute dissimilarities. |
| **Group by** | Defines the groups for PERMANOVA/PERMDISP and the colour coding of points. |
| **Metric** | Dissimilarity index. Bray–Curtis is recommended for relative abundance data. Jaccard is presence/absence only. Euclidean is not recommended for compositional data. |
| **Ellipses** | 95% confidence ellipses (normal distribution approximation) for each group. Useful when groups overlap. |
| **Sample labels** | Places one label per Treatment+Day combination at the centroid of its 3 replicates. |
| **Top taxa** | How many families are ranked per axis in the second figure. |

---

**Which taxa drive each axis (second figure).** A PCoA is built from distances, not from the
taxa, so it has **no loadings**. What the second figure shows is the **Spearman correlation of
each family with the axis scores** — how strongly it tracks the axis. Green = more abundant
towards the positive side of the axis, red = towards the negative. Only taxa reaching 2 %
somewhere are ranked (a rare taxon can hit |ρ| = 1 by luck); the ordination itself still uses
every taxon. No p-values are given: the three 16S replicate rows are exact copies, so they land
on the same point and only 16 observations are independent — that leaves ρ unchanged but would
inflate any p-value.
""")
    st.divider()
    ctrl3, plot_area3 = st.columns([1, 3])
    with ctrl3:
        pc_level   = st.selectbox("Level", ["Family", "Genus"], key="pc_level")
        pc_groupby = st.selectbox("Group by",
                                   ["Treatment", "Day", "Treatment+Day"], key="pc_groupby")
        pc_metric  = st.selectbox("Metric",
                                   ["braycurtis", "jaccard", "euclidean"], key="pc_metric")
        pc_ell     = st.checkbox("Ellipses", value=False, key="pc_ell")
        pc_labels  = st.checkbox("Sample labels", value=True, key="pc_labels")
        pc_top     = st.slider("Top taxa", 3, 15, 8, 1, key="pc_top")

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
        # Which taxa track each axis. Spearman (rank-based) because a distance-based ordination
        # is not expected to be linear in abundance. No p-values: the 16S replicate rows are
        # exact copies, so the three replicates of a condition land on the same point and only
        # 16 of the 48 rows are independent -- the correlation is unaffected, its p-value would
        # not be. The full table is returned so the "Top taxa" control does not re-run PERMANOVA.
        keep = rel.columns[rel.max(axis=0) >= PCO_MIN_ABUND]
        corr = pd.DataFrame({f"PCo{k_ + 1}": [spearmanr(rel[t].values, xy[:, k_])[0]
                                              for t in keep]
                             for k_ in (0, 1)}, index=keep).fillna(0)
        return (xy, pe, F, R2, float(pm["p-value"]),
                float(pd_["test statistic"]), float(pd_["p-value"]),
                g.values.tolist(), d["label"].values.tolist(), corr)

    xy_pc, pe_pc, F_pc, R2_pc, pm_p, pdF_pc, pdp_pc, g_pc, lbl_pc, corr_pc = _pcoa_stats(
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
        # One label per Treatment_Day combination, at its centroid, repelled from points
        fig_pc.canvas.draw()
        _place_labels(ax_pc, xy_pc, lbl_pc, g_pc, cmap_pc)
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

    fig_ld, axes_ld = plt.subplots(1, 2, figsize=(13, 0.34 * pc_top + 2.2), sharex=True)
    for ax_ld, axis in zip(axes_ld, corr_pc.columns):
        # Pick the pc_top strongest by |rho|, then order them by signed rho so the positive
        # and negative ends of the axis read as two blocks instead of interleaving
        top = corr_pc[axis].reindex(corr_pc[axis].abs().sort_values().index)[-pc_top:].sort_values()
        ax_ld.barh(range(len(top)), top.values, edgecolor="white", linewidth=0.5,
                   color=["#2e8b57" if v > 0 else "#c0392b" for v in top.values])
        ax_ld.set_yticks(range(len(top)))
        ax_ld.set_yticklabels(top.index, fontsize=7.5)
        ax_ld.axvline(0, color="gray", lw=0.6)
        ax_ld.set_xlim(-1, 1)
        ax_ld.set_xlabel("Spearman ρ with the axis scores")
        ax_ld.set_title(f"{axis} ({pe_pc[int(axis[-1]) - 1]*100:.1f}%)",
                        fontsize=11, fontweight="bold")
    fig_ld.suptitle(f"Taxa most associated with each axis ({pc_level}, top {pc_top} by |ρ|)",
                    fontsize=12, fontweight="bold")
    plt.tight_layout()
    with plot_area3:
        st.pyplot(fig_ld)
    plt.close(fig_ld)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Forward selection
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Forward selection of physicochemical variables")
    st.markdown("""
Before building the RDA model, we need to decide which physicochemical variables to include.
Including all ~14 candidates would over-fit the model (the axes would partially reflect noise,
not real community–environment relationships). **Forward selection** solves this by adding
variables one at a time, keeping only those that pass two simultaneous stopping criteria:

1. **Statistical significance** — the variable must contribute significantly to the explained
   variation (permutation test, p ≤ α).
2. **Adjusted R² ceiling** — the adjusted R² of the growing model must not exceed the adjusted
   R² of the global model that includes all variables. This is the double stopping criterion
   from [Blanchet, Legendre & Borcard (2008)](https://doi.org/10.1890/07-0986.1),
   equivalent to R's `vegan::ordiR2step`.

At each step the candidate that most increases the adjusted R² is identified, and its **marginal**
contribution — conditioned on the variables already selected — is assessed with a partial-RDA
permutation test (the same pseudo-F + reduced-model scheme as vegan's `anova.cca`). It is added only
if that marginal test is significant *and* the adjusted R² stays below the ceiling. The algorithm
stops as soon as the best candidate fails either criterion.

Y (Hellinger-transformed) is only **centred** before the RDA — matching vegan's `rda()`, not
standardised — so the procedure reproduces `ordiR2step` (verified against vegan on this dataset:
same variables, same order, same per-step adjusted R²).

**How to use this tab**

Run forward selection at least once before using the RDA tab. The selected variables will be
automatically pre-loaded as the default variable set in the RDA. You can always override them
manually in the RDA tab.

**Parameters**

| Parameter | Effect |
|-----------|--------|
| **Level** | Should match the level you plan to use in the RDA. |
| **Taxon threshold** | Removes very rare taxa before computing the Hellinger-transformed abundance matrix Y. |
| **Significance α** | Permutation p-value threshold for retaining a variable. Default 0.05. |
| **Permutations** | More permutations → more precise p-values. 999 is recommended (it runs in ~1 s). |
""")
    st.info("**Tip:** Run this tab first, then switch to the RDA tab — results are passed automatically.")
    st.caption("⏱️ Fast (~1 s): the permutation test runs only on the best candidate at each step.")

    col_a, col_b = st.columns(2)
    with col_a:
        fs_level  = st.selectbox("Level", ["Family", "Genus"], key="fs_level")
        fs_thr    = st.slider("Taxon threshold", 0.01, 0.10, 0.02, 0.01,
                               format="%.2f", key="fs_thr")
    with col_b:
        fs_alpha  = st.slider("Significance α", 0.01, 0.10, 0.05, 0.01,
                               format="%.2f", key="fs_alpha")
        fs_nperm  = st.select_slider("Permutations",
                                      options=[99, 199, 299, 499, 999], value=999, key="fs_nperm")

    if st.button("▶ Run forward selection", type="primary"):
        # Faithful replica of vegan::ordiR2step.
        # RDA = multivariate regression of Y (Hellinger, only centred — like vegan's
        # rda(), not standardised) on X (standardised physicochemicals). Each step adds
        # the variable that most increases the adjusted R², provided (a) its MARGINAL
        # contribution is significant (partial RDA conditioned on the already-selected
        # variables, same pseudo-F + reduced-model permutation as anova.cca) and (b) the
        # adjusted R² stays below the full-model ceiling (R2scope).
        cols  = levels[fs_level]
        d     = df.dropna(subset=cols).copy()
        rel   = d[cols].div(d[cols].sum(axis=1), axis=0)
        Y     = np.sqrt(rel[rel.columns[rel.max(axis=0) >= fs_thr]]).values   # Hellinger
        X     = d[fq_cols].apply(pd.to_numeric, errors="coerce")
        X     = ((X - X.mean()) / X.std()).dropna(axis=1)                     # standardised
        Xcols = list(X.columns)
        X     = X.values
        n     = len(Y)
        Yc    = Y - Y.mean(0)            # vegan rda() centres Y (does not scale it)
        ss_tot = (Yc ** 2).sum()
        rng   = np.random.default_rng(42)

        def _r2(idx):
            Z = np.column_stack([np.ones(n), X[:, idx]])
            beta = np.linalg.lstsq(Z, Yc, rcond=None)[0]
            return ((Z @ beta) ** 2).sum() / ss_tot

        def _r2adj(r2, p):
            return 1 - (1 - r2) * (n - 1) / (n - p - 1)

        def _marginal_p(sel_idx, var_idx):
            Zc = np.column_stack([np.ones(n)] + ([X[:, sel_idx]] if sel_idx else []))
            Pc = Zc @ np.linalg.pinv(Zc)
            Yr = Yc - Pc @ Yc
            vr = X[:, var_idx] - Pc @ X[:, var_idx]
            df_res = n - np.linalg.matrix_rank(Zc) - 1

            def _F(Ym):
                nu = ((Ym.T @ vr) ** 2).sum() / (vr @ vr) if vr @ vr > 1e-12 else 0.0
                return nu / (((Ym ** 2).sum() - nu) / df_res)

            f_obs, cnt = _F(Yr), 1
            for _ in range(fs_nperm):
                Yp = Yr[rng.permutation(n)]
                Yp = Yp - Pc @ Yp
                cnt += _F(Yp) >= f_obs - 1e-12
            return cnt / (fs_nperm + 1)

        p_full      = np.linalg.matrix_rank(np.column_stack([np.ones(n), X])) - 1
        r2adj_scope = _r2adj(_r2(list(range(X.shape[1]))), p_full)
        log_area = st.empty()
        lines = [f"R²adj full model (scope): **{r2adj_scope:.4f}**"]
        log_area.markdown("\n\n".join(lines))

        selected, remaining = [], list(range(X.shape[1]))
        with st.spinner("Running…"):
            while remaining:
                r2adj_new, var = max((_r2adj(_r2(selected + [v]), len(selected) + 1), v)
                                     for v in remaining)
                if r2adj_new > r2adj_scope:                    # stop (a): R2scope
                    lines.append(f"Stop: **{Xcols[var]}** would exceed the ceiling "
                                 f"(R²adj={r2adj_new:.4f}).")
                    break
                p_val = _marginal_p(selected, var)
                if p_val > fs_alpha:                           # stop (b): not significant
                    lines.append(f"Stop: **{Xcols[var]}** not significant (p={p_val:.3f}).")
                    break
                selected.append(var)
                remaining.remove(var)
                lines.append(f"Step {len(selected)}: **+{Xcols[var]}** → "
                             f"R²adj={r2adj_new:.4f}, p={p_val:.3f}")
                log_area.markdown("\n\n".join(lines))

        sel_names = [Xcols[i] for i in selected]
        log_area.markdown("\n\n".join(lines))
        st.session_state["vars_fs"] = sel_names
        st.success(f"**Selected ({len(sel_names)}):** {sel_names}  \nSwitch to the **RDA** tab to visualise.")

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
    st.markdown("""
**Redundancy Analysis (RDA)** is a *constrained* ordination: unlike PCoA, which places samples
freely in ordination space, RDA only allows the axes to be linear combinations of the
physicochemical predictor variables. The result is a biplot that simultaneously shows:

- **Sample scores** (points) — where each sample sits in the space defined by the predictors.
  Samples that are close together have similar microbial communities *and* similar physicochemical
  conditions.
- **Arrows** — the selected physicochemical variables. The direction of each arrow shows where
  that variable increases; the length is proportional to how strongly it drives community
  variation. Arrows pointing in the same direction indicate correlated variables. Arrows pointing
  to a cluster of samples indicate that those samples have higher values of that variable.
- **Axis labels (%)** — the percentage of *total constrained variation* captured by each axis.
  Note: this is not the same as in PCoA — RDA axes explain variation in Y *that is attributable
  to X*, not total community variation.

**Hellinger transformation** — species abundance data are Hellinger-transformed (√ relative
abundance) before the RDA. This down-weights dominant taxa and makes the ordination less sensitive
to rare taxa with extreme values, which is the recommended pre-treatment for RDA on ecological
community data (Legendre & Gallagher, 2001).

**Global permutation test** — the R² reported below the plot is the total constrained R²
(fraction of community variation explained by all selected variables together). The p-value
comes from a permutation test (sample labels shuffled, RDA re-computed).

**Parameters**

| Parameter | Effect |
|-----------|--------|
| **Level** | Taxonomy level for the community matrix Y. |
| **Physicochemical variables** | Variables to use as predictors X. Pre-populated from the forward selection result; you can add or remove manually. |
| **Taxon threshold** | Exclude very rare taxa from Y (same logic as bar plots). |
| **Permutations** | Number of permutations for the global test. |
| **Ellipses** | 95% confidence ellipses per treatment group. |
| **Labels** | Show Treatment+Day label for each individual sample point. |
""")
    st.divider()

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
                                       options=[99, 199, 299, 499, 999], value=999, key="rda_nperm")
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
            # scale_Y=False -> Y only centred, like vegan's rda() (Y is already Hellinger).
            # Consistent with the forward-selection tab.
            res    = rda(Y, X, scale_Y=False, scaling=2)
            pe     = res.proportion_explained.values[:2]
            sc     = res.samples.iloc[:, :2].values
            bp     = res.biplot_scores.iloc[:, :2].values
            p      = X.shape[1]
            R2_obs = float(res.proportion_explained.iloc[:p].sum())
            np.random.seed(42)
            R2_null = [
                rda(Y.sample(frac=1).set_axis(Y.index), X, scale_Y=False, scaling=2)
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
        # Arrows (drawn before labels so their tips can be avoided by the sample labels)
        k_scale = 2.8 * np.abs(sc_r).max() / max(np.abs(bp_r).max(), 1e-9)
        tips = bp_r[:, :2] * k_scale
        for i, v in enumerate(rda_vars):
            ax_r.arrow(0, 0, tips[i, 0], tips[i, 1],
                       color="#333", width=0.002, head_width=0.06,
                       length_includes_head=True, zorder=4)
            ax_r.text(tips[i, 0]*1.13, tips[i, 1]*1.13,
                      v, fontsize=8.5, color="#111", ha="center", fontweight="bold", zorder=5)
        if rda_labels:
            # One label per Treatment_Day combination, at its centroid, repelled from points + arrow tips
            fig_r.canvas.draw()
            _place_labels(ax_r, sc_r, lbl_r, g_r, cmap_r, avoid=tips)
        lx = tips[:, 0] * 1.18
        ly = tips[:, 1] * 1.18
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

# ─────────────────────────────────────────────────────────────────────────────
# 6. Co-occurrence network
# ─────────────────────────────────────────────────────────────────────────────
with tab6:
    st.subheader("Co-occurrence network (taxa × time × treatment)")
    st.markdown("""
**Which bacteria move together — and does each of those groups follow time or treatment?**

**Reading the figures**

- **Circle** = one family (or genus); the bigger, the more abundant. **Line** = those two rise
  and fall together: **green** = both go up, **red** = one up, the other down; thicker = stronger.
- **Colour** = *module*, a block of families that behaves as one. The legend gives its trend with
  time, ρ(day), and/or the treatment where it peaks (`*` p<0.05, `**` p<0.01, `***` p<0.001).
- Families with no significant line are not drawn and belong to no module.
- **Second figure:** how much of the community each module holds in every condition (%).

**How it is built**

1. One row per condition — 16S is sequenced on a *pooled* sample, so there are
   **16 observations, not 48**.
2. *Edges from* fixes what a line means (below).
3. Abundances → % of the community; taxa above the **threshold** in at least one condition are kept.
4. CLR transform, so that "one goes up, the rest go down" is not just the constant sum.
5. Spearman on every pair; a line needs **FDR < 0.05 and |ρ| ≥ 0.6** (fixed in the code as
   `NET_ALPHA` and `NET_MIN_RHO`).
6. **Modules** blocks by greedy modularity maximisation on the network. The modularity **Q** in
   the footer says how well that number of blocks fits — compare it as you move the slider.

**Edges from** — it does not filter samples. It subtracts the mean of the *other* factor and
correlates what is left over.

| Mode | What is subtracted | Conditions | A line means |
|------|--------------------|-----------|--------------|
| **Both** | nothing | 16 | "these two co-vary somewhere in the experiment" — not why |
| **Time** | each treatment's own mean | 14 | "these two follow the same trajectory through time" |
| **Treatment** | each day's mean | 14 | "these two respond to the treatments in the same way" |

*Time* drops CT and VCT, *Treatment* drops day 15: subtracting a mean needs ≥ 3 conditions per
group, or it manufactures structure instead of removing it. It also costs one degree of freedom
per group, so p-values use the residual df (n − g − 1). Each module is tested **only against the
factor its edges came from** — a module with a strong ρ(day) but no treatment effect is
*successional*; the reverse is a *treatment response*.
""")
    st.warning(
        "**n = 16, not 48.** The 16S data comes from a **pooled** sample per treatment × day: "
        "the three replicate rows are exact copies of each other. Correlating across the 48 rows "
        "would be pseudoreplication and would shrink every p-value, so the network is built on "
        "the 16 genuinely independent observations."
    )
    st.divider()
    ctrl6, plot_area6 = st.columns([1, 4])
    with ctrl6:
        nw_level  = st.selectbox("Level", ["Family", "Genus"], key="nw_level")
        nw_factor = st.selectbox("Edges from", ["Both", "Time", "Treatment"],
                                 key="nw_factor")
        nw_thr   = st.slider("Threshold", 0.005, 0.10, 0.02, 0.005,
                             format="%.3f", key="nw_thr")
        nw_k     = st.slider("Modules", 2, 8, 4, 1, key="nw_k")
        nw_lbl   = st.checkbox("Node labels", value=True, key="nw_lbl")
        st.caption(f"Fixed: FDR<{NET_ALPHA}, |ρ|≥{NET_MIN_RHO}, CLR on")

    cols_n = levels[nw_level]
    d_n    = df.dropna(subset=cols_n).copy()
    # Pooled 16S: collapse the 3 identical replicate rows to the 16 real observations
    key_n  = d_n["Tratamiento"].astype(str) + "_" + d_n["Dia"].astype(str)
    td_n   = d_n.groupby(key_n)[cols_n].mean()
    meta_n = d_n.groupby(key_n)[["Tratamiento", "Dia"]].first()

    # `nw_factor` decides what an edge MEANS. 'Time' removes each treatment's own mean, so
    # what is left is how a taxon moves through time; 'Treatment' removes each day's mean, so
    # what is left is how it responds to the treatments. Conditions that cannot survive the
    # centring are dropped first: a group of size 1 centres to exactly zero (CT and VCT, one
    # day each), and a group of size 2 centres to two mirror-image rows whose correlation is
    # forced to -1 for every taxon (day 15, only BA and BS). Hence the >= 3 rule.
    grp_n = {"Time": meta_n["Tratamiento"].astype(str),
             "Treatment": meta_n["Dia"].astype(str)}.get(nw_factor)
    if grp_n is not None:
        ok_n    = grp_n.groupby(grp_n.values).transform("size") >= 3
        dropped_n = sorted(set(grp_n[~ok_n]))
        td_n, meta_n, grp_n = td_n[ok_n.values], meta_n[ok_n.values], grp_n[ok_n.values]
    else:
        dropped_n = []

    # Threshold applied AFTER the subset, so taxa are selected on the data actually used
    rel_n  = td_n.div(td_n.sum(axis=1), axis=0)
    R_n    = rel_n[rel_n.columns[rel_n.max(axis=0) >= nw_thr]]
    R_n    = R_n.loc[:, R_n.std() > 0]          # a constant taxon has undefined correlation
    taxa_n = list(R_n.columns)

    n_obs_n = R_n.shape[0]
    n_g_n   = grp_n.nunique() if grp_n is not None else 1
    df_e_n  = n_obs_n - n_g_n - 1

    if len(taxa_n) < 3:
        with plot_area6:
            st.info("Fewer than 3 taxa pass the threshold — lower it.")
    elif df_e_n < 1:
        with plot_area6:
            st.info(f"Not enough residual degrees of freedom "
                    f"(n={n_obs_n}, groups={n_g_n}).")
    else:
        # Centred log-ratio: relative abundances sum to 1, so an abundant taxon mechanically
        # pushes every other one down. Without CLR only 8 of 124 edges came out negative here,
        # which is the constant sum talking, not biology. Applied BEFORE the factor centring,
        # since it is a per-sample transform.
        M_n = np.log(R_n + 1e-6)
        M_n = M_n.sub(M_n.mean(axis=1), axis=0)
        if grp_n is not None:
            M_n = M_n.sub(M_n.groupby(grp_n.values).transform("mean"))
        rho_n, _ = spearmanr(M_n.values)
        rho_n     = np.nan_to_num(rho_n)
        # Centring spends one degree of freedom per group, so the p-values spearmanr derives
        # from n-2 would be anticonservative. Recompute them from the t approximation on the
        # residual df (g = 1 when nothing is centred out, which recovers the usual n-2).
        tstat_n   = rho_n * np.sqrt(df_e_n / np.clip(1 - rho_n ** 2, 1e-12, None))
        pmat_n    = 2 * tdist.sf(np.abs(tstat_n), df_e_n)
        iu_n      = np.triu_indices(len(taxa_n), k=1)
        pv_n      = np.nan_to_num(pmat_n[iu_n], nan=1.0)
        rv_n      = rho_n[iu_n]
        # Benjamini-Hochberg FDR across every taxon pair
        o_n       = np.argsort(pv_n)
        q_n       = np.empty_like(pv_n)
        q_n[o_n]  = np.minimum.accumulate(
            (pv_n[o_n] * pv_n.size / np.arange(1, pv_n.size + 1))[::-1])[::-1]
        keep_n    = (np.clip(q_n, 0, 1) < NET_ALPHA) & (np.abs(rv_n) >= NET_MIN_RHO)
        if not keep_n.any():
            with plot_area6:
                st.info("No pair survives FDR and |ρ| — lower the threshold to bring in "
                        "more taxa.")
            st.stop()   # section 6 is the last one rendered, so nothing else is lost

        G_n = nx.Graph()
        G_n.add_nodes_from(range(len(taxa_n)))
        G_n.add_edges_from((int(a), int(b), {"w": abs(float(r)), "r": float(r)})
                           for a, b, r in zip(iu_n[0][keep_n], iu_n[1][keep_n], rv_n[keep_n]))
        # Taxa without a single significant edge are not part of any block of the network:
        # dropped from the drawing (the layout flings them to the margins) and from the modules
        shown_n = [i for i in G_n.nodes if G_n.degree(i) > 0]
        G_n     = G_n.subgraph(shown_n).copy()

        # Modules = blocks of the network itself, by greedy modularity maximisation
        # (Clauset-Newman-Moore) split into exactly nw_k blocks (cutoff=best_n=k). This replaced
        # cutting the taxa dendrogram, which optimises nothing about the network: at the same k
        # it gives a clearly worse partition (Q=0.22 vs 0.35 here, family level, Both).
        # Communities come back largest-first, so M1 is always the biggest block.
        comms_n = nx.community.greedy_modularity_communities(G_n, weight="w",
                                                             cutoff=nw_k, best_n=nw_k)
        Q_n     = nx.community.modularity(G_n, comms_n, weight="w")
        mod_n   = {i: m for m, c in enumerate(comms_n, start=1) for i in c}

        # Per-module response, reported only for the factor the edges were built from:
        # showing a treatment test on a network whose treatment effect was centred out would
        # invite a reading the figure does not support. Always computed on UNCENTRED
        # abundances, so the numbers stay in interpretable units.
        show_time_n = nw_factor in ("Both", "Time")
        show_trt_n  = nw_factor in ("Both", "Treatment")
        mstats_n = {}
        for m in sorted(set(mod_n.values())):
            members = [taxa_n[i] for i in shown_n if mod_n[i] == m]
            score   = R_n[members].sum(axis=1)  # share of the community held by the module
            r_t = p_t = p_tr = np.nan
            if show_time_n:
                r_t, p_t = spearmanr(score.values, meta_n["Dia"].values)
            if show_trt_n:
                try:
                    p_tr = kruskal(*[score.values[meta_n["Tratamiento"].values == t]
                                     for t in meta_n["Tratamiento"].unique()])[1]
                except ValueError:              # identical values inside some treatment
                    p_tr = np.nan
            mstats_n[m] = dict(n=len(members), score=score, rho=r_t, p_time=p_t, p_trt=p_tr,
                               top=score.groupby(meta_n["Tratamiento"].values).mean().idxmax())

        def _mlabel_n(m):
            s, bits = mstats_n[m], [f"M{m} (n={mstats_n[m]['n']})"]
            if show_time_n:
                bits.append(f"ρ(day)={s['rho']:+.2f}{_star(s['p_time'])}")
            if show_trt_n:
                bits.append(f"top:{s['top']}{_star(s['p_trt'])}")
            return "   ".join(bits)

        pos_n   = nx.spring_layout(G_n, seed=42, weight="w",
                                   k=3.4 / np.sqrt(max(len(shown_n), 1)), iterations=400)
        xy_n    = np.array([pos_n[i] for i in shown_n]) if shown_n else np.empty((0, 2))
        mods_n  = np.array([mod_n[i] for i in shown_n])
        cmap_n  = _palette(mods_n)
        dark_n  = {m: tuple(np.r_[np.array(c[:3]) * 0.55, 1.0]) for m, c in cmap_n.items()}
        amean_n = R_n.mean()
        size_n  = 90 + 1400 * (amean_n / amean_n.max()).values[shown_n]

        fig_n, ax_n = plt.subplots(figsize=(11, 9))
        for sign, colr in ((1, "#2e8b57"), (-1, "#c0392b")):  # green = co-occur, red = exclude
            sel = [(a, b) for a, b, e in G_n.edges(data=True) if np.sign(e["r"]) == sign]
            if sel:
                nx.draw_networkx_edges(G_n, pos_n, edgelist=sel, ax=ax_n, edge_color=colr,
                                       alpha=0.35, width=[G_n[a][b]["w"] * 1.8 for a, b in sel])
        if shown_n:
            ax_n.scatter(xy_n[:, 0], xy_n[:, 1], s=size_n, c=[cmap_n[m] for m in mods_n],
                         edgecolor="white", linewidth=0.8, zorder=3)
        for m in sorted(mstats_n):
            ax_n.scatter([], [], s=60, color=cmap_n[m], edgecolor="white", linewidth=0.8,
                         label=_mlabel_n(m))
        if nw_lbl and shown_n:
            _place_labels(ax_n, xy_n, np.array(taxa_n)[shown_n], mods_n, dark_n)
        edge_of_n = {"Both": "time + treatment", "Time": "time only",
                     "Treatment": "treatment only"}[nw_factor]
        ax_n.set_title(
            f"Co-occurrence network ({nw_level}, {len(shown_n)} of {len(taxa_n)} taxa connected)"
            f"\nedges from: {edge_of_n}",
            fontsize=13, fontweight="bold",
        )
        ax_n.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize="small")
        ax_n.axis("off")
        npos_n  = int((rv_n[keep_n] > 0).sum())
        nneg_n  = int(keep_n.sum()) - npos_n
        plt.tight_layout()
        # Footer anchored to the figure: the legend shifts the axes and an axes-relative
        # footer gets clipped off the left edge.
        fig_n.text(0.5, 0.015,
                   f"n={n_obs_n} pooled conditions, df={df_e_n}"
                   f"{'   (excluded: ' + ', '.join(dropped_n) + ')' if dropped_n else ''}   |   "
                   f"{int(keep_n.sum())} edges ({npos_n} positive, {nneg_n} negative) "
                   f"of {pv_n.size} pairs   |   "
                   f"FDR<{NET_ALPHA}, |ρ|≥{NET_MIN_RHO}, CLR   |   "
                   f"{len(mstats_n)} modules (Q={Q_n:.3f})",
                   ha="center", fontsize=8.5,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f5",
                             edgecolor="#aaa", alpha=0.8))
        with plot_area6:
            st.pyplot(fig_n)
        plt.close(fig_n)

        # Module × condition panel: the half that literally answers "time and treatment"
        # Only the conditions that actually entered the calculation, in the canonical ORDER
        cond_n  = [c for c in ORDER if c in R_n.index]
        panel_n = pd.DataFrame({m: mstats_n[m]["score"] for m in sorted(mstats_n)}).T
        panel_n = panel_n.reindex(columns=cond_n)
        fig_p, ax_p = plt.subplots(figsize=(13, 0.9 * len(mstats_n) + 2.2))
        sns.heatmap(panel_n * 100, ax=ax_p, cmap="YlGnBu", annot=True, fmt=".1f",
                    annot_kws={"size": 7}, linewidths=0.4, linecolor="white",
                    cbar_kws={"label": "Module share of community (%)"})
        ax_p.set_yticklabels([_mlabel_n(m) for m in sorted(mstats_n)], rotation=0)
        ax_p.set(xlabel="Treatment + Day", ylabel="")
        ax_p.set_title(f"Module abundance across treatment and time "
                       f"({nw_level}, edges from {edge_of_n})",
                       fontsize=12, fontweight="bold")
        plt.tight_layout()
        with plot_area6:
            st.pyplot(fig_p)
        plt.close(fig_p)
