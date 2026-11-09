import sys
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns

####Todo###
#corr_xyに1.0と-1.0の組み合わせがあるがrmsは異なる
#preschoolerのcorr_xyは高いのに、固定効果の係数は負


df=pd.read_csv(sys.argv[1])

df_within = df[(df["species1"] == df["species2"])&(df["subj1"] != df["subj2"])].copy()
df_within["species"] = df_within["species1"]

# ペアを順序不問で正規化（subj1-subj2 と subj2-subj1 を同一視）
df_within["pair_id"] = df_within.apply(
    lambda r: "_".join(sorted([r["subj1"], r["subj2"]])), axis=1
)

print(f"the pairs of same species: {len(df_within)}")
print(df_within.groupby("species")["pair_id"].nunique()
      .rename("the number of unique pairs"))

# ============================================================
# 2. 群内分散の除外：trial → pair レベルへ集約
#    vid × species × pair_id で平均を取る
#    → 同一ペアが同一動画を複数trialで見た変動を除去
# ============================================================
df_agg = (
    df_within
    .groupby(["vid", "species", "pair_id", "subj1", "subj2"])[["corr_xy", "rms"]]
    .mean()
    .reset_index()
)

print(f"\n aggregated rows（vid-pair level）: {len(df_agg)}")
print(df_agg.groupby("species").size().rename("samples"))

# ============================================================
# 3. 線形混合効果モデル（LMM）
#    固定効果: species
#    ランダム効果:
#      - vid（同じ動画を見た複数ペアの共変動）
#      - pair_id（同一ペアがvidをまたいで繰り返し登場）
#
#    ※ statsmodelsはcrossed random effectsを直接サポートしないため
#      vid をグループとし pair_id を vc_formula で追加
# ============================================================

def run_lmm(dv: str, data: pd.DataFrame):
    """
    LMMを実行し結果を返す。
    - 固定効果: species（基準カテゴリは最初のアルファベット順）
    - ランダム効果: vid（グループ）+ pair_id（variance component）
    """
    print(f"dependent varialbe: {dv}")

    d = data[["vid", "species", "pair_id", dv]].dropna().copy()
    d["species"] = pd.Categorical(d["species"])


    # --- 参照カテゴリの確認 ---
    ref = d["species"].cat.categories[0]
    print(f"categories（intercept）: {ref}")

    pairs_per_vid = d.groupby("pair_id")["vid"].nunique()
    n_cross = (pairs_per_vid > 1).sum()
    print(f"\n data check")
    print(f"unique pairs: {d['pair_id'].nunique()}")
    print(f"shared pairs: {n_cross} / {len(pairs_per_vid)}")

    crossed = n_cross > 0


    # --- モデル構築 ---
    # vid をランダム切片グループ、pair_id を variance component として追加

    if crossed:
        try:
            model = smf.mixedlm(
                formula=f"{dv} ~ species",
                data=d,
                groups=d["vid"],
                vc_formula={"vid":"0 + C(vid)", "pair_id": "0 + C(pair_id)"},
            )
            result = model.fit(method="lbfgs", disp=False)
            model_type = "crossed (vid + pair_id)"
        except Exception as e:
            print(f"fail to estimate cross: {e}")
            print("only video effect")
            model = smf.mixedlm(
                formula=f"{dv} ~ species",
                data=d,
                groups=d["vid"],
            )
            result = model.fit(method="lbfgs", disp=False)
            model_type = "vid only (fallback)"

    else:
        print("pair_id were nested in videos: only video effect")
        model = smf.mixedlm(
            formula=f"{dv} ~ species",
            data=d,
            groups=d["vid"],
        )
        result = model.fit(method="lbfgs", disp=False)
        model_type = "vid only (nested)"

    print(f"used model: {model_type}")


    # --- 固定効果の表示 ---
    fe = pd.DataFrame({
        "coeffecient": result.fe_params,
        "SE": result.bse_fe,
        "z-value":  result.tvalues,
        "p-value":  result.pvalues[result.fe_params.index],
    })
    fe["statistical significance"] = fe["p-value"].apply(
        lambda p: "***" if p < 0.001 else "**" if p < 0.01
                  else "*" if p < 0.05 else "n.s."
    )
    print("fixed effect")
    print(fe.to_string(float_format="{:.4f}".format))

    # --- ランダム効果の分散 ---
    print(f"\n variance in random effect")

    # variance components（pair_id等）
    if hasattr(result, "vcomp") and result.vcomp is not None and len(result.vcomp) > 0:
        vc_names = model.exog_vc.names
        for name, var in zip(vc_names, result.vcomp):
            print(f"  {name} variance (vcomp): {var:.4f}")
            
    else:
        print("  variance components: none")

    print(f"  residual variance: {result.scale:.4f}")


    # 分散の割合（ICC的な確認）
    total_var = result.scale
    if result.cov_re is not None and result.cov_re.size > 0:
        total_var += result.cov_re.values[0][0]
    if hasattr(result, "vcomp") and result.vcomp is not None and len(result.vcomp) > 0:
        total_var += result.vcomp.sum() 

    print(f"\n variance ratio（total: {total_var:.4f}）")
    if result.cov_re is not None and result.cov_re.size > 0:
        v = result.cov_re.values[0][0]
        print(f"  vid:  {v:.4f} ({v/total_var*100:.1f}%)")
    if hasattr(result, "vcomp") and result.vcomp is not None and len(result.vcomp) > 0:
        vc_names = model.exog_vc.names
        for name, var in zip(vc_names, result.vcomp):
            print(f"  {name}: {var:.4f} ({var/total_var*100:.1f}%)")
    print(f" residual variance: {result.scale:.4f} ({result.scale/total_var*100:.1f}%)")

    # --- 多重比較：Tukey HSD（集約データで実施）---
    print(f"\n【Tukey HSD】")
    tukey = pairwise_tukeyhsd(d[dv], d["species"])
    print(tukey.summary())

    return result, fe

# ============================================================
# 4. 実行
# ============================================================
results = {}
for dv in ["corr_xy", "rms"]:
    res, fe = run_lmm(dv, df_agg)
    results[dv] = {"result": res, "fe": fe}

# ============================================================
# 5. 可視化
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("comparison between species(LMM)", fontsize=14)

#species_order = sorted(df_agg["species"].unique())
species_order = ['adult','infant','toddler','preschooler','marmo','macaq']
palette = "Set2"

for row_i, dv in enumerate(["corr_xy", "rms"]):
    # --- 箱ひげ図（集約後データ）---
    ax1 = axes[row_i][0]
    sns.boxplot(data=df_agg, x="species", y=dv,
                order=species_order, palette=palette, ax=ax1)
    #sns.stripplot(data=df_agg, x="species", y=dv,
                  #order=species_order, color="gray", alpha=0.3,
                  #size=3, jitter=True, ax=ax1)
    ax1.set_title(dv)
    ax1.set_xlabel("Species")

    # --- LMM係数プロット（固定効果）---
    ax2 = axes[row_i][1]
    fe = results[dv]["fe"]
    # Interceptを除いてspecies係数のみプロット
    fe_sp = fe[fe.index.str.startswith("species[T.")].copy()
    #fe_sp["species"] = fe_sp.index.str.replace("species\\[T\\.", "", regex=True).str.rstrip("]")
    fe_sp['species']=['infant','toddler','preschooler','marmo','macaq'][::-1]

    if len(fe_sp) > 0:
        ax2.barh(fe_sp["species"], fe_sp["coeffecient"],
                 xerr=fe_sp["SE"] * 1.96, capsize=4,
                 color=["#66c2a5", "#fc8d62", "#8da0cb"][:len(fe_sp)])
        ax2.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax2.set_title(f"{dv} LMM fixed effect coeffecient(95%CI)\n reference: {species_order[0]}")
        ax2.set_xlabel("coefficient(difference from referred categories)")
    else:
        ax2.set_visible(False)

plt.tight_layout()
plt.savefig("lmm_species_comparison.png", dpi=150, bbox_inches="tight")
print("\n saved the figure as lmm_species_comparison.png")