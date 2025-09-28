from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt

# Stats
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# PDF
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib import colors

# ----------------------------
# Paths & constants
# ----------------------------

INPUT_CSV = Path("data/inputs/survey_results_public.csv")

OUTPUT_DIR = Path("data/outputs")
FIG_DIR = OUTPUT_DIR / "figs"
OUTPUT_PDF = OUTPUT_DIR / "report_so_2017.pdf"

FIG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Column names (2017 schema)
SALARY_COL = "Salary"
JOB_SAT_COL = "JobSatisfaction"
CAREER_SAT_COL = "CareerSatisfaction"
PROFESSIONAL_COL = "Professional"
MOBILE_TYPE_COL = "MobileDeveloperType"
GIF_COL = "PronounceGIF"
TABS_SPACES_COL = "TabsSpaces"
YEARS_PROGRAM_COL = "YearsProgram"

# ----------------------------
# Metadata
# ----------------------------

@dataclass
class Meta:
    group: str
    participants: str
    toolchain: str = "Python 3 (pandas, numpy, matplotlib, scipy, statsmodels, reportlab)"


# ----------------------------
# Helpers
# ----------------------------

def load_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found at {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    return df


def to_numeric(series: pd.Series) -> pd.Series:
    """Coerce possibly string-encoded numerics to float; invalid -> NaN."""
    return pd.to_numeric(series, errors="coerce")


def clean_salary(series: pd.Series) -> pd.Series:
    """Keep positive salaries only; coerce to float."""
    s = to_numeric(series)
    return s[(~s.isna()) & (s > 0)]


def ensure_min_group_size(df: pd.DataFrame, group_col: str, min_n: int = 20) -> pd.DataFrame:
    """Filter groups with at least min_n observations to avoid misleading small-n results."""
    counts = df[group_col].value_counts()
    valid_groups = counts[counts >= min_n].index
    return df[df[group_col].isin(valid_groups)].copy()


def has_variation(series: pd.Series) -> bool:
    """Check if a numeric series has at least two distinct finite values."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    return s.nunique(dropna=True) >= 2


# ----------------------------
# Q1: Tool used
# ----------------------------

def q1_tool_used(meta: Meta) -> str:
    return meta.toolchain


# ----------------------------
# Q2: Salary stats
# ----------------------------

def q2_salary_stats(salary: pd.Series) -> Dict[str, float]:
    valid = clean_salary(salary)
    return {
        "mean": float(valid.mean()) if not valid.empty else float("nan"),
        "median": float(valid.median()) if not valid.empty else float("nan"),
        "q3": float(valid.quantile(0.75)) if not valid.empty else float("nan"),
        "max": float(valid.max()) if not valid.empty else float("nan"),
        "n": int(valid.shape[0]),
    }


# ----------------------------
# Q3: Histogram & density
# ----------------------------

def q3_hist_and_density(salary: pd.Series) -> Tuple[Optional[Path], Optional[Path], str]:
    valid = clean_salary(salary)

    hist_path = None
    density_path = None

    if not valid.empty:
        # Histogram
        fig1 = plt.figure()
        plt.hist(valid, bins=50)
        plt.title("Salary - Histogram")
        plt.xlabel("Annual salary (USD)")
        plt.ylabel("Count")
        hist_path = FIG_DIR / "q3_salary_histogram.png"
        fig1.savefig(hist_path, bbox_inches="tight", dpi=150)
        plt.close(fig1)

        # Density (KDE) with 1–99% trimming
        if valid.size >= 10:
            p01, p99 = np.percentile(valid, [1, 99])
            kde_data = valid[(valid >= p01) & (valid <= p99)]
        else:
            kde_data = valid

        if not kde_data.empty:
            fig2 = plt.figure()
            kde_data.plot(kind="kde")
            plt.title("Salary - Density (trimmed 1%-99%)")
            plt.xlabel("Annual salary (USD)")
            density_path = FIG_DIR / "q3_salary_density.png"
            fig2.savefig(density_path, bbox_inches="tight", dpi=150)
            plt.close(fig2)

    preference_comment = (
        "The density plot (trimmed to 1%-99%) is preferable to visualize the central mass "
        "without extreme-tail distortion; the histogram is useful to see discrete bin counts."
    )
    return hist_path, density_path, preference_comment


# ----------------------------
# Q4: Correlation Job vs Career satisfaction
# ----------------------------

def q4_satisfaction_correlation(df: pd.DataFrame) -> Dict[str, float | str]:
    x = to_numeric(df[JOB_SAT_COL])
    y = to_numeric(df[CAREER_SAT_COL])
    mask = (~x.isna()) & (~y.isna())
    x, y = x[mask], y[mask]

    if x.empty or y.empty or not has_variation(x) or not has_variation(y):
        return {"spearman_rho": float("nan"), "p_value": float("nan"), "interpretation": "insufficient data", "n": int(0)}

    rho, pval = stats.spearmanr(x, y)
    strength = (
        "strong" if abs(rho) >= 0.7 else
        "moderate" if abs(rho) >= 0.4 else
        "weak" if abs(rho) >= 0.2 else
        "very weak/negligible"
    )
    return {"spearman_rho": float(rho), "p_value": float(pval), "interpretation": strength, "n": int(len(x))}


# ----------------------------
# Q5: Which Professional group has highest CareerSatisfaction?
# ----------------------------

def q5_best_professional_group(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    tmp = df[[PROFESSIONAL_COL, CAREER_SAT_COL]].copy()
    tmp[CAREER_SAT_COL] = to_numeric(tmp[CAREER_SAT_COL])
    tmp = tmp.dropna(subset=[PROFESSIONAL_COL, CAREER_SAT_COL])
    if tmp.empty:
        return pd.DataFrame(columns=["mean", "count"]), "No data available."
    grouped = tmp.groupby(PROFESSIONAL_COL)[CAREER_SAT_COL].agg(["mean", "count"]).sort_values("mean", ascending=False)
    if grouped.empty:
        return grouped, "No data available."
    best = grouped.iloc[0]
    best_group = f"Top group: '{grouped.index[0]}' (mean={best['mean']:.2f}, n={int(best['count'])})"
    return grouped, best_group


# ----------------------------
# Q6: Mobile devs (Android vs iOS) - Welch t-test
# ----------------------------

def _select_single_platform(df: pd.DataFrame, platform_name: str) -> pd.Series:
    """
    Return the Salary series for respondents who chose exactly one mobile platform
    and that platform equals `platform_name` (case-insensitive, exact match).
    """
    mobile = df[MOBILE_TYPE_COL].astype("string")  # preserve index alignment
    mobile_clean = mobile.fillna("").str.strip()

    # Rows that answered something (non-empty) and did NOT mark multiple platforms
    mask_single_choice = mobile_clean.ne("") & mobile_clean.str.count(";").eq(0)

    subset = df.loc[mask_single_choice, [MOBILE_TYPE_COL, SALARY_COL]].copy()
    subset[MOBILE_TYPE_COL] = subset[MOBILE_TYPE_COL].astype("string").str.strip()

    # Exact platform match (case-insensitive)
    is_target_platform = subset[MOBILE_TYPE_COL].str.fullmatch(platform_name, case=False, na=False)

    salaries = clean_salary(subset.loc[is_target_platform, SALARY_COL]).dropna()

    return salaries

def q6_android_ios_diff(df: pd.DataFrame) -> Dict[str, float]:
    android = _select_single_platform(df, "Android")
    ios = _select_single_platform(df, "iOS")

    if android.empty or ios.empty:
        return {
            "android_mean": float("nan"),
            "ios_mean": float("nan"),
            "t_stat": float("nan"),
            "p_value": float("nan"),
            "n_android": int(android.size),
            "n_ios": int(ios.size),
        }

    t_stat, p_value = stats.ttest_ind(android, ios, equal_var=False, nan_policy="omit")

    return {
        "android_mean": float(android.mean()),
        "ios_mean": float(ios.mean()),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "n_android": int(android.size),
        "n_ios": int(ios.size),
    }



# ----------------------------
# Q7: GIF pronunciation - boxplots, ANOVA, Tukey HSD
# ----------------------------

def q7_gif_boxplot_anova_tukey(df: pd.DataFrame) -> Tuple[Optional[Path], Dict[str, float], pd.DataFrame]:
    tmp = df[[GIF_COL, SALARY_COL]].copy()
    tmp[SALARY_COL] = clean_salary(tmp[SALARY_COL])
    tmp = tmp.dropna(subset=[GIF_COL, SALARY_COL])
    tmp = ensure_min_group_size(tmp, GIF_COL, min_n=20)

    boxplot_path = None

    if not tmp.empty:
        # Boxplot
        fig = plt.figure()
        tmp.boxplot(column=SALARY_COL, by=GIF_COL, grid=False, rot=20)
        plt.title("Salary by GIF Pronunciation")
        plt.suptitle("")
        plt.xlabel("PronounceGIF")
        plt.ylabel("Annual salary (USD)")
        boxplot_path = FIG_DIR / "q7_boxplot_salary_by_gif.png"
        fig.savefig(boxplot_path, bbox_inches="tight", dpi=150)
        plt.close(fig)

    # ANOVA on log-salary (only if there is variation)
    anova_result = {"F": float("nan"), "p_value": float("nan"), "n": int(tmp.shape[0])}
    tukey_df = pd.DataFrame(columns=["group1", "group2", "meandiff", "p-adj", "lower", "upper", "reject"])

    if not tmp.empty and tmp[GIF_COL].nunique() >= 2:
        tmp = tmp.rename(columns={GIF_COL: "PronounceGIF"})
        tmp["LogSalary"] = np.log(tmp[SALARY_COL])
        if has_variation(tmp["LogSalary"]):
            model = smf.ols("LogSalary ~ C(PronounceGIF)", data=tmp).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            anova_result = {
                "F": float(anova_table["F"].iloc[0]),
                "p_value": float(anova_table["PR(>F)"].iloc[0]),
                "n": int(tmp.shape[0]),
            }
            tukey = pairwise_tukeyhsd(endog=tmp["LogSalary"], groups=tmp["PronounceGIF"], alpha=0.05)
            tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])

    return boxplot_path, anova_result, tukey_df


# ----------------------------
# Q8: Tabs vs Spaces - means, Welch test, confounder check
# ----------------------------

def q8_tabs_spaces(df: pd.DataFrame) -> tuple[dict, dict, Optional[sm.regression.linear_model.RegressionResultsWrapper]]:
    """
    Compute mean salaries by indentation preference (Tabs vs Spaces),
    run a Welch t-test, and (if possible) an OLS on log-salary controlling for YearsProgram.
    Returns: (means_and_counts, welch_test, ols_model_or_none)
    """
    tmp = df[[TABS_SPACES_COL, SALARY_COL, YEARS_PROGRAM_COL]].copy()

    # Clean columns
    tmp[SALARY_COL] = clean_salary(tmp[SALARY_COL])
    tmp[YEARS_PROGRAM_COL] = to_numeric(tmp[YEARS_PROGRAM_COL])
    tmp[TABS_SPACES_COL] = tmp[TABS_SPACES_COL].astype("string").str.strip()

    # Keep ONLY pure Tabs or Spaces (exact match, case-insensitive)
    valid_ts = tmp[TABS_SPACES_COL].str.fullmatch(r"(Tabs|Spaces)", case=False, na=False)
    tmp = tmp[valid_ts].copy()
    tmp["TabsSpaces"] = tmp[TABS_SPACES_COL].str.capitalize()

    # Drop rows without salary
    tmp = tmp.dropna(subset=[SALARY_COL])

    means, counts = {}, {}
    if not tmp.empty:
        means = tmp.groupby("TabsSpaces")[SALARY_COL].mean().to_dict()
        counts = tmp.groupby("TabsSpaces")[SALARY_COL].count().to_dict()

    # Welch test (Spaces vs Tabs)
    spaces = tmp.loc[tmp["TabsSpaces"] == "Spaces", SALARY_COL]
    tabs = tmp.loc[tmp["TabsSpaces"] == "Tabs", SALARY_COL]

    if spaces.empty or tabs.empty:
        welch = {"t_stat": float("nan"), "p_value": float("nan"), "n_spaces": int(spaces.size), "n_tabs": int(tabs.size)}
    else:
        t_stat, p_value = stats.ttest_ind(spaces, tabs, equal_var=False, nan_policy="omit")
        welch = {"t_stat": float(t_stat), "p_value": float(p_value), "n_spaces": int(spaces.size), "n_tabs": int(tabs.size)}

    # OLS with control for YearsProgram (only if we still have both levels and enough data)
    model = None
    reg_df = tmp.dropna(subset=[YEARS_PROGRAM_COL]).copy()
    if reg_df["TabsSpaces"].nunique() >= 2 and len(reg_df) >= 10 and has_variation(reg_df[YEARS_PROGRAM_COL]):
        reg_df["LogSalary"] = np.log(reg_df[SALARY_COL])
        reg_df["TabsSpacesCat"] = reg_df["TabsSpaces"].astype("category")
        if has_variation(reg_df["LogSalary"]):
            model = smf.ols("LogSalary ~ C(TabsSpacesCat) + YearsProgram", data=reg_df).fit()

    return (means | {"counts": counts}), welch, model


# ----------------------------
# Q9: Free analysis (example): salary vs experience nonlinearity
# ----------------------------

def q9_extra_analysis(df: pd.DataFrame) -> Tuple[Optional[Path], Dict[str, float]]:
    tmp = df[[SALARY_COL, YEARS_PROGRAM_COL]].copy()
    tmp[SALARY_COL] = clean_salary(tmp[SALARY_COL])
    tmp[YEARS_PROGRAM_COL] = to_numeric(tmp[YEARS_PROGRAM_COL])
    tmp = tmp.dropna()

    fig_path: Optional[Path] = None

    if not tmp.empty:
        # Scatter with median line per experience year (winsorize to reduce tail splash)
        p99 = tmp[SALARY_COL].quantile(0.99) if tmp.shape[0] >= 5 else tmp[SALARY_COL].max()
        plot_df = tmp[tmp[SALARY_COL] <= p99].copy()

        if not plot_df.empty:
            medians = plot_df.groupby(YEARS_PROGRAM_COL)[SALARY_COL].median().reset_index()

            fig = plt.figure()
            plt.scatter(plot_df[YEARS_PROGRAM_COL], plot_df[SALARY_COL], alpha=0.2, s=8)
            if not medians.empty:
                plt.plot(medians[YEARS_PROGRAM_COL], medians[SALARY_COL])
            plt.title("Salary vs. Years of Programming (median line)")
            plt.xlabel("YearsProgram")
            plt.ylabel("Annual salary (USD)")
            fig_path = FIG_DIR / "q9_salary_vs_years_program.png"
            fig.savefig(fig_path, bbox_inches="tight", dpi=150)
            plt.close(fig)

    # Simple log-quadratic fit as summary (only if we have variation)
    beta_years = float("nan")
    beta_years_sq = float("nan")
    r2 = float("nan")

    if not tmp.empty and has_variation(tmp[YEARS_PROGRAM_COL]) and has_variation(np.log(tmp[SALARY_COL])):
        tmp["LogSalary"] = np.log(tmp[SALARY_COL])
        # Fit
        model = smf.ols("LogSalary ~ YearsProgram + I(YearsProgram**2)", data=tmp).fit()
        # Handle possible key name variants for the squared term
        keys = list(model.params.index)
        sq_key = next((k for k in keys if "YearsProgram" in k and "^2" in k or "I(YearsProgram**2)" in k), None)
        beta_years = float(model.params.get("YearsProgram", np.nan))
        beta_years_sq = float(model.params.get(sq_key, np.nan)) if sq_key else float("nan")
        r2 = float(model.rsquared)

    summary = {"beta_years": beta_years, "beta_years_sq": beta_years_sq, "r2": r2}
    return fig_path, summary


# ----------------------------
# PDF assembly
# ----------------------------

def build_pdf(meta: Meta,
              q2: Dict[str, float],
              q3_paths: Tuple[Optional[Path], Optional[Path], str],
              q4: Dict[str, float | str],
              q5_table: pd.DataFrame, q5_text: str,
              q6: Dict[str, float],
              q7_box_path: Optional[Path], q7_anova: Dict[str, float], q7_tukey: pd.DataFrame,
              q8_means: Dict[str, float], q8_welch: Dict[str, float], q8_model: Optional[sm.regression.linear_model.RegressionResultsWrapper],
              q9_path: Optional[Path], q9_summary: Dict[str, float]) -> None:

    doc = SimpleDocTemplate(str(OUTPUT_PDF), pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # Header
    story.append(Paragraph("Stack Overflow Developer Survey 2017 — Report", styles['Title']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Group: {meta.group}<br/>Participants: {meta.participants}", styles['Normal']))

    # Q1
    story.append(Spacer(1, 12))
    story.append(Paragraph("1) Tool used", styles['Heading2']))
    story.append(Paragraph(meta.toolchain, styles['Normal']))

    # Q2
    story.append(Spacer(1, 12))
    story.append(Paragraph("2) Salary statistics", styles['Heading2']))
    data = [["Metric", "Value", "N"],
            ["Mean", f"{q2['mean']:.2f}" if not np.isnan(q2['mean']) else "NA", f"{q2['n']}"],
            ["Median", f"{q2['median']:.2f}" if not np.isnan(q2['median']) else "NA", ""],
            ["3rd Quartile", f"{q2['q3']:.2f}" if not np.isnan(q2['q3']) else "NA", ""],
            ["Max", f"{q2['max']:.2f}" if not np.isnan(q2['max']) else "NA", ""]]
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('GRID', (0,0), (-1,-1), 0.25, colors.black)
    ]))
    story.append(table)

    # Q3
    hist_path, density_path, pref_text = q3_paths
    story.append(Spacer(1, 12))
    story.append(Paragraph("3) Histogram and density plot for Salary", styles['Heading2']))
    if hist_path is not None:
        story.append(Image(str(hist_path), width=480, height=320))
    if density_path is not None:
        story.append(Spacer(1, 6))
        story.append(Image(str(density_path), width=480, height=320))
    story.append(Paragraph(f"Preference: {pref_text}", styles['Normal']))

    # Q4
    story.append(Spacer(1, 12))
    story.append(Paragraph("4) Correlation between Job and Career Satisfaction", styles['Heading2']))
    story.append(Paragraph(
        f"Spearman’s ρ = {q4['spearman_rho']:.3f} | p = {q4['p_value']:.3g} | N = {q4['n']} "
        f"→ {q4['interpretation']}.", styles['Normal'])
    )

    # Q5
    story.append(Spacer(1, 12))
    story.append(Paragraph("5) Highest career satisfaction by Professional group", styles['Heading2']))
    if not q5_table.empty:
        q5_tbl = Table([["Professional", "Mean", "Count"]] + [[idx, f"{row['mean']:.2f}", f"{int(row['count'])}"]
                                                              for idx, row in q5_table.iterrows()])
        q5_tbl.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
                                    ('GRID', (0,0), (-1,-1), 0.25, colors.black)]))
        story.append(q5_tbl)
    story.append(Paragraph(q5_text, styles['Normal']))

    # Q6
    story.append(Spacer(1, 12))
    story.append(Paragraph("6) Android vs iOS salary comparison (single-platform devs)", styles['Heading2']))
    story.append(Paragraph(
        f"Android mean = {q6['android_mean']:.2f} (n={q6['n_android']}), "
        f"iOS mean = {q6['ios_mean']:.2f} (n={q6['n_ios']}). "
        f"Welch t-test: t = {q6['t_stat']:.3f}, p = {q6['p_value']:.3g}.", styles['Normal'])
    )

    # Q7
    story.append(Spacer(1, 12))
    story.append(Paragraph("7) Salary by GIF pronunciation: boxplots, ANOVA, and multiple comparisons", styles['Heading2']))
    if q7_box_path is not None:
        story.append(Image(str(q7_box_path), width=480, height=320))
    story.append(Paragraph(f"ANOVA on log-salary: F = {q7_anova['F']:.3f}, p = {q7_anova['p_value']:.3g}, N = {q7_anova['n']}.", styles['Normal']))

    if not q7_tukey.empty:
        tukey_table = Table([tukey_colnames(q7_tukey)] + q7_tukey.values.tolist())
        tukey_table.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
                                         ('GRID', (0,0), (-1,-1), 0.25, colors.black)]))
        story.append(tukey_table)

    # Q8
    story.append(Spacer(1, 12))
    story.append(Paragraph("8) Tabs vs Spaces", styles['Heading2']))
    means_counts = q8_means.copy()
    counts = means_counts.pop("counts", {})
    if means_counts:
        rows = [["Group", "Mean salary", "Count"]] + [[k, f"{v:.2f}", str(counts.get(k, ''))] for k, v in means_counts.items()]
        means_table = Table(rows)
        means_table.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
                                         ('GRID', (0,0), (-1,-1), 0.25, colors.black)]))
        story.append(means_table)
    story.append(Paragraph(
        f"Welch t-test (Spaces vs Tabs): t = {q8_welch['t_stat']:.3f}, p = {q8_welch['p_value']:.3g} "
        f"(n_spaces={q8_welch['n_spaces']}, n_tabs={q8_welch['n_tabs']}).", styles['Normal'])
    )

    story.append(Paragraph("Confounder test: OLS on log-salary with YearsProgram control.", styles['Normal']))
    if q8_model is not None:
        coef_rows = [["Variable", "Coef.", "p-value"]] + [
            [name, f"{coef:.3f}", f"{pval:.3g}"]
            for name, coef, pval in zip(q8_model.params.index, q8_model.params.values, q8_model.pvalues.values)
        ]
        coef_table = Table(coef_rows)
        coef_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
            ('GRID', (0,0), (-1,-1), 0.25, colors.black)
        ]))
        story.append(coef_table)
    else:
        story.append(Paragraph(
            "OLS skipped: after cleaning there was not enough variation (both groups required) "
            "or insufficient rows with YearsProgram.", styles['Italic'])
        )

    story.append(Paragraph(
        "Interpretation of the blog’s language plot: languages may proxy seniority, geography, or industry; "
        "thus salary differences by language likely reflect ecosystem/market effects rather than causal impact of indentation style.",
        styles['Normal'])
    )

    # Q9
    story.append(Spacer(1, 12))
    story.append(Paragraph("9) Extra analysis: nonlinearity of salary vs experience", styles['Heading2']))
    if q9_path is not None:
        story.append(Image(str(q9_path), width=480, height=320))
    story.append(Paragraph(
        f"Log-quadratic model: β_years = {q9_summary['beta_years']:.3f}, "
        f"β_years^2 = {q9_summary['beta_years_sq']:.3f}, R² = {q9_summary['r2']:.3f}.", styles['Normal'])
    )

    doc.build(story)


def tukey_colnames(df: pd.DataFrame) -> list[str]:
    return list(df.columns)


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    # --- metadata ---
    meta = Meta(group="Meu Grupo", participants="Participante 1, Participante 2")

    # --- data ---
    df = load_data(INPUT_CSV)

    # Q2
    q2 = q2_salary_stats(df.get(SALARY_COL, pd.Series(dtype=float)))

    # Q3
    q3_paths = q3_hist_and_density(df.get(SALARY_COL, pd.Series(dtype=float)))

    # Q4
    q4 = q4_satisfaction_correlation(df)

    # Q5
    q5_table, q5_text = q5_best_professional_group(df)

    # Q6
    q6 = q6_android_ios_diff(df)

    # Q7
    q7_box_path, q7_anova, q7_tukey = q7_gif_boxplot_anova_tukey(df)

    # Q8
    q8_means, q8_welch, q8_model = q8_tabs_spaces(df)

    # Q9
    q9_path, q9_summary = q9_extra_analysis(df)

    # PDF
    build_pdf(meta, q2, q3_paths, q4, q5_table, q5_text, q6,
              q7_box_path, q7_anova, q7_tukey,
              q8_means, q8_welch, q8_model,
              q9_path, q9_summary)

    print("✅ Report generated at data/outputs/report_so_2017.pdf")


if __name__ == "__main__":
    main()
