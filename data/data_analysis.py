import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib as mpl

# ---------------------------------------------------------------------------
# Plotting / display defaults
# ---------------------------------------------------------------------------
# This sets some nicer defaults for plotting.
# This must be run in a separate cell from importing matplotlib due to a bug.
params = {'legend.fontsize': 'large',
          'figure.figsize': (11.0, 11.0),
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'xx-large',
          'xtick.labelsize': 'large',
          'ytick.labelsize': 'large'}
mpl.rcParams.update(params)

# Don't truncate dataframes horizontally when viewing.
pd.options.display.max_columns = 200
pd.options.display.width = 200


def _show(obj):
    """display() in Jupyter, else print() — so this runs as a script too."""
    try:
        display(obj)  # noqa: F821  (provided by IPython)
    except NameError:
        print(obj)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
# Analyze the RAW data so cleaning choices are made before any transformation.
# NOTE: data-processor.py derives the target with `x == 'Default'`, but the
# actual LoanStatus values are 'Defaulted' / 'Chargedoff' (no bare 'Default'),
# so that comparison yields an all-zero target. We treat both as "bad" here.
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
RAW_PATH = os.path.join(HERE, "raw", "prosperLoanData.csv")
BAD_STATUSES = {"Defaulted", "Chargedoff"}
TARGET = "is_bad"


def load_data(path=RAW_PATH, derive_target=True):
    """Load the raw loan data and attach a binary default target (is_bad)."""
    df = pd.read_csv(path)
    if derive_target:
        df = df[df["LoanStatus"].notna()].copy()
        df[TARGET] = df["LoanStatus"].isin(BAD_STATUSES).astype(int)
    return df


# ---------------------------------------------------------------------------
# Iteration logging — snapshot a result table to disk
# ---------------------------------------------------------------------------
# Pass name=... to any analysis function below and it writes the table it
# produced to data-cleaning-iterations/ so you can diff before/after a cleaning
# step. The timestamp makes every file unique, so nothing is overwritten.
ITER_DIR = os.path.join(HERE, "data-cleaning-iterations")


def _save_table(table, name, label):
    """Write `table` to data-cleaning-iterations/<name>_<label>_<timestamp>.csv."""
    os.makedirs(ITER_DIR, exist_ok=True)
    safe = re.sub(r'[^0-9A-Za-z]+', '_', f"{name}_{label}").strip('_')
    path = os.path.join(ITER_DIR, f"{safe}_{datetime.now():%Y%m%d_%H%M%S}.csv")
    table.to_csv(path)
    print(f"saved -> {path}")
    return path


# ---------------------------------------------------------------------------
# Whole-frame overview
# ---------------------------------------------------------------------------
def summarize_dataframe(df, name=None):
    """Per-column dtype, missing count, unique count + full describe() stats.

    Pass name=... to also snapshot the table to data-cleaning-iterations/.
    """
    overview = pd.DataFrame({
        'Variable Name': df.columns,
        'Data Type': df.dtypes,
        'Missing Values': df.isnull().sum(),
        'Unique Values': [df[col].nunique() for col in df.columns],
    }).set_index('Variable Name')
    # datetime_is_numeric was removed in pandas 2.0; describe() handles
    # datetimes natively now, so the old kwarg is gone.
    table = pd.concat([overview, df.describe(include='all').transpose()], axis=1).fillna("")
    if name:
        _save_table(table, name, 'summarize')
    with pd.option_context("display.max_rows", 1000):
        _show(table)
    return table


def missing_report(df, name=None):
    """Missing count/% per column, worst first — the drop-vs-impute view."""
    table = pd.DataFrame({
        'Missing': df.isnull().sum(),
        'Missing %': (df.isnull().mean() * 100).round(2),
        'Dtype': df.dtypes.astype(str),
        'Unique': df.nunique(),
    }).sort_values('Missing %', ascending=False)
    if name:
        _save_table(table, name, 'missing')
    return table


# ---------------------------------------------------------------------------
# Pivot / "view by" the default target
# ---------------------------------------------------------------------------
# For arbitrary cross-tabs just use pandas directly, e.g.
#   pd.pivot_table(df, index='IncomeRange', columns='IsBorrowerHomeowner',
#                  values='is_bad', aggfunc='mean')
#   pd.crosstab(df['EmploymentStatus'], df['IsBorrowerHomeowner'], normalize='index')
# The helpers below add the one thing pandas doesn't: bad-rate + volume
# together, with NaNs kept visible.

def target_rate(df, col, target=TARGET, bins=10, min_count=1, name=None):
    """Default rate + volume per level of `col`, sorted by risk.

    Numeric columns with many values are quantile-binned; NaNs become their
    own '<missing>' bucket so missingness is never hidden. This is the core
    "is this column predictive / how should I bin or group it?" view.
    Pass name=... to also snapshot the table to data-cleaning-iterations/.
    """
    s = df[col]
    if pd.api.types.is_numeric_dtype(s) and s.nunique() > bins:
        key = pd.qcut(s, q=bins, duplicates='drop').cat.add_categories(['<missing>']).fillna('<missing>')
        sort_by_risk = False  # keep bin order to read monotonicity
    else:
        key = s.fillna('<missing>')
        sort_by_risk = True

    g = df.groupby(key, observed=False)[target]
    out = pd.DataFrame({
        'Count': g.size(),
        'Bads': g.sum(),
        'Bad rate %': (g.mean() * 100).round(2),
    })
    out['Share %'] = (out['Count'] / len(df) * 100).round(2)
    out = out[out['Count'] >= min_count]
    if sort_by_risk:
        out = out.sort_values('Bad rate %', ascending=False)
    if name:
        _save_table(out, name, f'target_rate_{col}')
    return out


def woe_iv(df, col, target=TARGET, bins=10, name=None):
    """WOE table + total Information Value (out.attrs['IV_total']) for `col`.

    IV ranks predictive power model-free:
        <0.02 useless | 0.02-0.1 weak | 0.1-0.3 medium | 0.3-0.5 strong | >0.5 suspicious/leakage
    Pass name=... to also snapshot the table to data-cleaning-iterations/.
    """
    s = df[col]
    if pd.api.types.is_numeric_dtype(s) and s.nunique() > bins:
        key = pd.qcut(s, q=bins, duplicates='drop').cat.add_categories(['<missing>']).fillna('<missing>')
    else:
        key = s.fillna('<missing>')

    g = df.groupby(key, observed=False)[target]
    out = pd.DataFrame({'Count': g.size(), 'Bads': g.sum()})
    out['Goods'] = out['Count'] - out['Bads']
    # +0.5 avoids divide-by-zero / log(0) in empty buckets.
    dist_bad = (out['Bads'] + 0.5) / out['Bads'].sum()
    dist_good = (out['Goods'] + 0.5) / out['Goods'].sum()
    out['Bad rate %'] = (out['Bads'] / out['Count'] * 100).round(2)
    out['WOE'] = np.log(dist_good / dist_bad).round(4)
    out['IV'] = ((dist_good - dist_bad) * out['WOE']).round(4)
    out.attrs['IV_total'] = round(out['IV'].sum(), 4)
    if name:
        _save_table(out, name, f'woe_{col}')
    return out


def iv_ranking(df, target=TARGET, bins=10, exclude=None, name=None):
    """Rank every column by Information Value — what to clean carefully first."""
    exclude = set(exclude or []) | {target}
    rows = []
    for col in df.columns:
        if col in exclude:
            continue
        try:
            rows.append({'Column': col, 'IV': woe_iv(df, col, target, bins).attrs['IV_total']})
        except Exception as exc:  # all-unique IDs etc. — note and keep scanning
            rows.append({'Column': col, 'IV': np.nan, 'note': str(exc)[:60]})
    table = pd.DataFrame(rows).set_index('Column').sort_values('IV', ascending=False)
    if name:
        _save_table(table, name, 'iv_ranking')
    return table


# ---------------------------------------------------------------------------
# Default report when run as a script (not when imported).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = load_data()
    print(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Overall bad rate: {df[TARGET].mean() * 100:.2f}%\n")
    print("=== Missing-value report (top 20) ===")
    _show(missing_report(df).head(20))
    print("\n=== Information Value ranking (top 25) ===")
    _show(iv_ranking(df).head(25))
