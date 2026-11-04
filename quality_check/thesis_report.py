"""Thesis tables and figures for one decisions folder (no API calls; the dataset is not touched).

Usage: python thesis_report.py decisions/<folder>
Reads the folder's decisions.csv and policy.json, its run's judgments.jsonl and run.json, and
thesis_data/mother_metadata.csv (from extract_metadata.py). Writes thesis_data/<folder>/:
  master_table.csv   one row per Mother image: metadata + every Jev answer + decision
  tables/*.csv       the numbers behind each table and figure
  figures/*.png      300 dpi figures; each one's numbers are in tables/
  README.md          what every file holds
Needs pandas and matplotlib.
"""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from typesafe_image_filter import CANDIDATES, CLOUDS_REMOVE, DEFECT_REMOVE, DEFECTS, HERE

# Reference palette from the dataviz skill (light surface), validated for these four adjacent slots.
SURFACE, INK, INK_2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
SLOTS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
REASONS = [f'defect_{name}' for name in DEFECTS] + ['mostly_clouds']
SCORES = dict(zip(REASONS, [*DEFECTS, 'clouds_over_half']))
CUTOFF = {**{f'defect_{name}': cut for name, cut in DEFECT_REMOVE.items()}, 'mostly_clouds': CLOUDS_REMOVE}
SOURCE_ORDER = ['aerial_rgb', 'earthview', 'skysat_raw_window', 'maxar']
plt.rcParams.update({
    'font.family': ['Segoe UI', 'DejaVu Sans'], 'font.size': 9, 'figure.facecolor': SURFACE,
    'axes.facecolor': SURFACE, 'axes.edgecolor': AXIS, 'axes.labelcolor': INK_2, 'axes.titlecolor': INK,
    'axes.titlesize': 10, 'axes.grid': True, 'axes.axisbelow': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK, 'axes.spines.top': False,
    'axes.spines.right': False, 'legend.frameon': False, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})


def flatten(answers):
    """Every Jev answer as flat columns: Noul values, Score/Choice values, confidence and probabilities."""
    out = {}
    for key, answer in answers.items():
        if answer['type'] == 'noul':
            out[key] = answer['noul']
        elif answer['type'] == 'score':
            out.update({f'{key}_score': answer['score'], f'{key}_confidence': answer['confidence']})
            out.update({f'{key}_p{level}': p for level, p in answer['probabilities'].items()})
        else:
            out.update({key: answer['choice'], f'{key}_confidence': answer['confidence']})
            out.update({f'{key}_p_{option}': p for option, p in answer['probabilities'].items()})
    return out


def has(series, label):
    return series.fillna('').str.split(';').apply(lambda labels: label in labels)


def breakdown(df, column):
    """Images, removals per reason and kept candidates per group of `column`."""
    groups = df.groupby(column, dropna=False)
    table = pd.DataFrame({'images': groups.size(), 'removed': groups['removed'].sum()})
    table['removed_pct'] = (100 * table['removed'] / table['images']).round(2)
    for reason in REASONS:
        table[reason] = groups[reason].sum()
    for candidate in CANDIDATES:
        table[f'kept_{candidate}'] = groups[f'kept_{candidate}'].sum()
    return table.sort_values('images', ascending=False).reset_index()


def composition(df, column):
    before = df[column].value_counts()
    after = df.loc[~df['removed'], column].value_counts()
    table = pd.DataFrame({'images_before': before, 'images_after': after}).fillna(0).astype(int)
    table['share_before_pct'] = (100 * table['images_before'] / table['images_before'].sum()).round(2)
    table['share_after_pct'] = (100 * table['images_after'] / table['images_after'].sum()).round(2)
    return table.sort_values('images_before', ascending=False).rename_axis(column).reset_index()


def label_bars(ax, bars, fmt):
    for bar in bars:
        if bar.get_height() > 0:
            ax.annotate(fmt(bar.get_height()), (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 2), textcoords='offset points', ha='center', va='bottom', fontsize=7, color=INK_2)


def main(folder):
    folder = folder.resolve()
    policy = json.loads((folder / 'policy.json').read_text(encoding='utf-8'))
    run_dir = Path(policy['run_dir'])
    run = json.loads((run_dir / 'run.json').read_text(encoding='utf-8'))
    out = HERE / 'thesis_data' / folder.name
    (out / 'tables').mkdir(parents=True, exist_ok=True)
    (out / 'figures').mkdir(exist_ok=True)

    meta = pd.read_csv(HERE / 'thesis_data' / 'mother_metadata.csv', dtype=str)
    decisions = pd.read_csv(folder / 'decisions.csv', dtype={'remove_reasons': str, 'candidates': str})
    decisions['sample_id'] = decisions['image'].str.removesuffix('.npz')
    jev = []
    for line in (run_dir / 'judgments.jsonl').read_text(encoding='utf-8').splitlines():
        record = json.loads(line)
        jev.append(dict(sample_id=record['sample_id'], jev_model=record['model'], jev_request_id=record['request_id'],
                        jev_answered_utc=record['answered_utc'], jev_input_tokens=record['input_tokens'],
                        png_sha256=record['png_sha256'], **flatten(record['answers'])))
    jev = pd.DataFrame(jev)
    master = (meta.merge(decisions[['sample_id', 'decision', 'remove_reasons', 'candidates', 'clouds_over_half']],
                         on='sample_id', how='left')
                  .merge(jev, on='sample_id', how='left'))
    master['decision'] = master['decision'].fillna('not_judged')
    master.to_csv(out / 'master_table.csv', index=False)

    df = master[master['decision'] != 'not_judged'].copy()
    df['removed'] = df['decision'] == 'remove'
    for reason in REASONS:
        df[reason] = has(df['remove_reasons'], reason)
    for candidate in CANDIDATES:
        df[f'kept_{candidate}'] = ~df['removed'] & has(df['candidates'], candidate)
    df['source_kind'] = pd.Categorical(df['source_kind'], SOURCE_ORDER, ordered=True)

    tokens = int(df['jev_input_tokens'].sum())
    overview = {
        'images_in_mother': len(master), 'images_judged': len(df), 'images_not_judged': int((master['decision'] == 'not_judged').sum()),
        'images_listed_for_removal': int(df['removed'].sum()), 'removed_pct': round(100 * df['removed'].mean(), 2),
        'images_kept': int((~df['removed']).sum()),
        **{f'flagged_{reason}': int(df[reason].sum()) for reason in REASONS},
        **{f'kept_candidate_{c}': int(df[f'kept_{c}'].sum()) for c in CANDIDATES},
        'model': ';'.join(sorted(df['jev_model'].dropna().unique())), 'input_tokens': tokens,
        'tokens_per_image_mean': round(tokens / len(df), 1),
        'estimated_cost_usd': round(tokens / 1e6 * run['price_usd_per_mtok'], 4),
        'run': run_dir.name, 'run_status': run['status'], 'decisions': folder.name,
        'run_sessions': len(run['sessions']),
        'run_wall_time_h': round(sum((pd.Timestamp(s['ended_utc']) - pd.Timestamp(s['started_utc'])).total_seconds()
                                     for s in run['sessions'] if 'ended_utc' in s) / 3600, 3),
        'cutoffs': json.dumps(CUTOFF)}
    pd.Series(overview, name='value').rename_axis('measure').to_csv(out / 'tables' / 't1_overview.csv')
    for name, column in [('t2_by_split', 'split'), ('t3_by_source', 'source_kind'), ('t4_by_country', 'country'),
                         ('t5_by_jev_scene', 'scene')]:
        breakdown(df, column).to_csv(out / 'tables' / f'{name}.csv', index=False)
    overlap = pd.DataFrame({a: [int((df[a] & df[b]).sum()) for b in REASONS] for a in REASONS}, index=REASONS)
    overlap.rename_axis('reason').to_csv(out / 'tables' / 't6_reason_overlap.csv')
    for column in ('source_kind', 'split', 'scene'):
        composition(df, column).to_csv(out / 'tables' / f't7_composition_{column}.csv', index=False)

    cuts = np.round(np.arange(0.30, 0.951, 0.05), 2)
    sensitivity = pd.DataFrame({'cutoff': cuts})
    for reason, score in SCORES.items():
        values = df[score].astype(float)
        sensitivity[reason] = [int((values >= c).sum() if reason == 'mostly_clouds' else (values > c).sum()) for c in cuts]
    sensitivity.to_csv(out / 'tables' / 't8_threshold_sensitivity.csv', index=False)
    bins = np.round(np.arange(0, 1.0001, 0.05), 2)
    histogram = pd.DataFrame({'bin_low': bins[:-1], 'bin_high': bins[1:]})
    for score in SCORES.values():
        histogram[score] = np.histogram(df[score].astype(float), bins=bins)[0]
    histogram.to_csv(out / 'tables' / 't9_score_histograms.csv', index=False)

    # Figure 1: score distributions with the cut-offs (small multiples, one series each).
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 4.4), sharex=True)
    for ax, (reason, score) in zip(axes.flat, SCORES.items()):
        counts = histogram[score].where(histogram[score] > 0)  # empty bins are not drawn on the log scale
        ax.bar(histogram['bin_low'] + 0.025, counts, width=0.05, color=SLOTS[0], edgecolor=SURFACE, linewidth=1)
        ax.set_yscale('log')
        ax.set_ylim(bottom=0.7)
        ax.axvline(CUTOFF[reason], color=INK_2, linestyle='--', linewidth=1)
        ax.annotate(f'cut-off {CUTOFF[reason]}', (CUTOFF[reason], 1), xycoords=('data', 'axes fraction'),
                    xytext=(3, -10), textcoords='offset points', fontsize=7, color=INK_2)
        ax.set_title(score.replace('_', ' '), loc='left')
        ax.set_xlim(0, 1)
    for ax in axes[1]:
        ax.set_xlabel('Jev probability')
    for ax in axes[:, 0]:
        ax.set_ylabel('images (log scale)')
    fig.suptitle(f'Distribution of Jev answers for the removal criteria (n = {len(df):,})', x=0.02, ha='left', color=INK)
    fig.tight_layout()
    fig.savefig(out / 'figures' / 'f1_score_distributions.png')
    plt.close(fig)

    # Figure 2: share of each source flagged by each criterion.
    by_source = breakdown(df, 'source_kind').set_index('source_kind').reindex(SOURCE_ORDER)
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    width = 0.2
    x = np.arange(len(SOURCE_ORDER))
    for i, reason in enumerate(REASONS):
        share = 100 * by_source[reason] / by_source['images']
        bars = ax.bar(x + (i - 1.5) * width, share, width * 0.9, color=SLOTS[i], label=SCORES[reason].replace('_', ' '))
        label_bars(ax, bars, lambda v: f'{v:.1f}')
    ax.set_xticks(x, [f'{s}\n(n = {int(n):,})' for s, n in zip(SOURCE_ORDER, by_source['images'])])
    ax.set_ylabel('% of the source\'s images flagged')
    ax.legend(ncols=4, loc='upper left', bbox_to_anchor=(0, 1.15), fontsize=8)
    ax.grid(axis='x', visible=False)
    fig.savefig(out / 'figures' / 'f2_flagged_by_source.png')
    plt.close(fig)

    # Figure 3: threshold sensitivity, each criterion on its own.
    fig, ax = plt.subplots(figsize=(6.3, 3.2))
    for i, reason in enumerate(REASONS):
        ax.plot(sensitivity['cutoff'], sensitivity[reason], color=SLOTS[i], linewidth=2, label=SCORES[reason].replace('_', ' '))
        chosen = sensitivity.loc[np.isclose(sensitivity['cutoff'], CUTOFF[reason]), reason]
        if len(chosen):
            ax.plot([CUTOFF[reason]], chosen, 'o', markersize=8, color=SLOTS[i], markeredgecolor=SURFACE, markeredgewidth=2)
            ax.annotate(f'{int(chosen.iloc[0]):,}', (CUTOFF[reason], chosen.iloc[0]), xytext=(6, 4),
                        textcoords='offset points', fontsize=7, color=INK_2)
    ax.set_xlabel('cut-off (dots: chosen cut-off and resulting count)')
    ax.set_ylabel('images flagged')
    ax.legend(ncols=4, loc='upper left', bbox_to_anchor=(0, 1.15), fontsize=8)
    fig.savefig(out / 'figures' / 'f3_threshold_sensitivity.png')
    plt.close(fig)

    # Figure 4: dataset composition before and after removing the listed images.
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.0), gridspec_kw={'width_ratios': [4, 9]})
    for ax, column in zip(axes, ('source_kind', 'scene')):
        table = composition(df, column)
        x = np.arange(len(table))
        for i, (col, label) in enumerate([('share_before_pct', 'all images'), ('share_after_pct', 'after removal')]):
            bars = ax.bar(x + (i - 0.5) * 0.4, table[col], 0.36, color=SLOTS[i], label=label)
            label_bars(ax, bars, lambda v: f'{v:.0f}')
        ax.set_xticks(x, table[column].astype(str).str.replace('_', ' '), rotation=45, ha='right', fontsize=7)
        ax.set_title(column.replace('_', ' '), loc='left')
        ax.grid(axis='x', visible=False)
    axes[0].set_ylabel('% of images')
    axes[1].legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'figures' / 'f4_composition_before_after.png')
    plt.close(fig)

    (out / 'README.md').write_text(f"""# Thesis data for `{folder.name}`

Made by `thesis_report.py` from `decisions/{folder.name}` (run `runs/{run_dir.name}`) and
`thesis_data/mother_metadata.csv`. No API calls; the dataset was only read. Cut-offs: {json.dumps(CUTOFF)}.

- `master_table.csv`: one row per Mother image. Columns:
  - the metadata columns (see `../mother_metadata.json` for definitions);
  - `decision` (remove / keep / not_judged), `remove_reasons`, `candidates` and `clouds_over_half`;
  - every Jev answer: Noul probabilities, Score `_score` / `_confidence` / `_pN` level probabilities, and
    Choice value / `_confidence` / `_p_<option>`;
  - `jev_model`, `jev_request_id`, `jev_answered_utc`, `jev_input_tokens`, and `png_sha256` (the exact image sent).
- `tables/t1_overview.csv`: headline numbers, tokens, cost and run time.
- `tables/t2..t5`: images, removals per reason and kept candidates, by split, source, country (ISO code) and Jev scene.
- `tables/t6_reason_overlap.csv`: images flagged by both reasons (diagonal = each reason's total).
- `tables/t7_composition_*.csv`: counts and shares before and after removal.
- `tables/t8_threshold_sensitivity.csv`: images each criterion would flag at other cut-offs.
- `tables/t9_score_histograms.csv`: distribution of each criterion (bins of 0.05).
- Figures:
  - `figures/f1`: score distributions (data in t9);
  - `figures/f2`: flagged share by source (t3);
  - `figures/f3`: threshold sensitivity (t8);
  - `figures/f4`: composition (t7).
""", encoding='utf-8')
    print(f'{len(df)} judged of {len(master)} -> {out}')


if __name__ == '__main__':
    main(Path(sys.argv[1]))
