"""Render measured fixed-panel losses; never read target outcomes."""
from .common import RESULTS, read_json


def render(rows):
    if not rows:
        return None
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    styles = {
        'prefix': ('#666666', '-', 'Shared image prefix'),
        'J_uniform': ('#0072B2', '-', 'Uniform continuation'),
        'J_mined': ('#D55E00', '-', 'Mined continuation'),
        'G30': ('#7B4AB5', ':', 'Cached-feature control'),
    }
    panels = {(r['source'], r['partition']): r for r in
              read_json(RESULTS/'diagnostic_panel_composition.json')['panels']}
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), layout='constrained')
    for column, (source, seed) in enumerate(
            (s, seed) for s in ('44b6', '6bba') for seed in (20260916, 314159)):
        for row, partition in enumerate(('fit', 'calibration')):
            ax = axes[row, column]
            relevant = [r for r in rows if r['source'] == source and
                        r['seed'] == seed and r['partition'] == partition]
            for arm, (color, style, _) in styles.items():
                series = sorted((r for r in relevant if r['arm'] == arm),
                                key=lambda r: r['step'])
                # Connect a continuation to its measured common start without
                # treating the shared prefix as two independent training runs.
                if arm in ('J_uniform', 'J_mined') and series:
                    start = [r for r in relevant if r['arm'] == 'prefix' and
                             r['step'] == 2048]
                    series = start + [r for r in series if r['step'] > 2048]
                if series:
                    ax.plot([r['step'] for r in series],
                            [r['event'] for r in series], color=color,
                            linestyle=style, linewidth=1.6)
            panel = panels[source, partition]
            label = 'Fit panel' if partition == 'fit' else 'Held-source panel'
            ax.set_title(f'{source} · seed {seed}\n{label}: '
                         f'{panel["utility_positive_anchors"]}/32 positive-utility anchors',
                         fontsize=9)
            ax.set_yscale('log')
            ax.set_xlim(0, max(4096, max((r['step'] for r in relevant), default=4096)))
            ax.axvline(2048, color='#999999', linestyle='--', linewidth=.7)
            ax.grid(alpha=.2)
            ax.tick_params(labelsize=8)
            if row == 1:
                ax.set_xlabel('Actual joint updates (head updates for control)', fontsize=8)
            if column == 0:
                ax.set_ylabel('Supported action loss · log scale', fontsize=9)
            if not relevant:
                ax.text(.5, .5, 'Awaiting completed fit receipt', ha='center',
                        transform=ax.transAxes, fontsize=9)
    fig.suptitle('Fixed source diagnostics\n'
                 'Exploratory panels; full source graph scores govern qualification. '
                 'The 6bba held panel measures supported rejection.', fontsize=12)
    fig.legend(handles=[Line2D([0], [0], color=c, linestyle=s, label=label)
                        for c, s, label in styles.values()],
               loc='outside lower center', ncol=4, frameon=False, fontsize=9)
    destination = RESULTS/'training_event_curves.png'
    fig.savefig(destination, dpi=150)
    plt.close(fig)
    return destination
