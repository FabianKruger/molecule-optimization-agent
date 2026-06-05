import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = ROOT / 'analysis' / 'paper' / 'figures'
MPRO_EXPLANATION_DIR = ROOT / 'data' / 'results' / 'boltz2_mpro' / 'explanation'
TRIB2_EXPLANATION_DIR = ROOT / 'data' / 'results' / 'boltz2_trib2' / 'explanation'
HARD_EXPLANATION_DIR = ROOT / 'data' / 'results' / 'boltz2_hard' / 'explanation'
HARD_BASELINE_PATH = ROOT / 'data' / 'molecules' / 'baseline_molecules_boltz2_pred.csv'


def _load_target_traces(conversation_dir: Path, model_label: str, max_it: int = 51):
    trace_frames = []
    for trace_file in sorted(conversation_dir.glob('*.json')):
        with trace_file.open(encoding='utf-8') as f:
            conversation = json.load(f)

        df_trace = extract_iteration_score_table(conversation, max_it=max_it)
        df_trace['value'] = pd.to_numeric(df_trace['value'], errors='coerce').fillna(0.0).cummax()
        df_trace['model'] = model_label
        trace_frames.append(df_trace[['iteration', 'value', 'model']])

    if not trace_frames:
        return pd.DataFrame(columns=['iteration', 'value', 'model'])

    return pd.concat(trace_frames, ignore_index=True)


def extract_iteration_score_table(conversation, max_it=51):
    trace = conversation.get('trace', [])
    if not trace:
        return pd.DataFrame({'iteration': range(1, max_it + 1), 'value': [float('nan')] * max_it})

    df = pd.DataFrame(trace)
    score_col = 'original_score' if 'original_score' in df.columns else 'score'
    if score_col not in df.columns:
        raise KeyError("Expected either 'original_score' or 'score' in trace entries")

    df = df[['iteration', score_col]].rename(columns={score_col: 'value'})
    df['iteration'] = pd.to_numeric(df['iteration'], errors='coerce')
    df = df.dropna(subset=['iteration'])
    df['iteration'] = df['iteration'].astype(int)

    all_iterations = pd.DataFrame({'iteration': range(1, max_it + 1)})
    return all_iterations.merge(df, on='iteration', how='left')

def plotting(
    df_plot,
    name,
    model_colors,
    legend_labels,
    legend_order,
    legend_name,
    SHOW_LEGEND=True,
    reference_lines=None,
):
    # Set publication-quality style
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.figsize': (6, 4),
        'axes.linewidth': 1.2,
        'grid.linewidth': 0.6,
        'lines.linewidth': 2.5,
    })

    # Create figure with better proportions
    _, ax = plt.subplots(figsize=(7, 4), dpi=150)


    # Plot with seaborn (mean + 95% bootstrap CI)
    sns.lineplot(
        data=df_plot, 
        x="iteration", 
        y="value", 
        hue="model",
        hue_order=legend_order,
        palette=model_colors,
        errorbar=('ci', 95),
        err_kws={'alpha': 0.1},  # Make confidence interval more transparent
        linewidth=2.5,
        ax=ax
    )

    # Styling
    ax.set_xlabel("Oracle Calls", fontsize=13, fontweight='normal')
    ax.set_ylabel("Boltz-2 binding affinity probability", fontsize=13, fontweight='normal')
    ax.set_xlim(0, 50)
    #ax.set_ylim(0, 1)
    # Use log scale for IC50 as values can span several orders of magnitude
    #ax.set_yscale('log')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if reference_lines:
        for ref in reference_lines:
            y = ref['y']
            color = ref['color']
            label = ref['label']
            ax.axhline(y=y, color=color, linestyle='--', linewidth=1.8, alpha=0.9)
            ax.text(
                15.2,
                y,
                label,
                color=color,
                fontsize=10,
                ha='left',
                va='bottom',
                bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.0, 'pad': 1.5},
            )

    # Conditionally show/hide legend
    if SHOW_LEGEND:
        handles, labels = ax.get_legend_handles_labels()
        new_labels = [legend_labels.get(label, label) for label in labels]
        legend = ax.legend(
            handles,
            new_labels,
            title=legend_name,
            frameon=True,
            fancybox=False,
            shadow=False,
            framealpha=0.95,
            edgecolor='black',
            loc='lower right'
        )
        legend.get_frame().set_linewidth(0.5)
    else:
        ax.get_legend().remove()

    plt.tight_layout()
    # Save figure as PNG
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / f'boltz2_{name}.png', format='png', bbox_inches='tight', dpi=300)



def plot_easy_medium_target():
    df_plot = loading_easy_medium_target()

    # Define color palette for explanation conditions
    model_colors = {
        'SARS CoV-2 MPro': '#2E86AB',
        'TRIB2 pseudokinase': '#f58220',
    }

    # Define legend order and labels
    legend_order = [
        'SARS CoV-2 MPro',
        'TRIB2 pseudokinase',
    ]

    legend_labels = {
        'SARS CoV-2 MPro': 'SARS CoV-2 MPro',
        'TRIB2 pseudokinase': 'TRIB2 pseudokinase',
    }

    reference_lines = [
        {
            'y': 0.999,
            'color': model_colors['SARS CoV-2 MPro'],
            'label': 'Nirmatrelvir (0.999)',
        },
        {
            'y': 0.170,
            'color': model_colors['TRIB2 pseudokinase'],
            'label': 'Afatinib (0.170)',
        },
    ]

    plotting(
        df_plot,
        'easy_medium_target',
        model_colors,
        legend_labels,
        legend_order,
        legend_name='Target',
        reference_lines=reference_lines,
    )

def loading_easy_medium_target():
    df_mpro = _load_target_traces(MPRO_EXPLANATION_DIR, 'SARS CoV-2 MPro')
    df_trib2 = _load_target_traces(TRIB2_EXPLANATION_DIR, 'TRIB2 pseudokinase')
    df_plot = pd.concat([df_mpro, df_trib2], ignore_index=True)
    return df_plot.sort_values(['model', 'iteration']).reset_index(drop=True)

def plot_hard_target():

    df_plot =loading_hard_target()


    # Define color palette for explanation conditions
    model_colors = {
        'SEISMO': '#2E86AB',
        'GP-BO': '#00a69d',
        'GraphGA': '#ffcc00',
        'REINVENT': '#f58220',
    }

    # Define legend order and labels
    legend_order = [
        'SEISMO',
        'GP-BO',
        'GraphGA',
        'REINVENT',

    ]

    legend_labels = {
        'SEISMO': 'SEISMO',
        'GraphGA': 'GraphGA',
        'GP-BO': 'GP-BO',
        'REINVENT': 'REINVENT',
    }

    plotting(df_plot, 'hard_target', model_colors, legend_labels, legend_order, legend_name='Model')


def loading_hard_target():
    frames = []
    df_baseline = pd.read_csv(HARD_BASELINE_PATH)
    for model in ['GraphGA', 'GP-BO', 'REINVENT']:
        df_model = df_baseline.loc[df_baseline['model'] == model].copy()
        # Keep replicate-level traces and compute best-so-far within each replicate.
        df_model = df_model.sort_values(['replicate', 'iteration'])
        df_model['affinity_probability_binary'] = (
            pd.to_numeric(df_model['affinity_probability_binary'], errors='coerce')
            .fillna(0.0)
            .groupby(df_model['replicate'])
            .cummax()
        )
        df_model = df_model[['iteration', 'affinity_probability_binary', 'model', 'replicate']].rename(
            columns={'affinity_probability_binary': 'value'}
        )
        if not df_model.empty and not df_model[['iteration', 'value', 'model']].isna().all().all():
            frames.append(df_model[['iteration', 'value', 'model']])

    df_hard = _load_target_traces(HARD_EXPLANATION_DIR, 'SEISMO')
    if not df_hard.empty and not df_hard[['iteration', 'value', 'model']].isna().all().all():
        frames.append(df_hard[['iteration', 'value', 'model']])

    if not frames:
        return pd.DataFrame(columns=['iteration', 'value', 'model'])

    df_plot = pd.concat(frames, ignore_index=True)

    df_plot = df_plot.sort_values(['model', 'iteration']).reset_index(drop=True)
    return df_plot

if __name__ == "__main__":
    plot_hard_target()
    plot_easy_medium_target()

