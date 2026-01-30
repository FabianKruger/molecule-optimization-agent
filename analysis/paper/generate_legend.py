#!/usr/bin/env python3
"""
Generate a standalone horizontal legend PDF for the XAI comparison figures.

This legend can be placed at the top of a LaTeX figure containing multiple
subfigures that share the same legend entries.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


def generate_legend(output_path: Path = Path("./figures/xai_legend.pdf")):
    """Generate a standalone horizontal legend PDF.
    
    Args:
        output_path: Path to save the legend PDF
    """
    # Define legend entries with colors (matching the notebooks)
    legend_entries = [
        ("Full explanation", "#2E86AB"),
        ("Partial explanation", "#00a69d"),
        ("No explanation", "#ffcc00"),
        ("No description", "#f58220"),
        ("No sequence", "#B983D9"),
    ]
    
    # Set publication-quality style
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'serif',
        'legend.fontsize': 10,
    })
    
    # Create figure just for the legend
    fig, ax = plt.subplots(figsize=(8, 0.5))
    ax.set_axis_off()
    
    # Create legend handles
    handles = [
        mpatches.Patch(color=color, label=label)
        for label, color in legend_entries
    ]
    
    # Create horizontal legend
    legend = ax.legend(
        handles=handles,
        loc='center',
        ncol=len(legend_entries),
        frameon=True,
        fancybox=False,
        shadow=False,
        framealpha=0.95,
        edgecolor='black',
        handlelength=1.5,
        handletextpad=0.5,
        columnspacing=1.5,
    )
    legend.get_frame().set_linewidth(0.5)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save figure
    fig.savefig(
        output_path,
        format='pdf',
        bbox_inches='tight',
        dpi=300,
        pad_inches=0.05
    )
    plt.close(fig)
    
    print(f"Legend saved to: {output_path}")


if __name__ == "__main__":
    generate_legend()
