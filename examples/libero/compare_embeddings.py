"""Compare VLM embeddings from multiple different experiments.

This script loads multiple embedding files and creates comparative visualizations
to analyze differences between models or task suites.

Usage (2 files):
    python compare_embeddings.py \
        --embeddings_paths data/libero/embeddings/embeddings_libero_spatial.pkl \
                          data/libero/embeddings/embeddings_libero_goal.pkl \
        --labels "Spatial Suite" "Goal Suite"

Usage (3+ files):
    python compare_embeddings.py \
        --embeddings_paths data/libero/embeddings/embeddings_libero_spatial.pkl \
                          data/libero/embeddings/embeddings_libero_goal.pkl \
                          data/libero/embeddings/embeddings_libero_object.pkl \
        --labels "Spatial" "Goal" "Object"
"""

import argparse
import logging
import pathlib
import pickle
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns

# Optional: UMAP
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    logging.warning("UMAP not available. Install with: pip install umap-learn")


def load_embeddings(embeddings_path: str) -> dict[str, Any]:
    """Load embeddings from pickle file."""
    with open(embeddings_path, "rb") as f:
        data = pickle.load(f)
    logging.info(f"Loaded {len(data)} episodes from {embeddings_path}")
    return data


def prepare_combined_data(
    all_embeddings_list: list[list[dict]],
    labels: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Prepare embeddings from multiple sources for comparison.
    
    Args:
        all_embeddings_list: List of embedding sets
        labels: List of labels for each set
    
    Returns:
        embeddings: (N, D) array of all embeddings
        source_ids: (N,) array indicating source (0, 1, 2, ...)
        task_ids: (N,) array of task IDs (adjusted to avoid overlap)
        labels: list of labels
    """
    embeddings_list = []
    source_ids_list = []
    task_ids_list = []
    
    task_offset = 0
    
    # Process each embedding set
    for source_id, all_embeddings in enumerate(all_embeddings_list):
        for episode_data in all_embeddings:
            metadata = episode_data["metadata"]
            embeddings = episode_data["embeddings"]
            task_id = metadata["task_id"] + task_offset
            
            for emb_data in embeddings:
                embeddings_list.append(emb_data["embedding"])
                source_ids_list.append(source_id)
                task_ids_list.append(task_id)
        
        # Update offset for next source
        if task_ids_list:
            current_max = max(tid for tid, sid in zip(task_ids_list, source_ids_list) if sid == source_id)
            task_offset = current_max + 1
    
    # Convert to numpy arrays
    embeddings = np.stack(embeddings_list, axis=0)
    source_ids = np.array(source_ids_list)
    task_ids = np.array(task_ids_list)
    
    logging.info(f"Combined embeddings shape: {embeddings.shape}")
    for source_id, label in enumerate(labels):
        count = np.sum(source_ids == source_id)
        logging.info(f"{label}: {count} embeddings")
    
    return embeddings, source_ids, task_ids, labels


def visualize_comparison_tsne(
    embeddings: np.ndarray,
    source_ids: np.ndarray,
    task_ids: np.ndarray,
    labels: list[str],
    output_dir: pathlib.Path,
    perplexity: int = 30,
    n_iter: int = 2000,
):
    """Compare two embedding sets using t-SNE."""
    logging.info("Running t-SNE for comparison...")
    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=n_iter, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Define colors and markers for different sources
    n_sources = len(labels)
    import matplotlib.cm as cm
    colors = cm.tab10(np.linspace(0, 1, max(n_sources, 10)))[:n_sources]
    markers = ['o', '^', 's', 'D', 'v', '<', '>', 'p', '*', 'h'][:n_sources]
    
    # Create comparison visualizations
    n_cols = min(n_sources + 1, 4)
    n_rows = (n_sources + 1 + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 8 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Plot 1: All sources together
    ax = axes[0]
    for source_id in range(n_sources):
        mask = source_ids == source_id
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[colors[source_id]],
            marker=markers[source_id],
            alpha=0.5,
            s=30,
            label=labels[source_id],
        )
    
    ax.set_title(f"t-SNE Comparison by Source\nperplexity={perplexity}", fontsize=14)
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.legend(fontsize=10, loc='best')
    
    # Plot 2+: Each source highlighted individually
    for idx, source_id in enumerate(range(n_sources)):
        ax = axes[idx + 1]
        mask = source_ids == source_id
        
        # Plot other sources in gray
        ax.scatter(
            embeddings_2d[~mask, 0],
            embeddings_2d[~mask, 1],
            c='lightgray',
            alpha=0.2,
            s=10,
        )
        
        # Highlight current source
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[colors[source_id]],
            marker=markers[source_id],
            alpha=0.6,
            s=30,
        )
        
        ax.set_title(f"{labels[source_id]} (highlighted)", fontsize=14)
        ax.set_xlabel("t-SNE dimension 1")
        ax.set_ylabel("t-SNE dimension 2")
    
    # Hide unused subplots
    for idx in range(n_sources + 1, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    output_path = output_dir / "comparison_tsne.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logging.info(f"Saved comparison t-SNE to {output_path}")
    plt.close()
    
    # Create task-level comparison
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Plot with color by task, shape by source
    n_tasks = len(np.unique(task_ids))
    
    for source_id in range(n_sources):
        mask = source_ids == source_id
        scatter = ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=task_ids[mask],
            cmap='rainbow',
            marker=markers[source_id],
            alpha=0.6,
            s=30,
            label=labels[source_id],
        )
    
    ax.set_title(f"t-SNE Comparison (color=task, shape=source)", fontsize=14)
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.legend(fontsize=10, loc='best')
    
    plt.tight_layout()
    output_path = output_dir / "comparison_tsne_by_task.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logging.info(f"Saved task-level comparison t-SNE to {output_path}")
    plt.close()


def visualize_comparison_umap(
    embeddings: np.ndarray,
    source_ids: np.ndarray,
    task_ids: np.ndarray,
    labels: list[str],
    output_dir: pathlib.Path,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
):
    """Compare two embedding sets using UMAP."""
    if not UMAP_AVAILABLE:
        logging.warning("Skipping UMAP comparison (not installed)")
        return
    
    logging.info("Running UMAP for comparison...")
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings)
    
    # Define colors and markers for different sources
    n_sources = len(labels)
    import matplotlib.cm as cm
    colors = cm.tab10(np.linspace(0, 1, max(n_sources, 10)))[:n_sources]
    markers = ['o', '^', 's', 'D', 'v', '<', '>', 'p', '*', 'h'][:n_sources]
    
    # Create comparison visualizations
    n_cols = min(n_sources + 1, 4)
    n_rows = (n_sources + 1 + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 8 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Plot 1: All sources together
    ax = axes[0]
    for source_id in range(n_sources):
        mask = source_ids == source_id
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[colors[source_id]],
            marker=markers[source_id],
            alpha=0.5,
            s=30,
            label=labels[source_id],
        )
    
    ax.set_title(f"UMAP Comparison by Source\nn_neighbors={n_neighbors}", fontsize=14)
    ax.set_xlabel("UMAP dimension 1")
    ax.set_ylabel("UMAP dimension 2")
    ax.legend(fontsize=10, loc='best')
    
    # Plot 2+: Each source highlighted individually
    for idx, source_id in enumerate(range(n_sources)):
        ax = axes[idx + 1]
        mask = source_ids == source_id
        
        # Plot other sources in gray
        ax.scatter(
            embeddings_2d[~mask, 0],
            embeddings_2d[~mask, 1],
            c='lightgray',
            alpha=0.2,
            s=10,
        )
        
        # Highlight current source
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[colors[source_id]],
            marker=markers[source_id],
            alpha=0.6,
            s=30,
        )
        
        ax.set_title(f"{labels[source_id]} (highlighted)", fontsize=14)
        ax.set_xlabel("UMAP dimension 1")
        ax.set_ylabel("UMAP dimension 2")
    
    # Hide unused subplots
    for idx in range(n_sources + 1, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    output_path = output_dir / "comparison_umap.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logging.info(f"Saved comparison UMAP to {output_path}")
    plt.close()


def print_comparison_statistics(
    all_embeddings_list: list[list[dict]],
    labels: list[str],
):
    """Print comparison statistics."""
    print("\n" + "="*80)
    print("EMBEDDING COMPARISON STATISTICS")
    print("="*80)
    
    def get_stats(data, label):
        total_episodes = len(data)
        total_successes = sum(1 for ep in data if ep["metadata"]["success"])
        total_embeddings = sum(len(ep["embeddings"]) for ep in data)
        
        # Get unique tasks
        unique_tasks = set(ep["metadata"]["task_id"] for ep in data)
        
        print(f"\n{label}:")
        print(f"  Episodes: {total_episodes}")
        print(f"  Successes: {total_successes} ({total_successes/total_episodes*100:.1f}%)")
        print(f"  Total embeddings: {total_embeddings}")
        print(f"  Unique tasks: {len(unique_tasks)}")
        print(f"  Avg embeddings per episode: {total_embeddings/total_episodes:.1f}")
    
    for data, label in zip(all_embeddings_list, labels):
        get_stats(data, label)
    
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Compare VLM embeddings from multiple experiments")
    parser.add_argument(
        "--embeddings_paths",
        type=str,
        nargs='+',
        required=True,
        help="Paths to embeddings pickle files (space-separated, 2 or more files)",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs='+',
        default=None,
        help="Labels for each embedding set (optional, defaults to 'Set 1', 'Set 2', ...)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/libero/comparisons",
        help="Directory to save comparison visualizations",
    )
    parser.add_argument(
        "--tsne_perplexity",
        type=int,
        default=30,
        help="t-SNE perplexity parameter (default: 30)",
    )
    parser.add_argument(
        "--tsne_n_iter",
        type=int,
        default=2000,
        help="t-SNE number of iterations (default: 2000)",
    )
    parser.add_argument(
        "--umap_n_neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors parameter (default: 15)",
    )
    parser.add_argument(
        "--umap_min_dist",
        type=float,
        default=0.1,
        help="UMAP min_dist parameter (default: 0.1)",
    )
    parser.add_argument(
        "--skip_tsne",
        action="store_true",
        help="Skip t-SNE visualization",
    )
    parser.add_argument(
        "--skip_umap",
        action="store_true",
        help="Skip UMAP visualization",
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if len(args.embeddings_paths) < 2:
        parser.error("At least 2 embedding files are required for comparison")
    
    # Generate default labels if not provided
    if args.labels is None:
        args.labels = [f"Set {i+1}" for i in range(len(args.embeddings_paths))]
    elif len(args.labels) != len(args.embeddings_paths):
        parser.error(f"Number of labels ({len(args.labels)}) must match number of embedding files ({len(args.embeddings_paths)})")
    
    # Set up output directory
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all embedding sets
    logging.info(f"Loading {len(args.embeddings_paths)} embedding files...")
    all_embeddings_list = []
    for path in args.embeddings_paths:
        all_embeddings_list.append(load_embeddings(path))
    
    # Print statistics
    print_comparison_statistics(all_embeddings_list, args.labels)
    
    # Prepare combined data
    embeddings, source_ids, task_ids, labels = prepare_combined_data(
        all_embeddings_list,
        args.labels,
    )
    
    # Generate comparison visualizations
    if not args.skip_tsne:
        visualize_comparison_tsne(
            embeddings,
            source_ids,
            task_ids,
            labels,
            output_dir,
            perplexity=args.tsne_perplexity,
            n_iter=args.tsne_n_iter,
        )
    
    if not args.skip_umap:
        visualize_comparison_umap(
            embeddings,
            source_ids,
            task_ids,
            labels,
            output_dir,
            n_neighbors=args.umap_n_neighbors,
            min_dist=args.umap_min_dist,
        )
    
    logging.info(f"Comparison complete! Results saved in {output_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()

