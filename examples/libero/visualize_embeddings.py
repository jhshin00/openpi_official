"""Visualize VLM embeddings using t-SNE and UMAP.

This script loads embeddings saved by main.py and creates 2D visualizations
showing how the VLM embedding space organizes different tasks and trajectories.

Usage:
    python visualize_embeddings.py --embeddings_path data/libero/embeddings/embeddings_libero_spatial.pkl
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

# Optional: UMAP (install with: pip install umap-learn)
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


def prepare_data(all_embeddings: list[dict]) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, list[dict]]:
    """Prepare embeddings and labels for visualization.
    
    Returns:
        embeddings: (N, D) array of embeddings
        task_ids: (N,) array of task IDs
        task_names: list of unique task names
        success_flags: (N,) array of success flags (1=success, 0=failure)
        episode_info: list of dicts with episode metadata and embedding indices
    """
    embeddings_list = []
    task_ids_list = []
    task_names_dict = {}
    success_list = []
    episode_info = []
    
    current_idx = 0
    for episode_data in all_embeddings:
        metadata = episode_data["metadata"]
        embeddings = episode_data["embeddings"]
        
        task_id = metadata["task_id"]
        task_name = metadata["task_description"]
        success = metadata["success"]
        
        # Store task name mapping
        if task_id not in task_names_dict:
            task_names_dict[task_id] = task_name
        
        # Store episode info
        num_steps = len(embeddings)
        episode_info.append({
            "task_id": task_id,
            "episode_idx": metadata["episode_idx"],
            "success": success,
            "start_idx": current_idx,
            "end_idx": current_idx + num_steps,
            "num_steps": num_steps,
        })
        
        # Collect all embeddings from this episode
        for emb_data in embeddings:
            embeddings_list.append(emb_data["embedding"])
            task_ids_list.append(task_id)
            success_list.append(1 if success else 0)
        
        current_idx += num_steps
    
    # Convert to numpy arrays
    embeddings = np.stack(embeddings_list, axis=0)
    task_ids = np.array(task_ids_list)
    success_flags = np.array(success_list)
    
    # Create ordered list of task names
    unique_task_ids = sorted(task_names_dict.keys())
    task_names = [task_names_dict[tid] for tid in unique_task_ids]
    
    logging.info(f"Total embeddings: {len(embeddings)}")
    logging.info(f"Embedding dimension: {embeddings.shape[1]}")
    logging.info(f"Number of tasks: {len(task_names)}")
    logging.info(f"Success rate: {success_flags.mean():.2%}")
    
    return embeddings, task_ids, task_names, success_flags, episode_info


def visualize_tsne_plots(
    embeddings_2d: np.ndarray,
    task_ids: np.ndarray,
    task_names: list[str],
    success_flags: np.ndarray,
    output_dir: pathlib.Path,
    perplexity: int = 30,
    n_iter: int = 1000,
):
    """Create standard t-SNE scatter plots."""
    logging.info("Creating t-SNE scatter plots...")
    
    # Create visualizations
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: Color by task
    ax = axes[0]
    scatter = ax.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=task_ids,
        cmap='tab20',
        alpha=0.6,
        s=20,
    )
    ax.set_title(f"t-SNE Visualization (colored by task)\nperplexity={perplexity}, max_iter={n_iter}", fontsize=14)
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    
    # Add colorbar with task names (if not too many tasks)
    if len(task_names) <= 20:
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Task ID")
    
    # Plot 2: Color by success/failure
    ax = axes[1]
    colors = ['red' if s == 0 else 'green' for s in success_flags]
    ax.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=colors,
        alpha=0.6,
        s=20,
    )
    ax.set_title(f"t-SNE Visualization (colored by success/failure)\nGreen=Success, Red=Failure", fontsize=14)
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    
    plt.tight_layout()
    output_path = output_dir / "tsne_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logging.info(f"Saved t-SNE visualization to {output_path}")
    plt.close()
    
    # Create a separate plot for each task (if not too many)
    if len(task_names) <= 10:
        fig, axes = plt.subplots(2, 5, figsize=(25, 10))
        axes = axes.flatten()
        
        for task_id, task_name in enumerate(task_names):
            ax = axes[task_id]
            
            # Highlight current task
            mask = task_ids == task_id
            
            # Plot all points in gray
            ax.scatter(
                embeddings_2d[~mask, 0],
                embeddings_2d[~mask, 1],
                c='lightgray',
                alpha=0.3,
                s=10,
            )
            
            # Plot current task points colored by success
            task_colors = ['red' if s == 0 else 'green' for s in success_flags[mask]]
            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=task_colors,
                alpha=0.8,
                s=30,
            )
            
            ax.set_title(f"Task {task_id}: {task_name[:30]}...", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Hide unused subplots
        for i in range(len(task_names), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        output_path = output_dir / "tsne_by_task.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logging.info(f"Saved per-task t-SNE visualization to {output_path}")
        plt.close()


def visualize_umap_plots(
    embeddings_2d: np.ndarray,
    task_ids: np.ndarray,
    task_names: list[str],
    success_flags: np.ndarray,
    output_dir: pathlib.Path,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
):
    """Create standard UMAP scatter plots."""
    logging.info("Creating UMAP scatter plots...")
    
    # Create visualizations
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: Color by task
    ax = axes[0]
    scatter = ax.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=task_ids,
        cmap='tab20',
        alpha=0.6,
        s=20,
    )
    ax.set_title(f"UMAP Visualization (colored by task)\nn_neighbors={n_neighbors}, min_dist={min_dist}", fontsize=14)
    ax.set_xlabel("UMAP dimension 1")
    ax.set_ylabel("UMAP dimension 2")
    
    # Add colorbar with task names (if not too many tasks)
    if len(task_names) <= 20:
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Task ID")
    
    # Plot 2: Color by success/failure
    ax = axes[1]
    colors = ['red' if s == 0 else 'green' for s in success_flags]
    ax.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=colors,
        alpha=0.6,
        s=20,
    )
    ax.set_title(f"UMAP Visualization (colored by success/failure)\nGreen=Success, Red=Failure", fontsize=14)
    ax.set_xlabel("UMAP dimension 1")
    ax.set_ylabel("UMAP dimension 2")
    
    plt.tight_layout()
    output_path = output_dir / "umap_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logging.info(f"Saved UMAP visualization to {output_path}")
    plt.close()
    
    # Create a separate plot for each task (if not too many)
    if len(task_names) <= 10:
        fig, axes = plt.subplots(2, 5, figsize=(25, 10))
        axes = axes.flatten()
        
        for task_id, task_name in enumerate(task_names):
            ax = axes[task_id]
            
            # Highlight current task
            mask = task_ids == task_id
            
            # Plot all points in gray
            ax.scatter(
                embeddings_2d[~mask, 0],
                embeddings_2d[~mask, 1],
                c='lightgray',
                alpha=0.3,
                s=10,
            )
            
            # Plot current task points colored by success
            task_colors = ['red' if s == 0 else 'green' for s in success_flags[mask]]
            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=task_colors,
                alpha=0.8,
                s=30,
            )
            
            ax.set_title(f"Task {task_id}: {task_name[:30]}...", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Hide unused subplots
        for i in range(len(task_names), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        output_path = output_dir / "umap_by_task.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logging.info(f"Saved per-task UMAP visualization to {output_path}")
        plt.close()


def visualize_trajectories_tsne(
    embeddings_2d: np.ndarray,
    episode_info: list[dict],
    task_names: list[str],
    output_dir: pathlib.Path,
    max_episodes_per_task: int = None,
):
    """Visualize episode trajectories with time progression shown by transparency.
    
    Args:
        embeddings_2d: (N, 2) array of 2D embeddings
        episode_info: List of episode metadata
        task_names: List of task names
        output_dir: Output directory
        max_episodes_per_task: Maximum episodes to show per task (None = show all)
    """
    logging.info("Creating trajectory visualizations...")
    
    # Group episodes by task
    task_episodes = {}
    for ep in episode_info:
        task_id = ep["task_id"]
        if task_id not in task_episodes:
            task_episodes[task_id] = []
        task_episodes[task_id].append(ep)
    
    # Create per-task trajectory plots (if not too many tasks)
    if len(task_names) <= 10:
        fig, axes = plt.subplots(2, 5, figsize=(30, 12))
        axes = axes.flatten()
        
        for task_id in sorted(task_episodes.keys()):
            ax = axes[task_id]
            episodes = task_episodes[task_id] if max_episodes_per_task is None else task_episodes[task_id][:max_episodes_per_task]
            
            for ep in episodes:
                start, end = ep["start_idx"], ep["end_idx"]
                traj = embeddings_2d[start:end]
                
                if len(traj) < 2:
                    continue
                
                # Create alpha values: start transparent (0.2) -> end opaque (1.0)
                alphas = np.linspace(0.2, 1.0, len(traj))
                
                # Plot trajectory as line segments with varying alpha
                for i in range(len(traj) - 1):
                    color = 'green' if ep["success"] else 'red'
                    ax.plot(
                        traj[i:i+2, 0],
                        traj[i:i+2, 1],
                        color=color,
                        alpha=alphas[i],
                        linewidth=2,
                    )
                
                # Mark start and end points
                ax.scatter(traj[0, 0], traj[0, 1], c='blue', s=100, marker='o', 
                          edgecolors='black', linewidths=2, zorder=5, label='Start' if ep == episodes[0] else '')
                ax.scatter(traj[-1, 0], traj[-1, 1], c='orange', s=100, marker='*', 
                          edgecolors='black', linewidths=2, zorder=5, label='End' if ep == episodes[0] else '')
            
            ax.set_title(f"Task {task_id}: {task_names[task_id][:40]}...\nGreen=Success, Red=Failure", fontsize=10)
            ax.set_xlabel("Dimension 1")
            ax.set_ylabel("Dimension 2")
            if task_id == 0:
                ax.legend(loc='upper right', fontsize=8)
        
        # Hide unused subplots
        for i in range(len(task_names), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle("Episode Trajectories (transparency: early→late steps)", fontsize=16, y=1.00)
        plt.tight_layout()
        output_path = output_dir / "trajectories_by_task.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logging.info(f"Saved trajectory visualization to {output_path}")
        plt.close()
    
    # Create overall trajectory plot
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    
    # Plot 1: All trajectories colored by task
    ax = axes[0]
    import matplotlib.cm as cm
    colors = cm.tab20(np.linspace(0, 1, len(task_names)))
    
    for task_id, episodes in task_episodes.items():
        episodes_to_plot = episodes if max_episodes_per_task is None else episodes[:max_episodes_per_task]
        for ep in episodes_to_plot:
            start, end = ep["start_idx"], ep["end_idx"]
            traj = embeddings_2d[start:end]
            
            if len(traj) < 2:
                continue
            
            alphas = np.linspace(0.2, 1.0, len(traj))
            
            for i in range(len(traj) - 1):
                ax.plot(
                    traj[i:i+2, 0],
                    traj[i:i+2, 1],
                    color=colors[task_id],
                    alpha=alphas[i],
                    linewidth=1.5,
                )
            
            # Mark start point
            ax.scatter(traj[0, 0], traj[0, 1], c=[colors[task_id]], s=50, 
                      marker='o', edgecolors='black', linewidths=1, zorder=5, alpha=0.7)
    
    ax.set_title("All Trajectories (colored by task)\nTransparency: early steps (transparent) → late steps (opaque)", fontsize=12)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    
    # Plot 2: Success vs Failure trajectories
    ax = axes[1]
    
    for task_id, episodes in task_episodes.items():
        episodes_to_plot = episodes if max_episodes_per_task is None else episodes[:max_episodes_per_task]
        for ep in episodes_to_plot:
            start, end = ep["start_idx"], ep["end_idx"]
            traj = embeddings_2d[start:end]
            
            if len(traj) < 2:
                continue
            
            alphas = np.linspace(0.2, 1.0, len(traj))
            color = 'green' if ep["success"] else 'red'
            
            for i in range(len(traj) - 1):
                ax.plot(
                    traj[i:i+2, 0],
                    traj[i:i+2, 1],
                    color=color,
                    alpha=alphas[i],
                    linewidth=1.5,
                )
            
            # Mark start point
            ax.scatter(traj[0, 0], traj[0, 1], c=color, s=50, 
                      marker='o', edgecolors='black', linewidths=1, zorder=5, alpha=0.5)
    
    ax.set_title("All Trajectories (colored by success/failure)\nGreen=Success, Red=Failure", fontsize=12)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    
    plt.tight_layout()
    output_path = output_dir / "trajectories_overall.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logging.info(f"Saved overall trajectory visualization to {output_path}")
    plt.close()


def visualize_trajectories_umap(
    embeddings_2d: np.ndarray,
    episode_info: list[dict],
    task_names: list[str],
    output_dir: pathlib.Path,
    max_episodes_per_task: int = None,
):
    """Visualize episode trajectories using UMAP with time progression shown by transparency."""
    logging.info("Creating UMAP trajectory visualizations...")
    
    # Group episodes by task
    task_episodes = {}
    for ep in episode_info:
        task_id = ep["task_id"]
        if task_id not in task_episodes:
            task_episodes[task_id] = []
        task_episodes[task_id].append(ep)
    
    # Create per-task trajectory plots (if not too many tasks)
    if len(task_names) <= 10:
        fig, axes = plt.subplots(2, 5, figsize=(30, 12))
        axes = axes.flatten()
        
        for task_id in sorted(task_episodes.keys()):
            ax = axes[task_id]
            episodes = task_episodes[task_id] if max_episodes_per_task is None else task_episodes[task_id][:max_episodes_per_task]
            
            for ep in episodes:
                start, end = ep["start_idx"], ep["end_idx"]
                traj = embeddings_2d[start:end]
                
                if len(traj) < 2:
                    continue
                
                alphas = np.linspace(0.2, 1.0, len(traj))
                
                for i in range(len(traj) - 1):
                    color = 'green' if ep["success"] else 'red'
                    ax.plot(
                        traj[i:i+2, 0],
                        traj[i:i+2, 1],
                        color=color,
                        alpha=alphas[i],
                        linewidth=2,
                    )
                
                # Mark start and end points
                ax.scatter(traj[0, 0], traj[0, 1], c='blue', s=100, marker='o', 
                          edgecolors='black', linewidths=2, zorder=5, label='Start' if ep == episodes[0] else '')
                ax.scatter(traj[-1, 0], traj[-1, 1], c='orange', s=100, marker='*', 
                          edgecolors='black', linewidths=2, zorder=5, label='End' if ep == episodes[0] else '')
            
            ax.set_title(f"Task {task_id}: {task_names[task_id][:40]}...\nGreen=Success, Red=Failure", fontsize=10)
            ax.set_xlabel("UMAP Dimension 1")
            ax.set_ylabel("UMAP Dimension 2")
            if task_id == 0:
                ax.legend(loc='upper right', fontsize=8)
        
        # Hide unused subplots
        for i in range(len(task_names), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle("UMAP Episode Trajectories (transparency: early→late steps)", fontsize=16, y=1.00)
        plt.tight_layout()
        output_path = output_dir / "umap_trajectories_by_task.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logging.info(f"Saved UMAP trajectory visualization to {output_path}")
        plt.close()


def print_statistics(all_embeddings: list[dict], task_names: list[str]):
    """Print statistics about the collected embeddings."""
    print("\n" + "="*80)
    print("EMBEDDING COLLECTION STATISTICS")
    print("="*80)
    
    # Overall statistics
    total_episodes = len(all_embeddings)
    total_successes = sum(1 for ep in all_embeddings if ep["metadata"]["success"])
    print(f"\nTotal episodes: {total_episodes}")
    print(f"Total successes: {total_successes} ({total_successes/total_episodes*100:.1f}%)")
    
    # Per-task statistics
    print(f"\nPer-task breakdown:")
    print("-" * 80)
    
    task_stats = {}
    for episode_data in all_embeddings:
        metadata = episode_data["metadata"]
        task_id = metadata["task_id"]
        
        if task_id not in task_stats:
            task_stats[task_id] = {
                "name": metadata["task_description"],
                "total": 0,
                "successes": 0,
                "embeddings": 0,
            }
        
        task_stats[task_id]["total"] += 1
        if metadata["success"]:
            task_stats[task_id]["successes"] += 1
        task_stats[task_id]["embeddings"] += len(episode_data["embeddings"])
    
    for task_id in sorted(task_stats.keys()):
        stats = task_stats[task_id]
        success_rate = stats["successes"] / stats["total"] * 100 if stats["total"] > 0 else 0
        avg_emb = stats["embeddings"] / stats["total"] if stats["total"] > 0 else 0
        print(f"Task {task_id}: {stats['name']}")
        print(f"  Episodes: {stats['total']}, Successes: {stats['successes']} ({success_rate:.1f}%)")
        print(f"  Avg embeddings per episode: {avg_emb:.1f}")
    
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Visualize VLM embeddings from LIBERO experiments")
    parser.add_argument(
        "--embeddings_path",
        type=str,
        required=True,
        help="Path to embeddings pickle file (e.g., data/libero/embeddings/embeddings_libero_spatial.pkl)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save visualizations (default: same as embeddings_path)",
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
    parser.add_argument(
        "--max_episodes_per_task",
        type=int,
        default=None,
        help="Maximum number of episodes to show per task in trajectory plots (default: all episodes)",
    )
    
    args = parser.parse_args()
    
    # Set up output directory
    if args.output_dir is None:
        args.output_dir = pathlib.Path(args.embeddings_path).parent
    else:
        args.output_dir = pathlib.Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load embeddings
    all_embeddings = load_embeddings(args.embeddings_path)
    
    # Prepare data
    embeddings, task_ids, task_names, success_flags, episode_info = prepare_data(all_embeddings)
    
    # Print statistics
    print_statistics(all_embeddings, task_names)
    
    # Generate visualizations
    embeddings_2d_tsne = None
    embeddings_2d_umap = None
    
    if not args.skip_tsne:
        from sklearn.manifold import TSNE
        logging.info("Running t-SNE...")
        tsne = TSNE(n_components=2, perplexity=args.tsne_perplexity, max_iter=args.tsne_n_iter, random_state=42)
        embeddings_2d_tsne = tsne.fit_transform(embeddings)
        
        # Create standard visualizations
        visualize_tsne_plots(
            embeddings_2d_tsne,
            task_ids,
            task_names,
            success_flags,
            args.output_dir,
            perplexity=args.tsne_perplexity,
            n_iter=args.tsne_n_iter,
        )
        
        # Create trajectory visualizations
        visualize_trajectories_tsne(
            embeddings_2d_tsne,
            episode_info,
            task_names,
            args.output_dir,
            max_episodes_per_task=args.max_episodes_per_task,
        )
    
    if not args.skip_umap:
        if not UMAP_AVAILABLE:
            logging.warning("Skipping UMAP visualization (not installed)")
        else:
            logging.info("Running UMAP...")
            reducer = umap.UMAP(n_neighbors=args.umap_n_neighbors, min_dist=args.umap_min_dist, random_state=42)
            embeddings_2d_umap = reducer.fit_transform(embeddings)
            
            # Create standard visualizations
            visualize_umap_plots(
                embeddings_2d_umap,
                task_ids,
                task_names,
                success_flags,
                args.output_dir,
                n_neighbors=args.umap_n_neighbors,
                min_dist=args.umap_min_dist,
            )
            
            # Create trajectory visualizations
            visualize_trajectories_umap(
                embeddings_2d_umap,
                episode_info,
                task_names,
                args.output_dir,
                max_episodes_per_task=args.max_episodes_per_task,
            )
    
    logging.info("Visualization complete!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()

