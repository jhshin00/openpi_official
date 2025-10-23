#!/bin/bash
# Visualization script for VLM embeddings with various hyperparameter configurations

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

# Default paths
DEFAULT_EMBEDDINGS_PATH="data/libero/embeddings/libero_finetune/embeddings_libero_object.pkl"
DEFAULT_OUTPUT_DIR="data/libero/visualizations/libero_finetune/libero_object"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# Helper Functions
# ============================================================================

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_file_exists() {
    if [ ! -f "$1" ]; then
        print_error "File not found: $1"
        exit 1
    fi
}

# ============================================================================
# Visualization Presets
# ============================================================================

visualize_default() {
    local embeddings_path=$1
    local output_dir="${2}/default"
    
    print_header "Running Default Visualization"
    print_info "Balanced view for task clustering"
    print_info "Output: $output_dir"
    
    python examples/libero/visualize_embeddings.py \
        --embeddings_path "$embeddings_path" \
        --output_dir "$output_dir" \
        --tsne_perplexity 30 \
        --tsne_n_iter 1500 \
        --umap_n_neighbors 15 \
        --umap_min_dist 0.1
    
    print_info "✓ Default visualization complete"
}

visualize_local() {
    local embeddings_path=$1
    local output_dir="${2}/local_structure"
    
    print_header "Running Local Structure Visualization"
    print_info "Focus on fine-grained trajectory details"
    print_info "Output: $output_dir"
    
    python examples/libero/visualize_embeddings.py \
        --embeddings_path "$embeddings_path" \
        --output_dir "$output_dir" \
        --tsne_perplexity 10 \
        --tsne_n_iter 2000 \
        --umap_n_neighbors 5 \
        --umap_min_dist 0.0
    
    print_info "✓ Local structure visualization complete"
}

visualize_global() {
    local embeddings_path=$1
    local output_dir="${2}/global_structure"
    
    print_header "Running Global Structure Visualization"
    print_info "Focus on overall task relationships"
    print_info "Output: $output_dir"
    
    python examples/libero/visualize_embeddings.py \
        --embeddings_path "$embeddings_path" \
        --output_dir "$output_dir" \
        --tsne_perplexity 50 \
        --tsne_n_iter 1500 \
        --umap_n_neighbors 50 \
        --umap_min_dist 0.3
    
    print_info "✓ Global structure visualization complete"
}

visualize_compact() {
    local embeddings_path=$1
    local output_dir="${2}/compact_clusters"
    
    print_header "Running Compact Cluster Visualization"
    print_info "Dense clusters for clear task separation"
    print_info "Output: $output_dir"
    
    python examples/libero/visualize_embeddings.py \
        --embeddings_path "$embeddings_path" \
        --output_dir "$output_dir" \
        --tsne_perplexity 20 \
        --tsne_n_iter 1500 \
        --umap_n_neighbors 10 \
        --umap_min_dist 0.0
    
    print_info "✓ Compact cluster visualization complete"
}

visualize_spread() {
    local embeddings_path=$1
    local output_dir="${2}/spread_view"
    
    print_header "Running Spread View Visualization"
    print_info "Dispersed points for relationship analysis"
    print_info "Output: $output_dir"
    
    python examples/libero/visualize_embeddings.py \
        --embeddings_path "$embeddings_path" \
        --output_dir "$output_dir" \
        --tsne_perplexity 40 \
        --tsne_n_iter 1500 \
        --umap_n_neighbors 30 \
        --umap_min_dist 0.5
    
    print_info "✓ Spread view visualization complete"
}

visualize_all_presets() {
    local embeddings_path=$1
    local output_dir=$2
    
    print_header "Running All Visualization Presets"
    
    visualize_default "$embeddings_path" "$output_dir"
    echo ""
    visualize_local "$embeddings_path" "$output_dir"
    echo ""
    visualize_global "$embeddings_path" "$output_dir"
    echo ""
    visualize_compact "$embeddings_path" "$output_dir"
    echo ""
    visualize_spread "$embeddings_path" "$output_dir"
    
    print_header "All Presets Complete!"
    print_info "Results saved in: $output_dir"
    print_info "Subdirectories:"
    print_info "  - default/          : Balanced view (recommended)"
    print_info "  - local_structure/  : Fine-grained details"
    print_info "  - global_structure/ : Overall relationships"
    print_info "  - compact_clusters/ : Clear task separation"
    print_info "  - spread_view/      : Relationship analysis"
}

visualize_custom() {
    local embeddings_path=$1
    local output_dir=$2
    shift 2
    
    print_header "Running Custom Visualization"
    print_info "Output: $output_dir"
    
    python examples/libero/visualize_embeddings.py \
        --embeddings_path "$embeddings_path" \
        --output_dir "$output_dir" \
        "$@"
    
    print_info "✓ Custom visualization complete"
}

# ============================================================================
# Parameter Sweep
# ============================================================================

sweep_perplexity() {
    local embeddings_path=$1
    local output_dir="${2}/perplexity_sweep"
    
    print_header "Sweeping t-SNE Perplexity"
    
    for perp in 5 10 20 30 50 80; do
        local dir="${output_dir}/perplexity_${perp}"
        print_info "Testing perplexity=$perp → $dir"
        
        python examples/libero/visualize_embeddings.py \
            --embeddings_path "$embeddings_path" \
            --output_dir "$dir" \
            --tsne_perplexity $perp \
            --skip_umap
    done
    
    print_info "✓ Perplexity sweep complete"
}

sweep_neighbors() {
    local embeddings_path=$1
    local output_dir="${2}/neighbors_sweep"
    
    print_header "Sweeping UMAP n_neighbors"
    
    for n in 5 10 15 30 50 100; do
        local dir="${output_dir}/neighbors_${n}"
        print_info "Testing n_neighbors=$n → $dir"
        
        python examples/libero/visualize_embeddings.py \
            --embeddings_path "$embeddings_path" \
            --output_dir "$dir" \
            --umap_n_neighbors $n \
            --skip_tsne
    done
    
    print_info "✓ Neighbors sweep complete"
}

sweep_min_dist() {
    local embeddings_path=$1
    local output_dir="${2}/min_dist_sweep"
    
    print_header "Sweeping UMAP min_dist"
    
    for dist in 0.0 0.1 0.3 0.5 0.8; do
        local dir="${output_dir}/min_dist_${dist}"
        print_info "Testing min_dist=$dist → $dir"
        
        python examples/libero/visualize_embeddings.py \
            --embeddings_path "$embeddings_path" \
            --output_dir "$dir" \
            --umap_min_dist $dist \
            --skip_tsne
    done
    
    print_info "✓ Min_dist sweep complete"
}

# ============================================================================
# Comparison Functions
# ============================================================================

compare_embeddings() {
    local embeddings_path_1=$1
    local embeddings_path_2=$2
    local label_1=$3
    local label_2=$4
    local output_dir=$5
    
    print_header "Comparing Two Embedding Sets"
    print_info "Set 1: $embeddings_path_1 ($label_1)"
    print_info "Set 2: $embeddings_path_2 ($label_2)"
    print_info "Output: $output_dir"
    
    python examples/libero/compare_embeddings.py \
        --embeddings_path_1 "$embeddings_path_1" \
        --embeddings_path_2 "$embeddings_path_2" \
        --label_1 "$label_1" \
        --label_2 "$label_2" \
        --output_dir "$output_dir"
    
    print_info "✓ Comparison complete"
}

# ============================================================================
# Usage Information
# ============================================================================

show_usage() {
    cat << EOF
${BLUE}VLM Embedding Visualization Script${NC}

${GREEN}USAGE:${NC}
    $0 [MODE] [OPTIONS]

${GREEN}MODES:${NC}
    default              Run default visualization (balanced view)
    local                Focus on local structure (fine-grained)
    global               Focus on global structure (task relationships)
    compact              Compact clusters (clear separation)
    spread               Spread view (relationship analysis)
    all                  Run all presets above
    
    sweep-perplexity     Sweep t-SNE perplexity values
    sweep-neighbors      Sweep UMAP n_neighbors values
    sweep-min-dist       Sweep UMAP min_dist values
    
    compare              Compare two embedding files
    custom               Custom parameters (see below)

${GREEN}OPTIONS:${NC}
    -i, --input PATH     Input embeddings file
                         Default: $DEFAULT_EMBEDDINGS_PATH
    
    -o, --output DIR     Output directory
                         Default: $DEFAULT_OUTPUT_DIR
    
    -h, --help           Show this help message

${GREEN}COMPARE MODE OPTIONS:${NC}
    --input1 PATH        First embeddings file (required for compare mode)
    --input2 PATH        Second embeddings file (required for compare mode)
    --label1 TEXT        Label for first embedding set (default: "Embeddings 1")
    --label2 TEXT        Label for second embedding set (default: "Embeddings 2")

${GREEN}CUSTOM MODE OPTIONS:${NC}
    --tsne_perplexity N          t-SNE perplexity (default: 30)
    --tsne_n_iter N              t-SNE iterations (default: 1000)
    --umap_n_neighbors N         UMAP neighbors (default: 15)
    --umap_min_dist F            UMAP min distance (default: 0.1)
    --skip_tsne                  Skip t-SNE visualization
    --skip_umap                  Skip UMAP visualization

${GREEN}EXAMPLES:${NC}
    # Run default visualization
    $0 default

    # Run all presets
    $0 all

    # Custom input/output paths
    $0 default -i data/embeddings.pkl -o results/

    # Custom parameters
    $0 custom -i data/embeddings.pkl --tsne_perplexity 50 --umap_n_neighbors 30

    # Sweep perplexity values
    $0 sweep-perplexity -i data/embeddings.pkl
    
    # Compare two embedding files
    $0 compare --input1 data/embeddings_spatial.pkl --input2 data/embeddings_goal.pkl \\
        --label1 "Spatial Suite" --label2 "Goal Suite" -o results/comparison/

${GREEN}PRESET CONFIGURATIONS:${NC}
    default:  perplexity=30,  n_neighbors=15,  min_dist=0.1  (recommended)
    local:    perplexity=10,  n_neighbors=5,   min_dist=0.0  (fine details)
    global:   perplexity=50,  n_neighbors=50,  min_dist=0.3  (big picture)
    compact:  perplexity=20,  n_neighbors=10,  min_dist=0.0  (clear clusters)
    spread:   perplexity=40,  n_neighbors=30,  min_dist=0.5  (relationships)

EOF
}

# ============================================================================
# Main Script
# ============================================================================

main() {
    # Default values
    local embeddings_path="$DEFAULT_EMBEDDINGS_PATH"
    local output_dir="$DEFAULT_OUTPUT_DIR"
    local mode=""
    local input1=""
    local input2=""
    local label1="Embeddings 1"
    local label2="Embeddings 2"
    
    # Parse arguments
    if [ $# -eq 0 ]; then
        show_usage
        exit 0
    fi
    
    mode=$1
    shift
    
    # Parse options
    while [[ $# -gt 0 ]]; do
        case $1 in
            -i|--input)
                embeddings_path="$2"
                shift 2
                ;;
            -o|--output)
                output_dir="$2"
                shift 2
                ;;
            --input1)
                input1="$2"
                shift 2
                ;;
            --input2)
                input2="$2"
                shift 2
                ;;
            --label1)
                label1="$2"
                shift 2
                ;;
            --label2)
                label2="$2"
                shift 2
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                # For custom mode, pass remaining args to visualize_custom
                if [ "$mode" = "custom" ]; then
                    break
                else
                    print_error "Unknown option: $1"
                    show_usage
                    exit 1
                fi
                ;;
        esac
    done
    
    # Check if embeddings file exists
    check_file_exists "$embeddings_path"
    
    # Create output directory
    mkdir -p "$output_dir"
    
    print_info "Embeddings file: $embeddings_path"
    print_info "Output directory: $output_dir"
    echo ""
    
    # Execute mode
    case $mode in
        default)
            visualize_default "$embeddings_path" "$output_dir"
            ;;
        local)
            visualize_local "$embeddings_path" "$output_dir"
            ;;
        global)
            visualize_global "$embeddings_path" "$output_dir"
            ;;
        compact)
            visualize_compact "$embeddings_path" "$output_dir"
            ;;
        spread)
            visualize_spread "$embeddings_path" "$output_dir"
            ;;
        all)
            visualize_all_presets "$embeddings_path" "$output_dir"
            ;;
        sweep-perplexity)
            sweep_perplexity "$embeddings_path" "$output_dir"
            ;;
        sweep-neighbors)
            sweep_neighbors "$embeddings_path" "$output_dir"
            ;;
        sweep-min-dist)
            sweep_min_dist "$embeddings_path" "$output_dir"
            ;;
        compare)
            if [ -z "$input1" ] || [ -z "$input2" ]; then
                print_error "Compare mode requires --input1 and --input2"
                show_usage
                exit 1
            fi
            check_file_exists "$input1"
            check_file_exists "$input2"
            compare_embeddings "$input1" "$input2" "$label1" "$label2" "$output_dir"
            ;;
        custom)
            visualize_custom "$embeddings_path" "$output_dir" "$@"
            ;;
        -h|--help|help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown mode: $mode"
            echo ""
            show_usage
            exit 1
            ;;
    esac
    
    echo ""
    print_header "Complete! 🎉"
}

# Run main function
main "$@"

