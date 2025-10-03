import argparse
import json
import numpy as np
from pathlib import Path
import io

class OutputWriter:
    def __init__(self, output_file):
        self.output_file = output_file
        self.buffer = io.StringIO()
    
    def write(self, text):
        self.buffer.write(text)
    
    def save_to_file(self):
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(self.buffer.getvalue())
        self.buffer.close()

def load_results(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    return results

def write_header(writer, title, char="="):
    width = 70
    writer.write("\n" + char * width + "\n")
    writer.write(f"{title:^{width}}\n")
    writer.write(char * width + "\n")

def write_section(writer, title, char="-"):
    writer.write(f"\n{char * 50}\n")
    writer.write(f"{title}\n")
    writer.write(f"{char * 50}\n")

def analyze_overall_stats(writer, results):
    write_header(writer, "OVERALL STATISTICS")
    
    layers = results.get('layers', [])
    total_samples = results.get('counts', 0)
    top_k = results.get('top_k', 50)
    
    writer.write(f"Dataset Summary:\n")
    writer.write(f"  - Total samples processed: {total_samples:,}\n")
    writer.write(f"  - Layers analyzed: {layers}\n")
    writer.write(f"  - Top-K selection: {top_k}\n")
    writer.write(f"  - Number of layers: {len(layers)}\n")
    
    if total_samples > 0:
        writer.write(f"\nSample Distribution Analysis:\n")
        per_layer = results.get('per_layer', {})
        if per_layer:
            first_layer = list(per_layer.keys())[0]
            layer_data = per_layer[first_layer]
            latent_dim = layer_data.get('latent_dim', 0)
            writer.write(f"  - Latent dimension per layer: {latent_dim:,}\n")

def analyze_layer_performance(writer, results):
    write_header(writer, "LAYER-WISE ANALYSIS")
    
    per_layer = results.get('per_layer', {})
    
    for layer_id, layer_data in per_layer.items():
        write_section(writer, f"Layer {layer_id}")
        
        # Basic info
        latent_dim = layer_data.get('latent_dim', 0)
        top_indices = layer_data.get('top_indices', [])
        top_scores = layer_data.get('top_scores', [])
        
        writer.write(f"  - Latent dimension: {latent_dim:,}\n")
        writer.write(f"  - Top latents identified: {len(top_indices)}\n")
        
        # Metric arrays
        dfreq = np.array(layer_data.get('dfreq', []))
        dmag = np.array(layer_data.get('dmag', []))
        auroc = np.array(layer_data.get('auroc', []))
        consensus = np.array(layer_data.get('consensus', []))
        
        # Overall metric statistics
        writer.write(f"\n   Metric Statistics (all {len(consensus):,} latents):\n")
        writer.write(f"    - Frequency Diff (dfreq):\n")
        writer.write(f"      - Mean: {dfreq.mean():.4f} ± {dfreq.std():.4f}\n")
        writer.write(f"      - Range: [{dfreq.min():.4f}, {dfreq.max():.4f}]\n")
        writer.write(f"      - Positive: {(dfreq > 0).sum():,} ({(dfreq > 0).mean()*100:.1f}%)\n")
        
        writer.write(f"    - Magnitude Diff (dmag):\n")
        writer.write(f"      - Mean: {dmag.mean():.4f} ± {dmag.std():.4f}\n")
        writer.write(f"      - Range: [{dmag.min():.4f}, {dmag.max():.4f}]\n")
        writer.write(f"      - Positive: {(dmag > 0).sum():,} ({(dmag > 0).mean()*100:.1f}%)\n")
        
        writer.write(f"    - AUROC:\n")
        writer.write(f"      - Mean: {auroc.mean():.4f} ± {auroc.std():.4f}\n")
        writer.write(f"      - Range: [{auroc.min():.4f}, {auroc.max():.4f}]\n")
        writer.write(f"      - Above 0.6: {(auroc > 0.6).sum():,} ({(auroc > 0.6).mean()*100:.1f}%)\n")
        writer.write(f"      - Above 0.7: {(auroc > 0.7).sum():,} ({(auroc > 0.7).mean()*100:.1f}%)\n")
        
        writer.write(f"    - Consensus Score:\n")
        writer.write(f"      - Mean: {consensus.mean():.4f} ± {consensus.std():.4f}\n")
        writer.write(f"      - Range: [{consensus.min():.4f}, {consensus.max():.4f}]\n")
        
        # Top latents analysis
        if len(top_indices) > 0:
            writer.write(f"\n   Top {min(10, len(top_indices))} Toxic Latents:\n")
            for i, (idx, score) in enumerate(zip(top_indices[:10], top_scores[:10])):
                idx = int(idx)
                writer.write(f"    {i+1:2d}. Latent {idx:4d}: consensus={score:.3f}, "
                      f"dfreq={dfreq[idx]:.3f}, dmag={dmag[idx]:.3f}, auroc={auroc[idx]:.3f}\n")

def analyze_cross_layer_comparison(writer, results):
    write_header(writer, "CROSS-LAYER COMPARISON")
    
    per_layer = results.get('per_layer', {})
    layers = list(per_layer.keys())
    
    if len(layers) < 2:
        writer.write("Need at least 2 layers for comparison\n")
        return
    
    # Collect metrics for comparison
    layer_stats = {}
    for layer_id in layers:
        layer_data = per_layer[layer_id]
        dfreq = np.array(layer_data.get('dfreq', []))
        dmag = np.array(layer_data.get('dmag', []))
        auroc = np.array(layer_data.get('auroc', []))
        consensus = np.array(layer_data.get('consensus', []))
        top_scores = layer_data.get('top_scores', [])
        
        layer_stats[layer_id] = {
            'dfreq_mean': dfreq.mean(),
            'dmag_mean': dmag.mean(),
            'auroc_mean': auroc.mean(),
            'consensus_mean': consensus.mean(),
            'top_consensus_mean': np.mean(top_scores[:10]) if top_scores else 0,
            'strong_latents': (consensus > 1.0).sum(),
            'discriminative_latents': (auroc > 0.7).sum(),
        }
    
    writer.write("Layer Comparison Summary:\n")
    writer.write(f"{'Layer':<8} {'dfreq':<8} {'dmag':<8} {'auroc':<8} {'consensus':<10} {'strong':<8} {'discrim':<8}\n")
    writer.write("-" * 70 + "\n")
    
    for layer_id in layers:
        stats = layer_stats[layer_id]
        writer.write(f"{layer_id:<8} "
              f"{stats['dfreq_mean']:<8.3f} "
              f"{stats['dmag_mean']:<8.3f} "
              f"{stats['auroc_mean']:<8.3f} "
              f"{stats['consensus_mean']:<10.3f} "
              f"{stats['strong_latents']:<8} "
              f"{stats['discriminative_latents']:<8}\n")
    
    # Find best performing layer
    best_layer = max(layers, key=lambda x: layer_stats[x]['top_consensus_mean'])
    writer.write(f"\nBest performing layer: {best_layer}\n")
    writer.write(f"   (Highest top-10 consensus score: {layer_stats[best_layer]['top_consensus_mean']:.3f})\n")

def analyze_metric_distributions(writer, results):
    write_header(writer, "METRIC DISTRIBUTION ANALYSIS")
    
    per_layer = results.get('per_layer', {})
    
    for layer_id, layer_data in per_layer.items():
        write_section(writer, f"Layer {layer_id} Distribution Analysis")
        
        dfreq = np.array(layer_data.get('dfreq', []))
        dmag = np.array(layer_data.get('dmag', []))
        auroc = np.array(layer_data.get('auroc', []))
        consensus = np.array(layer_data.get('consensus', []))
        
        # Percentile analysis
        writer.write("Percentile Analysis:\n")
        percentiles = [50, 75, 90, 95, 99]
        
        writer.write(f"{'Metric':<12} {'50th':<8} {'75th':<8} {'90th':<8} {'95th':<8} {'99th':<8}\n")
        writer.write("-" * 60 + "\n")
        
        for name, values in [('dfreq', dfreq), ('dmag', dmag), ('auroc', auroc), ('consensus', consensus)]:
            percs = [np.percentile(values, p) for p in percentiles]
            writer.write(f"{name:<12} " + " ".join(f"{p:<8.3f}" for p in percs) + "\n")
        
        # Correlation analysis
        writer.write(f"\nMetric Correlations:\n")
        metrics = {'dfreq': dfreq, 'dmag': dmag, 'auroc': auroc}
        metric_names = list(metrics.keys())
        
        writer.write(f"{'':>10}")
        for name in metric_names:
            writer.write(f"{name:>10}")
        writer.write("\n")
        
        for i, name1 in enumerate(metric_names):
            writer.write(f"{name1:>10}")
            for j, name2 in enumerate(metric_names):
                if i <= j:
                    corr = np.corrcoef(metrics[name1], metrics[name2])[0, 1]
                    writer.write(f"{corr:>10.3f}")
                else:
                    writer.write(f"{'':>10}")
            writer.write("\n")

def generate_summary_report(writer, results):
    write_header(writer, "EXECUTIVE SUMMARY", "=")
    
    per_layer = results.get('per_layer', {})
    layers = list(per_layer.keys())
    total_samples = results.get('counts', 0)
    
    # Key findings
    strong_latents_per_layer = {}
    best_auroc_per_layer = {}
    
    for layer_id, layer_data in per_layer.items():
        consensus = np.array(layer_data.get('consensus', []))
        auroc = np.array(layer_data.get('auroc', []))
        
        strong_latents_per_layer[layer_id] = (consensus > 1.0).sum()
        best_auroc_per_layer[layer_id] = auroc.max()
    
    writer.write("Key Findings:\n")
    writer.write(f"  - Analyzed {total_samples:,} samples across {len(layers)} layers\n")
    writer.write(f"  - Identified strong toxic latents (consensus > 1.0):\n")
    for layer_id in layers:
        writer.write(f"    - Layer {layer_id}: {strong_latents_per_layer[layer_id]:,} latents\n")
    
    writer.write(f"  - Best discriminative performance (max AUROC):\n")
    for layer_id in layers:
        writer.write(f"    - Layer {layer_id}: {best_auroc_per_layer[layer_id]:.3f}\n")
    
    # Recommendations
    writer.write(f"\nRecommendations:\n")
    best_layer = max(strong_latents_per_layer.keys(), 
                     key=lambda x: strong_latents_per_layer[x])
    writer.write(f"  - Focus on Layer {best_layer} (most strong latents: {strong_latents_per_layer[best_layer]:,})\n")
    writer.write(f"  - Consider latents with consensus > 1.0 for intervention\n")
    writer.write(f"  - Validate findings with larger balanced dataset\n")
    if total_samples < 1000:
        writer.write(f"   Current sample size ({total_samples:,}) may be small for robust analysis\n")

def main():
    parser = argparse.ArgumentParser(description="Analyze toxic latents selection results")
    parser.add_argument("--json", help="Path to results JSON file")
    parser.add_argument("--output", help="Output txt file path (default: auto-generated)")
    
    args = parser.parse_args()
    
    # Load and validate results
    json_path = Path(args.json)
    if not json_path.exists():
        print(f"File not found: {args.json}")
        return
    
    # Generate output filename if not provided
    if args.output:
        output_file = args.output
    else:
        output_file = json_path.parent / f"{json_path.stem}_analysis.txt"
    
    print(f"Loading results from: {args.json}")
    results = load_results(args.json)
    
    print(f"Running comprehensive analysis...")
    
    # Create writer and run all analyses
    writer = OutputWriter(output_file)
    
    # Add timestamp header
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    writer.write(f"Toxic Latents Analysis Report\n")
    writer.write(f"Generated: {timestamp}\n")
    writer.write(f"Source: {args.json}\n")
    writer.write("=" * 70 + "\n")
    
    # Run all analyses
    analyze_overall_stats(writer, results)
    analyze_layer_performance(writer, results)
    analyze_cross_layer_comparison(writer, results)
    analyze_metric_distributions(writer, results)
    generate_summary_report(writer, results)
    
    # Save to file
    writer.save_to_file()
    
    print(f"Analysis complete! Results saved to: {output_file}")
    print(f"File size: {Path(output_file).stat().st_size:,} bytes")

if __name__ == "__main__":
    main()
