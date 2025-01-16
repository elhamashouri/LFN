import pandas as pd
import vcf
from ete3 import Tree, TreeStyle, NodeStyle
import numpy as np
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, to_tree


def load_vcf(vcf_file):
    vcf_reader = vcf.Reader(open(vcf_file, 'r'))
    variants = []
    for record in vcf_reader:
        variant = {
            'CHROM': record.CHROM,
            'POS': record.POS,
            'ID': record.ID,
            'REF': record.REF,
            'ALT': [str(alt) for alt in record.ALT],
            'QUAL': record.QUAL,
            'FILTER': record.FILTER,
            'INFO': record.INFO,
            'FORMAT': record.FORMAT,
        }
        for sample in record.samples:
            variant[sample.sample] = sample.data._asdict()  
        variants.append(variant)
    return pd.DataFrame(variants)

def load_cnv(cnv_file):
    return pd.read_csv(cnv_file, sep="\t", names=['Region', 'CellID', 'CopyNumber'], dtype=str)

def analyze_data(vcf_data, cnv_data):
    num_samples = len([col for col in vcf_data.columns if col not in ['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO', 'FORMAT']])
    num_loci = vcf_data['POS'].nunique()
    genome_length = num_loci * 100000
    mutation_rate = len(vcf_data) / genome_length if genome_length > 0 else 0
    avg_quality = vcf_data['QUAL'].mean()

    cnv_data['CopyNumber'] = pd.to_numeric(cnv_data['CopyNumber'], errors='coerce')
    effective_population_size = cnv_data['CopyNumber'].mean() * 1000
    cnv_distribution = cnv_data['CopyNumber'].value_counts().to_dict()

    return {
        'n': 1,
        's': num_samples,
        'l': num_loci,
        'e': effective_population_size,
        'u': mutation_rate,
        'demographics': cnv_distribution,
        'g': avg_quality * 1e-5,
    }

def generate_tree(vcf_data, max_loci=500, num_samples=10):
    genotype_columns = [col for col in vcf_data.columns if col not in ['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO', 'FORMAT']]

    sampled_vcf = vcf_data.sample(n=max_loci, random_state=42)
    print(f"Randomly sampled {max_loci} loci for tree generation.")

    format_field = sampled_vcf['FORMAT'].iloc[0].split(':')
    gt_index = format_field.index('GT') if 'GT' in format_field else None
    if gt_index is None:
        raise ValueError("GT field not found in FORMAT column.")
    binary_matrix = []
    for i in sampled_vcf.index:
        locus_variants = []
        for col in genotype_columns:
            sample_data = sampled_vcf.at[i, col].get('GT', './.')  
            if "1" in sample_data:  
                locus_variants.append(1)
            elif sample_data == "0/0": 
                locus_variants.append(0)
            else:
                locus_variants.append(0)  
        binary_matrix.append(locus_variants)

    binary_matrix = np.array(binary_matrix)  
    print(f"Binary matrix shape: {binary_matrix.shape}")

    distances = pdist(binary_matrix.T, metric='hamming')  
    linkage_matrix = linkage(distances, method='average')
    tree_root, _ = to_tree(linkage_matrix, rd=True)

    # Convert to ete3 Tree format
    def build_ete_tree(node):
        t = Tree()
        stack = [(t, node)]
        while stack:
            parent, current_node = stack.pop()
            if current_node.is_leaf():
                parent.add_child(name=f"Sample_{current_node.id}")
            else:
                child = parent.add_child()
                stack.append((child, current_node.get_left()))
                stack.append((child, current_node.get_right()))
        return t

    ete_tree = build_ete_tree(tree_root)
    return ete_tree



def visualize_tree(tree):
    ts = TreeStyle()
    ts.show_leaf_name = True
    ts.scale = 20
    for node in tree.traverse():
        nstyle = NodeStyle()
        nstyle['shape'] = 'circle'
        nstyle['size'] = 10
        node.set_style(nstyle)
    tree.show(tree_style=ts)

# Main function
def main():
    vcf_file = "./P12-Publication.vcf"
    cnv_file = "./P12-Publication-CNVs.Ginkgo.txt"

    vcf_data = load_vcf(vcf_file)
    cnv_data = load_cnv(cnv_file)

    parameters = analyze_data(vcf_data, cnv_data)
    print("Suggested Parameters for CellCoal:")
    for key, value in parameters.items():
        print(f"{key}: {value}")

    tree = generate_tree(vcf_data)
    visualize_tree(tree)

if __name__ == "__main__":
    main()
