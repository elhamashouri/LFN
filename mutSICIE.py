import pandas as pd
import random

vcf_file = "./P12-Publication.vcf"

def parse_vcf(file_path):
    with open(file_path, 'r') as vcf:
        lines = [line.strip() for line in vcf if not line.startswith("##")]
    header = lines[0].split("\t")
    data = [line.split("\t") for line in lines[1:]]
    return pd.DataFrame(data, columns=header)

def vcf_to_mutation_matrix(vcf_df, max_rows=50):
    genotype_columns = vcf_df.columns[9:]

    if len(vcf_df) > max_rows:
        vcf_df = vcf_df.sample(n=max_rows, random_state=42)

    mutation_matrix = []
    
    for _, row in vcf_df.iterrows():
        mutation_row = []
        for col in genotype_columns:
            genotype = row[col].split(":")[0
            if genotype == "0/0":
                mutation_row.append(0)
            elif genotype in ["0/1", "1/1", "1/0"]:
                mutation_row.append(1)
            else:
                mutation_row.append(3)
        mutation_matrix.append(mutation_row)
    
    return pd.DataFrame(mutation_matrix, columns=genotype_columns)

vcf_df = parse_vcf(vcf_file)
mutation_matrix = vcf_to_mutation_matrix(vcf_df, max_rows=50)

output_file = "mutation_matrix_50.csv"
mutation_matrix.to_csv(output_file, index=False, header=False, sep=' ')
print(f"Mutation matrix saved to {output_file}")

try:
    matrix = pd.read_csv("./mutation_matrix_50.csv", header=None, delim_whitespace=True)
    print("Matrix Shape:", matrix.shape)
    print("Sample Data:\n", matrix.head())
    # Check for invalid entries
    valid_values = {0, 1, 3}
    if not set(matrix.values.flatten()).issubset(valid_values):
        print("Invalid entries found in the matrix!")
    else:
        print("Matrix format is valid.")
except Exception as e:
    print("Error reading the file:", e)
