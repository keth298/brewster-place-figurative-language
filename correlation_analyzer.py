from scipy.stats import pearsonr

TRAITS = ("racial_consciousness", "emotional_register", "domestic_score")


def run_correlations(merged_df):
    required = list(TRAITS) + ["figurative_per_1000_words"]
    if merged_df[required].isnull().any().any():
        raise ValueError("merged_df contains NaN values in correlation columns")
    results = []
    for trait in TRAITS:
        r, p = pearsonr(merged_df[trait], merged_df["figurative_per_1000_words"])
        results.append({"trait": trait, "r": float(r), "p_value": float(p)})
    return results


def print_correlations(results):
    print("Correlation Results:")
    for item in results:
        print(f"  {item['trait']:<22} vs figurative_per_1000_words:  r={item['r']:.4f}, p={item['p_value']:.4f}")
