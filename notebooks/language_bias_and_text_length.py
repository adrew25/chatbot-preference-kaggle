import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os


def make_plots_dir():
    if not os.path.exists("plots"):
        os.makedirs("plots")


plots_dir = "plots/"


df = pd.read_parquet("data/raw/train.parquet")


def plot_language_bias(df):
    # Count the occurrences of winner per language
    language_winner_counts = (
        df.groupby(["language", "winner"]).size().unstack().fillna(0)
    )

    # Normalize to get percentage
    language_winner_counts_percentage = language_winner_counts.div(
        language_winner_counts.sum(axis=1), axis=0
    )

    # Plotting
    plt.figure(figsize=(12, 6))
    language_winner_counts_percentage.plot(
        kind="bar", stacked=True, color=["#FF9999", "#66B2FF"], figsize=(12, 6)
    )
    plt.title("Language Bias - Distribution of Winner by Language")
    plt.ylabel("Percentage of Winner")
    plt.xlabel("Language")
    plt.xticks(rotation=90)
    plt.legend(title="Winner", labels=["Response A", "Response B"])
    plt.tight_layout()
    plt.savefig(plots_dir + "Language_Bias.png", bbox_inches="tight")


def plot_text_length_influence(df):
    # Create a new column for text length
    df["len_response_a"] = df["response_a"].apply(len)
    df["len_response_b"] = df["response_b"].apply(len)

    # Prepare a new DataFrame for plotting by melting the data into a long format
    df_melted = df.melt(
        id_vars=["winner"],
        value_vars=["len_response_a", "len_response_b"],
        var_name="response",
        value_name="text_length",
    )

    # Map 'len_response_a' and 'len_response_b' to 'Response A' and 'Response B' for better readability
    df_melted["response"] = df_melted["response"].map(
        {"len_response_a": "Response A", "len_response_b": "Response B"}
    )

    # Summary print
    summary = (
        df_melted.groupby(["winner", "response"])
        .agg(
            mean_length=("text_length", "mean"),
            median_length=("text_length", "median"),
            std_length=("text_length", "std"),
            count=("text_length", "count"),
        )
        .reset_index()
    )

    print(summary)

    # Plotting the distribution of response lengths based on the winner
    plt.figure(figsize=(12, 6))

    # Use boxplot to show the distribution
    sns.boxplot(
        x="winner",
        y="text_length",
        data=df_melted,
        hue="response",
        palette="Set2",
        showfliers=False,
    )

    # Title and labels
    plt.title("Text Length Influence on Winner")
    plt.ylabel("Response Length")
    plt.xlabel("Winner")

    # Adjust the legend and plot layout
    plt.legend(title="Response", loc="upper left")
    plt.tight_layout()

    plt.savefig(plots_dir + "Text_Length_Influence.png", bbox_inches="tight")


plot_language_bias(df)
plot_text_length_influence(df)
