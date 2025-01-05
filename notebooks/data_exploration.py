import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the data
df = pd.read_parquet("data/raw/train.parquet")

# Set up the style for seaborn plots
sns.set(style="whitegrid")


def make_plots_dir():
    if not os.path.exists("plots"):
        os.makedirs("plots")


plots_dir = "plots/"


# 1. Class Distribution (who wins more often: model_a or model_b?)
def plot_class_distribution(df):
    plt.figure(figsize=(8, 6))
    sns.countplot(x="winner", data=df, palette="Set2", hue="winner")
    plt.title("Class Distribution: Winner (Model A or Model B)")
    plt.xlabel("Winner (Model A or Model B)")
    plt.ylabel("Count")
    plt.savefig(plots_dir + "Class_Distribution.png", bbox_inches="tight")


# 2. Text Length Distribution (length of prompt, response_a, response_b)
def plot_text_length_distribution(df):
    df["prompt_length"] = df["prompt"].apply(len)
    df["response_a_length"] = df["response_a"].apply(len)
    df["response_b_length"] = df["response_b"].apply(len)

    plt.figure(figsize=(10, 6))
    sns.histplot(
        df["prompt_length"],
        color="blue",
        kde=True,
        label="Prompt Length",
        bins=30,
        alpha=0.6,
    )
    sns.histplot(
        df["response_a_length"],
        color="red",
        kde=True,
        label="Response A Length",
        bins=30,
        alpha=0.6,
    )
    sns.histplot(
        df["response_b_length"],
        color="green",
        kde=True,
        label="Response B Length",
        bins=30,
        alpha=0.6,
    )
    plt.title("Text Length Distribution")
    plt.xlabel("Text Length")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(plots_dir + "TLD.png", bbox_inches="tight")


# 3. Language Distribution
def plot_language_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.countplot(
        y="language",
        data=df,
        palette="Set1",
        order=df["language"].value_counts().index,
        hue="language",
    )
    plt.title("Language Distribution")
    plt.xlabel("Count")
    plt.ylabel("Language")
    plt.savefig(plots_dir + "Language_Distribution.png", bbox_inches="tight")


# 4. Sample Prompts and Responses
def display_sample_data(df, n=5):
    print(f"Displaying {n} Sample Rows of Data:")
    print(df[["prompt", "response_a", "response_b", "winner", "language"]].head(n))


plot_class_distribution(df)
plot_text_length_distribution(df)
plot_language_distribution(df)
display_sample_data(df)
