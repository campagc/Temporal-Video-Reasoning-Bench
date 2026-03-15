import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==============================================================================
#  CONFIGURAZIONE
# ==============================================================================
BASE_INPUT_DIR = "output_grid_internvl_captaincook"
OUTPUT_PLOT_DIR = os.path.join(BASE_INPUT_DIR, "analysis_categories_3")
os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)

BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
LABELS = [
    '0-20% (Pessimo)',
    '20-40% (Basso)',
    '40-60% (Random/Medio)',
    '60-80% (Buono)',
    '80-100% (Eccellente)'
]

COLORS = ["#d73027", "#fc8d59", "#fee08b", "#d9ef8b", "#1a9850"]


# ==============================================================================
# 🗺️ CATEGORY MAP (criterio severo HIGH)
# ==============================================================================

CATEGORY_MAP = {
    "panfriedtofu": "Low Visual Change",
    "spicedhotchocolate": "Low Visual Change",
    "spicytunaavocadowraps": "High Visual Change",
    "breakfastburritos": "Low Visual Change",
    "tomatochutney": "Low Visual Change",
    "microwaveeggsandwich": "Low Visual Change",
    "cucumberraita": "Low Visual Change",
    "broccolistirfry": "Low Visual Change",
    "ramen": "Low Visual Change",
    "microwavefrenchtoast": "Low Visual Change",
    "coffee": "Low Visual Change",
    "mugcake": "Low Visual Change",
    "tomatomozzarellasalad": "High Visual Change",
    "blenderbananapancakes": "Low Visual Change",
    "herbomeletwithfriedtomatoes": "Low Visual Change",
    "scrambledeggs": "Low Visual Change",
    "cheesepimiento": "Low Visual Change",
    "capresebruschetta": "High Visual Change",
    "microwavemugpizza": "Low Visual Change",
    "buttercorncup": "Low Visual Change",
    "zoodles": "High Visual Change"
}


# ==============================================================================
# 1. CARICAMENTO DATI
# ==============================================================================

def extract_params(filepath):
    parts = filepath.split(os.sep)
    n_frames = next((int(p.split('_')[1]) for p in parts if 'frames_' in p), None)
    n_clips = next((int(p.split('_')[1]) for p in parts if 'clips_' in p), None)
    return n_frames, n_clips


def load_data():
    all_files = glob.glob(os.path.join(BASE_INPUT_DIR, "**", "*_metrics.csv"), recursive=True)

    print(f"Trovati {len(all_files)} file CSV.")

    df_list = []

    for f in all_files:
        frames, clips = extract_params(f)
        if frames is None or clips is None:
            continue

        try:
            temp_df = pd.read_csv(f)

            temp_df['Frames'] = frames
            temp_df['Clips'] = clips
            temp_df['Config_Label'] = f"{frames}F / {clips}C"

            # normalizzazione nomi
            temp_df['activity_norm'] = (
                temp_df['activity']
                .astype(str)
                .str.lower()
                .str.replace(" ", "", regex=False)
                .str.replace("_", "", regex=False)
            )

            temp_df['Category'] = temp_df['activity_norm'].map(CATEGORY_MAP)
            temp_df['Category'] = temp_df['Category'].fillna('Uncategorized')

            df_list.append(temp_df)

        except Exception as e:
            print(f"Errore lettura {f}: {e}")

    if not df_list:
        return pd.DataFrame()

    return pd.concat(df_list, ignore_index=True)


# ==============================================================================
# 2. GRAFICO HIGH VS LOW
# ==============================================================================

def plot_side_by_side_distribution(df):

    df_clean = df[df['Category'] != 'Uncategorized'].copy()

    df_clean['Accuracy_Band'] = pd.cut(
        df_clean['LLM_Closure_Acc'],
        bins=BINS,
        labels=LABELS,
        right=False,
        include_lowest=True
    )

    df_clean.loc[df_clean['LLM_Closure_Acc'] >= 0.8, 'Accuracy_Band'] = LABELS[-1]

    categories = ["High Visual Change", "Low Visual Change"]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)

    for i, cat in enumerate(categories):

        subset = df_clean[df_clean['Category'] == cat]

        grouped = (
            subset
            .groupby(['Config_Label', 'Accuracy_Band'], observed=False)
            .size()
            .reset_index(name='Counts')
        )

        pivot_df = grouped.pivot(
            index='Config_Label',
            columns='Accuracy_Band',
            values='Counts'
        ).fillna(0)

        pivot_pct = pivot_df.div(pivot_df.sum(axis=1), axis=0) * 100

        pivot_pct.plot(
            kind='bar',
            stacked=True,
            color=COLORS,
            edgecolor='black',
            ax=axes[i],
            legend=False
        )

        axes[i].set_title(cat)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=5)

    out_path = os.path.join(OUTPUT_PLOT_DIR, "compare_high_vs_low.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)


# ==============================================================================
# 3. TREND DELTA
# ==============================================================================

def plot_delta_comparison(df):

    plt.figure(figsize=(10, 6))

    sns.lineplot(
        data=df,
        x="Config_Label",
        y="Delta_Closure_Acc",
        hue="Category",
        markers=True
    )

    plt.axhline(0, linestyle='--')
    plt.xticks(rotation=45)

    out_path = os.path.join(OUTPUT_PLOT_DIR, "delta_trend_comparison.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)


# ==============================================================================
# 4. SUMMARY CATEGORIA
# ==============================================================================

def print_category_summary(df):

    print("\n========== CATEGORY SUMMARY ==========\n")

    summary = (
        df[df['Category'] != 'Uncategorized']
        .groupby('Category')
        .agg(
            Mean_Acc=('LLM_Closure_Acc', 'mean'),
            Mean_Delta=('Delta_Closure_Acc', 'mean'),
            Samples=('LLM_Closure_Acc', 'count'),
            Good_Rate=('LLM_Closure_Acc', lambda x: (x >= 0.6).mean())
        )
    )

    print(summary)


# ==============================================================================
# 5. LEADERBOARD RICETTE
# ==============================================================================

def build_recipe_leaderboard(df):

    df_clean = df[df['Category'] != 'Uncategorized'].copy()

    recipe_stats = (
        df_clean
        .groupby(['activity_norm', 'Category'])
        .agg(
            Mean_Acc=('LLM_Closure_Acc', 'mean'),
            Std_Acc=('LLM_Closure_Acc', 'std'),
            Mean_Delta=('Delta_Closure_Acc', 'mean'),
            Samples=('LLM_Closure_Acc', 'count'),
            Good_Rate=('LLM_Closure_Acc', lambda x: (x >= 0.6).mean())
        )
        .reset_index()
        .sort_values(by=['Mean_Acc', 'Mean_Delta'], ascending=False)
    )

    # salva CSV
    out_csv = os.path.join(OUTPUT_PLOT_DIR, "recipe_leaderboard.csv")
    recipe_stats.to_csv(out_csv, index=False)

    print("\n🔥 TOP 5")
    print(recipe_stats.head(5))

    print("\n❄️ BOTTOM 5")
    print(recipe_stats.tail(5))

    # plot
    plt.figure(figsize=(10, max(6, len(recipe_stats)*0.35)))

    colors = recipe_stats['Category'].map({
        'High Visual Change': '#2ecc71',
        'Low Visual Change': '#e74c3c'
    })

    plt.barh(recipe_stats['activity_norm'], recipe_stats['Mean_Acc'], color=colors)
    plt.gca().invert_yaxis()
    plt.xlabel("Mean Accuracy")

    out_plot = os.path.join(OUTPUT_PLOT_DIR, "recipe_leaderboard.png")
    plt.tight_layout()
    plt.savefig(out_plot, dpi=300)


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":

    print(f"🔎 Analisi directory: {BASE_INPUT_DIR}")

    df = load_data()

    if df.empty:
        print(f"Nessun dato trovato")
        exit()

    plot_side_by_side_distribution(df)
    plot_delta_comparison(df)

    print_category_summary(df)
    build_recipe_leaderboard(df)

    print(f"\n Tutto salvato in: {OUTPUT_PLOT_DIR}")
