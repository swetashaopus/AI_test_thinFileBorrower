"""
Monthly Expense Detector & Category Classifier using XGBoost
with Full Explainability (SHAP)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)
from xgboost import XGBClassifier
import shap

# ─────────────────────────────────────────────
# 1. LOAD & CLEAN DATA
# ─────────────────────────────────────────────

def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.dropna(subset=['Category'], inplace=True)
    df = df[df['Category'].notna() & (df['Category'] != '')]

    # Parse dates – handle mixed formats
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df.dropna(subset=['Date'], inplace=True)

    df['Withdrawal'] = pd.to_numeric(df['Withdrawal'], errors='coerce').fillna(0)
    df['Deposit']    = pd.to_numeric(df['Deposit'],    errors='coerce').fillna(0)
    df['Balance']    = pd.to_numeric(df['Balance'],    errors='coerce').fillna(0)

    # Keep only expense rows (withdrawals)
    df = df[df['Withdrawal'] > 0].copy()
    df.reset_index(drop=True, inplace=True)
    return df


# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df['Year']          = df['Date'].dt.year
    df['Month']         = df['Date'].dt.month
    df['MonthName']     = df['Date'].dt.strftime('%b')
    df['Day']           = df['Date'].dt.day
    df['DayOfWeek']     = df['Date'].dt.dayofweek          # 0=Mon
    df['WeekOfMonth']   = (df['Day'] - 1) // 7 + 1
    df['IsWeekend']     = (df['DayOfWeek'] >= 5).astype(int)
    df['IsMonthStart']  = (df['Day'] <= 5).astype(int)
    df['IsMonthEnd']    = (df['Day'] >= 25).astype(int)
    df['Quarter']       = df['Date'].dt.quarter

    # Rolling / lag features per month
    df_sorted = df.sort_values('Date')
    df['CumMonthlyExpense'] = df_sorted.groupby(
        ['Year', 'Month'])['Withdrawal'].cumsum()

    # Balance-to-withdrawal ratio
    df['BalanceRatio'] = df['Balance'] / (df['Withdrawal'] + 1)

    # Log-transform skewed amount
    df['LogWithdrawal'] = np.log1p(df['Withdrawal'])

    return df


# ─────────────────────────────────────────────
# 3. MONTHLY EXPENSE SUMMARY
# ─────────────────────────────────────────────

def monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (df.groupby(['Year', 'Month', 'MonthName', 'Category'])
                 ['Withdrawal'].sum()
                 .reset_index()
                 .rename(columns={'Withdrawal': 'TotalExpense'}))
    summary['Period'] = summary['MonthName'] + ' ' + summary['Year'].astype(str)
    return summary


# ─────────────────────────────────────────────
# 4. MODEL: EXPENSE CATEGORY CLASSIFIER
# ─────────────────────────────────────────────

FEATURE_COLS = [
    'Withdrawal', 'LogWithdrawal', 'Balance', 'BalanceRatio',
    'Month', 'Day', 'DayOfWeek', 'WeekOfMonth',
    'IsWeekend', 'IsMonthStart', 'IsMonthEnd', 'Quarter',
    'CumMonthlyExpense'
]

def train_classifier(df: pd.DataFrame):
    le = LabelEncoder()
    y  = le.fit_transform(df['Category'])
    X  = df[FEATURE_COLS]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred, average='weighted')

    print("\n" + "="*55)
    print("  XGBoost Category Classifier — Results")
    print("="*55)
    print(f"  Accuracy : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  F1-Score : {f1:.4f}  (weighted)")
    print("\n  Classification Report:")
    present_labels = sorted(set(y_test) | set(y_pred))
    print(classification_report(y_test, y_pred,
                                labels=present_labels,
                                target_names=le.classes_[present_labels]))

    return model, le, X_train, X_test, y_train, y_test, y_pred


# ─────────────────────────────────────────────
# 5. VISUALISATIONS
# ─────────────────────────────────────────────

def plot_all(df, summary, model, le, X_train, X_test, y_test, y_pred,
             out_path="expense_analysis_report.png"):

    # ── colour palette
    CAT_COLORS = {
        'Food': '#FF6B6B', 'Misc': '#4ECDC4', 'Rent': '#45B7D1',
        'Shopping': '#96CEB4', 'Transport': '#FFEAA7', 'Salary': '#DDA0DD'
    }

    fig = plt.figure(figsize=(24, 28))
    fig.patch.set_facecolor('#0F0F23')
    gs  = gridspec.GridSpec(4, 3, figure=fig,
                            hspace=0.45, wspace=0.35)

    title_kw  = dict(color='white', fontsize=13, fontweight='bold', pad=10)
    label_kw  = dict(color='#CCCCCC', fontsize=9)
    tick_kw   = dict(colors='#AAAAAA', labelsize=8)

    def style_ax(ax, title):
        ax.set_facecolor('#1A1A2E')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333355')
        ax.tick_params(axis='both', **tick_kw)
        ax.set_title(title, **title_kw)
        ax.xaxis.label.set_color('#CCCCCC')
        ax.yaxis.label.set_color('#CCCCCC')

    # ── 1. Monthly total expenses (line)
    ax1 = fig.add_subplot(gs[0, :2])
    monthly_total = (summary.groupby(['Year', 'Month'])['TotalExpense']
                     .sum().reset_index())
    monthly_total['Label'] = (pd.to_datetime(
        monthly_total[['Year','Month']].assign(day=1))
        .dt.strftime('%b %Y'))
    ax1.plot(monthly_total['Label'], monthly_total['TotalExpense'],
             color='#00D4FF', marker='o', linewidth=2.5, markersize=6)
    ax1.fill_between(range(len(monthly_total)),
                     monthly_total['TotalExpense'], alpha=0.15, color='#00D4FF')
    ax1.set_xticks(range(len(monthly_total)))
    ax1.set_xticklabels(monthly_total['Label'], rotation=45, ha='right', fontsize=7)
    ax1.set_ylabel('Total Expense (₹)', **label_kw)
    style_ax(ax1, '📈 Monthly Total Expenses')

    # ── 2. Category pie
    ax2 = fig.add_subplot(gs[0, 2])
    cat_totals = summary.groupby('Category')['TotalExpense'].sum()
    colors_pie = [CAT_COLORS.get(c, '#888888') for c in cat_totals.index]
    wedges, texts, autotexts = ax2.pie(
        cat_totals, labels=cat_totals.index, autopct='%1.1f%%',
        colors=colors_pie, startangle=140,
        textprops={'color': 'white', 'fontsize': 8})
    for at in autotexts:
        at.set_fontsize(7)
    ax2.set_facecolor('#1A1A2E')
    style_ax(ax2, '🥧 Expense by Category')

    # ── 3. Monthly stacked bar by category
    ax3 = fig.add_subplot(gs[1, :])
    pivot = (summary.groupby(['Month', 'Category'])['TotalExpense']
             .sum().unstack(fill_value=0))
    pivot.index = [pd.Timestamp(2023, m, 1).strftime('%b') for m in pivot.index]
    bottom = np.zeros(len(pivot))
    for cat in pivot.columns:
        color = CAT_COLORS.get(cat, '#888888')
        ax3.bar(pivot.index, pivot[cat], bottom=bottom,
                label=cat, color=color, alpha=0.85, edgecolor='#222244', linewidth=0.4)
        bottom += pivot[cat].values
    ax3.legend(loc='upper right', facecolor='#1A1A2E',
               labelcolor='white', fontsize=8, framealpha=0.7)
    ax3.set_ylabel('Expense (₹)', **label_kw)
    style_ax(ax3, '📊 Monthly Expense Stack by Category')

    # ── 4. Confusion matrix
    ax4 = fig.add_subplot(gs[2, 0])
    cm   = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_,
                ax=ax4, cbar=False, annot_kws={'size': 8})
    ax4.set_xlabel('Predicted', **label_kw)
    ax4.set_ylabel('Actual', **label_kw)
    ax4.tick_params(axis='x', rotation=45, **tick_kw)
    ax4.tick_params(axis='y', rotation=0, **tick_kw)
    style_ax(ax4, '🎯 Confusion Matrix')

    # ── 5. Feature importance (gain)
    ax5 = fig.add_subplot(gs[2, 1])
    fi   = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values()
    colors_fi = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(fi)))
    fi.plot(kind='barh', ax=ax5, color=colors_fi, edgecolor='none')
    ax5.set_xlabel('Importance (gain)', **label_kw)
    style_ax(ax5, '⚡ Feature Importance (XGBoost)')

    # ── 6. Top-5 expense days heatmap
    ax6 = fig.add_subplot(gs[2, 2])
    heat_data = df.pivot_table(
        values='Withdrawal', index='DayOfWeek', columns='Month',
        aggfunc='sum', fill_value=0)
    heat_data.index = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    heat_data.columns = [pd.Timestamp(2023, m, 1).strftime('%b')
                         for m in heat_data.columns]
    sns.heatmap(heat_data, cmap='YlOrRd', ax=ax6, cbar=True,
                annot=False, linewidths=0.3, linecolor='#0F0F23')
    ax6.set_xlabel('Month', **label_kw)
    ax6.set_ylabel('Day of Week', **label_kw)
    style_ax(ax6, '🗓️ Spend Heatmap (Day × Month)')

    # ── 7. SHAP summary bar
    ax7 = fig.add_subplot(gs[3, :2])
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    sv = np.array(explainer.shap_values(X_test))  # (n_samples, n_features, n_classes)
    mean_shap = np.abs(sv).mean(axis=(0, 2)) if sv.ndim == 3 else np.abs(sv).mean(0)
    shap_df = pd.Series(mean_shap, index=FEATURE_COLS).sort_values()
    colors_shap = plt.cm.plasma(np.linspace(0.2, 0.85, len(shap_df)))
    shap_df.plot(kind='barh', ax=ax7, color=colors_shap, edgecolor='none')
    ax7.set_xlabel('Mean |SHAP value|', **label_kw)
    style_ax(ax7, '🔍 SHAP Explainability — Feature Impact on Category Prediction')

    # ── 8. Category avg spend
    ax8 = fig.add_subplot(gs[3, 2])
    cat_avg = df.groupby('Category')['Withdrawal'].mean().sort_values()
    colors_avg = [CAT_COLORS.get(c, '#888888') for c in cat_avg.index]
    cat_avg.plot(kind='barh', ax=ax8, color=colors_avg, edgecolor='none')
    ax8.set_xlabel('Avg Transaction (₹)', **label_kw)
    style_ax(ax8, '💰 Avg Expense per Category')

    # ── Super title
    fig.suptitle('💳  Personal Expense Intelligence Dashboard  |  XGBoost + SHAP',
                 color='white', fontsize=18, fontweight='bold', y=0.995)

    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  Dashboard saved → {out_path}")


# ─────────────────────────────────────────────
# 6. MONTHLY EXPENSE REPORT (text)
# ─────────────────────────────────────────────

def print_monthly_report(summary: pd.DataFrame):
    print("\n" + "="*60)
    print("  MONTHLY EXPENSE REPORT")
    print("="*60)
    for (yr, mo), grp in summary.groupby(['Year', 'Month']):
        period = f"{pd.Timestamp(yr, mo, 1).strftime('%B %Y')}"
        total  = grp['TotalExpense'].sum()
        print(f"\n  📅 {period}  |  Total: ₹{total:,.2f}")
        for _, row in grp.sort_values('TotalExpense', ascending=False).iterrows():
            pct = row['TotalExpense'] / total * 100
            bar = '█' * int(pct / 5)
            print(f"     {row['Category']:<12} ₹{row['TotalExpense']:>8,.2f}  "
                  f"({pct:5.1f}%)  {bar}")


# ─────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────

def main():
    DATA_PATH = "MyTransaction.csv"
    OUT_IMG   = "expense_analysis_report.png"

    print("Loading data …")
    df = load_and_clean(DATA_PATH)
    df = engineer_features(df)

    print(f"  Rows after cleaning : {len(df)}")
    print(f"  Categories found    : {sorted(df['Category'].unique())}")

    summary = monthly_summary(df)
    print_monthly_report(summary)

    print("\nTraining XGBoost classifier …")
    model, le, X_train, X_test, y_train, y_test, y_pred = train_classifier(df)

    print("\nGenerating dashboard …")
    plot_all(df, summary, model, le, X_train, X_test, y_test, y_pred,
             out_path=OUT_IMG)

    print("\n✅  All done!")


if __name__ == "__main__":
    main()