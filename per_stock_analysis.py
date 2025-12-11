"""
Per-Stock Performance Analysis
Analyze model predictions broken down by individual stocks
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List
import json
import os


def analyze_per_stock_accuracy(
    model,
    test_data: Dict,
    device: torch.device,
    output_dir: str = None
) -> pd.DataFrame:
    """
    Analyze model accuracy broken down by individual stock ticker.
    
    Args:
        model: Trained model
        test_data: Dict with 'x', 'lengths', 'direction', 'magnitude', 'meta'
        device: torch device
        output_dir: Directory to save results (optional)
    
    Returns:
        DataFrame with per-stock metrics
    """
    model.eval()
    
    # Get predictions
    with torch.no_grad():
        x = test_data['x'].to(device)
        lengths = test_data['lengths'].to(device)
        outputs = model(x, lengths)
        
        pred_dir = (outputs['direction_prob'] > 0.5).cpu().numpy().flatten()
        pred_mag = outputs['magnitude'].cpu().numpy().flatten()
    
    actual_dir = test_data['direction'].numpy()
    actual_mag = test_data['magnitude'].numpy()
    meta = test_data['meta']
    
    # Group by ticker
    stock_data = defaultdict(lambda: {
        'correct': 0,
        'total': 0,
        'mag_errors': [],
        'actual_returns': [],
        'pred_returns': [],
        'window_dates': []
    })
    
    for i, m in enumerate(meta):
        ticker = m['ticker']
        is_correct = (pred_dir[i] == actual_dir[i])
        mag_error = abs(pred_mag[i] - actual_mag[i])
        
        stock_data[ticker]['correct'] += int(is_correct)
        stock_data[ticker]['total'] += 1
        stock_data[ticker]['mag_errors'].append(mag_error)
        stock_data[ticker]['actual_returns'].append(actual_mag[i])
        stock_data[ticker]['pred_returns'].append(pred_mag[i])
        stock_data[ticker]['window_dates'].append(m.get('window_end_date', ''))
    
    # Build results DataFrame
    results = []
    for ticker, data in stock_data.items():
        accuracy = data['correct'] / data['total']
        mae = np.mean(data['mag_errors'])
        
        results.append({
            'ticker': ticker,
            'accuracy': accuracy,
            'correct': data['correct'],
            'total': data['total'],
            'mae': mae,
            'mean_actual_return': np.mean(data['actual_returns']),
            'mean_pred_return': np.mean(data['pred_returns']),
        })
    
    df = pd.DataFrame(results).sort_values('accuracy', ascending=False)
    
    # Print summary
    print("="*80)
    print("PER-STOCK PERFORMANCE ANALYSIS")
    print("="*80)
    print(f"\nTotal Stocks: {len(df)}")
    print(f"Overall Accuracy: {(pred_dir == actual_dir).mean():.2%}")
    print(f"\nTop 5 Performing Stocks:")
    print(df.head(5)[['ticker', 'accuracy', 'correct', 'total', 'mae']].to_string(index=False))
    print(f"\nBottom 5 Performing Stocks:")
    print(df.tail(5)[['ticker', 'accuracy', 'correct', 'total', 'mae']].to_string(index=False))
    
    # Save results
    if output_dir:
        df.to_csv(os.path.join(output_dir, 'per_stock_accuracy.csv'), index=False)
        print(f"\n✅ Saved results to {os.path.join(output_dir, 'per_stock_accuracy.csv')}")
    
    return df, stock_data


def visualize_per_stock_performance(
    df: pd.DataFrame,
    stock_data: Dict,
    output_dir: str = None
):
    """Create visualizations of per-stock performance"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Accuracy by stock (bar chart)
    ax1 = axes[0, 0]
    df_sorted = df.sort_values('accuracy', ascending=True)
    colors = ['red' if acc < 0.5 else 'orange' if acc < 0.55 else 'green' 
              for acc in df_sorted['accuracy']]
    ax1.barh(range(len(df_sorted)), df_sorted['accuracy'], color=colors, alpha=0.7)
    ax1.set_yticks(range(len(df_sorted)))
    ax1.set_yticklabels(df_sorted['ticker'], fontsize=8)
    ax1.axvline(0.5, color='black', linestyle='--', alpha=0.5, label='Random (50%)')
    ax1.set_xlabel('Direction Accuracy')
    ax1.set_title('Direction Accuracy by Stock')
    ax1.set_xlim(0, 1)
    ax1.legend()
    ax1.grid(alpha=0.3, axis='x')
    
    # 2. MAE by stock
    ax2 = axes[0, 1]
    df_sorted_mae = df.sort_values('mae', ascending=True)
    ax2.barh(range(len(df_sorted_mae)), df_sorted_mae['mae'], alpha=0.7, color='purple')
    ax2.set_yticks(range(len(df_sorted_mae)))
    ax2.set_yticklabels(df_sorted_mae['ticker'], fontsize=8)
    ax2.set_xlabel('Magnitude MAE')
    ax2.set_title('Magnitude Error by Stock')
    ax2.grid(alpha=0.3, axis='x')
    
    # 3. Scatter: Accuracy vs Number of Samples
    ax3 = axes[1, 0]
    ax3.scatter(df['total'], df['accuracy'], s=100, alpha=0.6)
    for _, row in df.iterrows():
        ax3.annotate(row['ticker'], (row['total'], row['accuracy']), 
                    fontsize=8, alpha=0.7, ha='right')
    ax3.set_xlabel('Number of Test Windows')
    ax3.set_ylabel('Direction Accuracy')
    ax3.set_title('Accuracy vs Sample Size\n(More data = more reliable?)')
    ax3.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Random')
    ax3.grid(alpha=0.3)
    ax3.legend()
    
    # 4. Distribution of accuracies
    ax4 = axes[1, 1]
    ax4.hist(df['accuracy'], bins=20, alpha=0.7, color='steelblue', edgecolor='black')
    ax4.axvline(df['accuracy'].mean(), color='red', linestyle='--', 
               label=f"Mean: {df['accuracy'].mean():.2%}", linewidth=2)
    ax4.axvline(0.5, color='orange', linestyle='--', 
               label='Random (50%)', linewidth=2)
    ax4.set_xlabel('Direction Accuracy')
    ax4.set_ylabel('Number of Stocks')
    ax4.set_title('Distribution of Per-Stock Accuracies')
    ax4.legend()
    ax4.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_dir:
        save_path = os.path.join(output_dir, 'per_stock_performance.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved visualization to {save_path}")
    
    plt.show()
    
    # Additional stats
    print("\n" + "="*80)
    print("STATISTICAL SUMMARY")
    print("="*80)
    print(f"\nAccuracy Statistics:")
    print(f"  Mean: {df['accuracy'].mean():.2%}")
    print(f"  Median: {df['accuracy'].median():.2%}")
    print(f"  Std Dev: {df['accuracy'].std():.2%}")
    print(f"  Min: {df['accuracy'].min():.2%} ({df.loc[df['accuracy'].idxmin(), 'ticker']})")
    print(f"  Max: {df['accuracy'].max():.2%} ({df.loc[df['accuracy'].idxmax(), 'ticker']})")
    
    print(f"\nStocks performing better than random (>50%):")
    above_random = df[df['accuracy'] > 0.5]
    print(f"  Count: {len(above_random)}/{len(df)} ({len(above_random)/len(df):.1%})")
    
    print(f"\nMAE Statistics:")
    print(f"  Mean: {df['mae'].mean():.4f} ({df['mae'].mean()*100:.2f}%)")
    print(f"  Best: {df['mae'].min():.4f} ({df.loc[df['mae'].idxmin(), 'ticker']})")
    print(f"  Worst: {df['mae'].max():.4f} ({df.loc[df['mae'].idxmax(), 'ticker']})")


def analyze_stock_predictions_over_time(
    stock_data: Dict,
    ticker: str,
    output_dir: str = None
):
    """Analyze predictions for a specific stock over time"""
    
    if ticker not in stock_data:
        print(f"❌ Ticker {ticker} not found in data")
        return
    
    data = stock_data[ticker]
    dates = data['window_dates']
    actual = np.array(data['actual_returns'])
    pred = np.array(data['pred_returns'])
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # 1. Time series of actual vs predicted
    ax1 = axes[0]
    x_pos = range(len(dates))
    ax1.plot(x_pos, actual, marker='o', label='Actual', alpha=0.7, linewidth=2)
    ax1.plot(x_pos, pred, marker='x', label='Predicted', alpha=0.7, linewidth=2)
    ax1.set_xlabel('Window Index')
    ax1.set_ylabel('Return Magnitude')
    ax1.set_title(f'{ticker} - Actual vs Predicted Returns Over Time')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. Error over time
    ax2 = axes[1]
    errors = np.abs(actual - pred)
    ax2.bar(x_pos, errors, alpha=0.7, color='red')
    ax2.axhline(errors.mean(), color='blue', linestyle='--', 
               label=f'Mean Error: {errors.mean():.4f}', linewidth=2)
    ax2.set_xlabel('Window Index')
    ax2.set_ylabel('Absolute Error')
    ax2.set_title(f'{ticker} - Prediction Errors Over Time')
    ax2.legend()
    ax2.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_dir:
        save_path = os.path.join(output_dir, f'{ticker}_time_series.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved {ticker} analysis to {save_path}")
    
    plt.show()
    
    # Stats
    print(f"\n{ticker} Statistics:")
    print(f"  Total Predictions: {len(actual)}")
    print(f"  Direction Accuracy: {data['correct']}/{data['total']} ({data['correct']/data['total']:.2%})")
    print(f"  MAE: {errors.mean():.4f} ({errors.mean()*100:.2f}%)")
    print(f"  Mean Actual Return: {actual.mean():.4f}")
    print(f"  Mean Predicted Return: {pred.mean():.4f}")


# Usage example
if __name__ == "__main__":
    # Example usage (you'd run this after training):
    """
    # Load your test data
    test_data = torch.load('tensors/run_YYYYMMDD_HHMMSS/test.pt')
    
    # Analyze per-stock performance
    df, stock_data = analyze_per_stock_accuracy(model, test_data, device, output_dir='results')
    
    # Visualize
    visualize_per_stock_performance(df, stock_data, output_dir='results')
    
    # Deep dive into specific stocks
    for ticker in ['AAPL', 'GOOGL', 'TSLA']:
        analyze_stock_predictions_over_time(stock_data, ticker, output_dir='results')
    """
    pass
