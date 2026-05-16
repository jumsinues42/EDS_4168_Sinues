import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Global tracking arrays needed explicitly by the freestanding animation frame updating function
_global_filtered_df = pd.DataFrame()
_global_point_handle = None

# =====================================================================
# FUNCTION 1: DATA INGESTION MODULE
# =====================================================================
def ingest_data(file_name):
    """Loads raw engineering dataset from the localized system path."""
    path = os.path.join('data', file_name)
    df = pd.read_csv(path)
    print(f"✅ [FUNCTION 1] Ingested raw dataset: {file_name}")
    return df

# =====================================================================
# FUNCTION 2: PIPELINE DATA CLEANING MODULE
# =====================================================================
def clean_data(df):
    """Cleans columns, fixes data types, and instantiates sequence metrics."""
    df.columns = [c.strip() for c in df.columns] 
    
    tps_match = [c for c in df.columns if 'TPS' in c or 'Tokens' in c or 'Throughput' in c]
    lat_match = [c for c in df.columns if 'Latency' in c or 'Time' in c]
    mod_match = [c for c in df.columns if 'Model' in c or 'Name' in c]

    if not tps_match or not lat_match or not mod_match:
        raise KeyError(f"Missing required columns! Found: {df.columns.tolist()}")

    df = df.rename(columns={tps_match[0]: 'TPS', lat_match[0]: 'Latency', mod_match[0]: 'Model'})
    df['Provider_Group'] = df['Model'].apply(lambda x: str(x).split()[0] if pd.notnull(x) else "Unknown")
    
    df['TPS'] = pd.to_numeric(df['TPS'].astype(str).str.replace(r'[$, ]', '', regex=True), errors='coerce')
    df['Latency'] = pd.to_numeric(df['Latency'].astype(str).str.replace(r'[$, ]', '', regex=True), errors='coerce')
    df['Sequence_Step'] = np.arange(len(df)) * 100
    
    cleaned_df = df.dropna(subset=['TPS', 'Latency']).copy()
    print("✅ [FUNCTION 2] Auto-cleaning and type correction pipeline complete.")
    return cleaned_df

# =====================================================================
# FUNCTION 3: NUMPY STATISTICAL ANALYSIS ENGINE
# =====================================================================
def compute_engineering_statistics(df):
    """Computes foundational statistics utilizing raw NumPy arrays."""
    tps_array = df['TPS'].values
    latency_array = df['Latency'].values

    stats = {
        'tps_mean': np.mean(tps_array),
        'tps_median': np.median(tps_array),
        'tps_std': np.std(tps_array),
        'tps_var': np.var(tps_array),
        'lat_mean': np.mean(latency_array),
        'lat_median': np.median(latency_array),
        'lat_std': np.std(latency_array),
        'lat_var': np.var(latency_array)
    }

    print("\n=======================================================")
    print("📊 [FUNCTION 3] NUMPY CALCULATED SYSTEM TELEMETRY METRICS")
    print("=======================================================")
    print(f"Throughput (TPS)   -> Mean: {stats['tps_mean']:.4f} | Median: {stats['tps_median']:.4f} | StdDev: {stats['tps_std']:.4f} | Variance: {stats['tps_var']:.4f}")
    print(f"Latency (Seconds)  -> Mean: {stats['lat_mean']:.4f} | Median: {stats['lat_median']:.4f} | StdDev: {stats['lat_std']:.4f} | Variance: {stats['lat_var']:.4f}")
    print("=======================================================\n")
    return stats

# =====================================================================
# FUNCTION 4: MACHINE LEARNING DECAY MODEL ANALYSIS
# =====================================================================
def run_decay_model(df):
    """Fits an ML Linear Regression to extract token generation decay rates."""
    try:
        X = df[['Sequence_Step']].values
        y = df['TPS'].values
        model = LinearRegression().fit(X, y)
        print(f"📉 [FUNCTION 4] Regression computed. Coefficient: {model.coef_[0]:.6f} TPS/token")
        return model
    except Exception as e:
        print(f"❌ [FUNCTION 4] Model Fitting Aborted: {e}")
        return None

# =====================================================================
# FUNCTION 5: TIME-VARIANT ANIMATION RENDER ENGINE
# =====================================================================
def update_animation_frames(frame_index, active_axis, visualization_mode):
    """Handles time-variant real-time graph mutations based on processing modes."""
    if visualization_mode == "scan":
        active_axis.clear()
        active_axis.set_title(f"Sequence Scan Profile: {frame_index * 10}%")
        active_axis.set_xlabel("Tokens Generated")
        active_axis.set_ylabel("Throughput (TPS)")
        slice_limit = int(len(_global_filtered_df) * (frame_index / 10))
        if slice_limit > 0:
            sns.scatterplot(data=_global_filtered_df.iloc[:slice_limit], x='Sequence_Step', y='TPS', ax=active_axis, color='teal')
            
    elif visualization_mode == "pulse":
        _global_point_handle.set_data(
            [_global_filtered_df['Sequence_Step'].iloc[frame_index]], 
            [_global_filtered_df['TPS'].iloc[frame_index]]
        )
        return _global_point_handle,

# =====================================================================
# AUXILIARY MODULE: STATIC VISUALIZATION PLOTTER
# =====================================================================
def generate_static_plots(full_df, filtered_df, model):
    """Generates the 6 mandatory static figures with customized layout padding."""
    os.makedirs('outputs/static_plots', exist_ok=True)
    os.makedirs('outputs/animations', exist_ok=True)
    sns.set_theme(style="darkgrid")

    # FIG 1: PIE (Model Family Share)
    plt.figure(figsize=(10, 8))
    pie_data = full_df['Provider_Group'].value_counts().nlargest(8)
    plt.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("Set3"))
    plt.title('Figure 1: Model Family Market Share')
    plt.savefig('outputs/static_plots/1_pie.png')
    plt.close()

    # FIG 2: DECAY SCATTER (The AIP-04 Regression Line Proof)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=filtered_df, x='Sequence_Step', y='TPS', alpha=0.6, color='blue', label='Actual Telemetry Data')
    X_line = np.linspace(filtered_df['Sequence_Step'].min(), filtered_df['Sequence_Step'].max(), 100).reshape(-1, 1)
    y_line = model.predict(X_line)
    plt.plot(X_line, y_line, color='red', linewidth=3, label='Performance Decay Trendline')
    plt.title('Figure 2: Token Generation Decay Analysis (AIP-04)')
    plt.xlabel('Tokens Generated')
    plt.ylabel('Throughput (TPS)')
    plt.legend()
    plt.savefig('outputs/static_plots/2_decay_scatter.png')
    plt.close()

    # FIG 3: BAR (Throughput Rankings)
    plt.figure(figsize=(12, 7))
    sns.barplot(data=filtered_df.nlargest(10, 'TPS'), x='Model', y='TPS', palette='magma')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.title('Figure 3: Top 10 Model Throughput (TPS)')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.35) 
    plt.savefig('outputs/static_plots/3_bar.png')
    plt.close()

    # FIG 4: BOX (Latency Stability Profiles)
    plt.figure(figsize=(12, 7))
    sns.boxplot(data=filtered_df, x='Model', y='Latency', palette='Set3')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.title('Figure 4: Latency Stability Comparison (Gemini Family)')
    plt.xlabel('Model Version')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3) 
    plt.savefig('outputs/static_plots/4_box.png')
    plt.close()

    # FIG 5: HISTOGRAM (Density Distribution)
    plt.figure(figsize=(10, 6))
    sns.histplot(filtered_df['TPS'], kde=True, color='teal')
    plt.title('Figure 5: Performance Density Distribution')
    plt.savefig('outputs/static_plots/5_hist.png')
    plt.close()

    # FIG 6: HEATMAP (Inter-variable Correlation Matrix)
    plt.figure(figsize=(8, 6))
    sns.heatmap(filtered_df[['Latency', 'TPS', 'Sequence_Step']].corr(), annot=True, cmap='Blues')
    plt.title('Figure 6: Metric Correlation Matrix')
    plt.savefig('outputs/static_plots/6_heatmap.png')
    plt.close()

# =====================================================================
# PIPELINE EXECUTION LIFECYCLE CONTROLLER
# =====================================================================
if __name__ == "__main__":
    # Ingest baseline log telemetry
    df_raw = ingest_data('ai_models_performance.csv')
    df_clean = clean_data(df_raw)
    
    # Isolate project specific target group slice (Gemini architectures)
    gemini_subset = df_clean[df_clean['Model'].str.contains('Gemini', case=False)].copy()
    
    if not gemini_subset.empty:
        # Save unique data filter outputs to repository asset pool
        os.makedirs('data', exist_ok=True)
        df_clean.to_csv('data/ai_models_performance_cleaned.csv', index=False)
        
        # Populate global states safely before invoking dynamic render cycles
        _global_filtered_df = gemini_subset.copy()
        
        # Execute processing functions sequentially
        compute_engineering_statistics(gemini_subset)
        trained_regression = run_decay_model(gemini_subset)
        generate_static_plots(df_clean, gemini_subset, trained_regression)
        
        print("🎬 Rendering dynamic visualization configurations via Function 5...")
        
        # Build Animation 1: Dynamic Scanning Profile Trace
        fig_scan, ax_scan = plt.subplots(figsize=(10, 6))
        ani1 = animation.FuncAnimation(
            fig_scan, update_animation_frames, frames=11, interval=300,
            fargs=(ax_scan, "scan")
        )
        ani1.save('outputs/animations/1_performance_mapping.gif', writer='pillow')
        plt.close(fig_scan)
        
        # Build Animation 2: Latency Trace Pulse Real-time Execution Simulation
        fig_pulse, ax_pulse = plt.subplots(figsize=(10, 6))
        _global_point_handle, = ax_pulse.plot([], [], 'ro', markersize=12)
        ax_pulse.set_xlim(gemini_subset['Sequence_Step'].min(), gemini_subset['Sequence_Step'].max())
        ax_pulse.set_ylim(gemini_subset['TPS'].min() * 0.9, gemini_subset['TPS'].max() * 1.1)
        ax_pulse.set_title("Real-Time Throughput Pulse Simulation")
        ax_pulse.set_xlabel("Tokens Generated")
        ax_pulse.set_ylabel("Throughput (TPS)")
        
        ani2 = animation.FuncAnimation(
            fig_pulse, update_animation_frames, frames=len(gemini_subset), interval=200,
            fargs=(ax_pulse, "pulse")
        )
        ani2.save('outputs/animations/2_latency_pulse.gif', writer='pillow')
        plt.close(fig_pulse)
        
        print("🚀 PIPELINE PROCESS COMPLETE: 5 Main Functions successfully executed.")
    else:
        print("❌ Pipeline Failure: Empty subset returned based on target execution criteria.")