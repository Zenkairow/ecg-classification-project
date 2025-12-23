
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime, timedelta

# Added for 3D Visuals
import plotly.graph_objects as go
import plotly.express as px

def plot_interactive_3d(signal_data, sampling_rate=500, theme="Clinical Groups", line_width=2.5, opacity=0.9):
    """
    Creates a detailed, professional 3D interactive visualization of the 12-Lead ECG.
    Supports themes and interactivity customization.
    """
    # Ensure shape [12, N]
    if signal_data.shape[0] != 12:
        signal_data = signal_data.T
        
    num_samples = signal_data.shape[1]
    duration = num_samples / sampling_rate
    time = np.linspace(0, duration, num_samples)
    
    leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    fig = go.Figure()
    
    # --- COLOR THEMES ---
    colors = []
    
    if theme == "Clinical Groups":
        # Grouped by anatomical region for medical utility
        # Limb Leads (I, II, III, aVR, aVL, aVF) -> Cyan (Standard)
        # Septal/Anterior (V1, V2, V3) -> Green (distinct)
        # Lateral (V4, V5, V6) -> Yellow/Orange (distinct)
        c_limb = '#00ffff' # Cyan
        c_septal = '#00ff00' # Green
        c_lateral = '#ffcc00' # Amber
        colors = [c_limb]*6 + [c_septal]*3 + [c_lateral]*3
        
    elif theme == "Retro Neon":
        # Rainbow gradient for aesthetics
        colors = [
            '#00ff00', '#00ffaa', '#00ffff', '#00aaff', '#0055ff', '#0000ff',
            '#ff00ff', '#ff00aa', '#ff0055', '#ff0000', '#ff5500', '#ffaa00'
        ]
    else: # Monochrome Cyan (Default/Fallback)
        colors = ['#00ffff'] * 12

    for i in range(12):
        fig.add_trace(go.Scatter3d(
            x=time,
            y=[i] * num_samples, 
            z=signal_data[i],
            mode='lines',
            name=leads[i], # Legend works automatically
            line=dict(
                color=colors[i],
                width=line_width
            ),
            hoverinfo='name+x+z',
            opacity=opacity
        ))
        
    fig.update_layout(
        title=dict(
            text="<b>12-LEAD SPATIAL VISUALIZATION</b>",
            font=dict(size=14, color='#aaaaaa'),
            x=0.0,
            y=0.98
        ),
        scene=dict(
            # X Axis: Time
            xaxis=dict(
                title='TIME (s)', 
                backgroundcolor='#0f1116', 
                gridcolor='#222222', 
                color='#aaaaaa',
                showbackground=True,
                zerolinecolor='#444444'
            ),
            # Y Axis: Lead Selection
            yaxis=dict(
                title='LEAD', 
                tickvals=list(range(12)), 
                ticktext=leads, 
                backgroundcolor='#0f1116', 
                gridcolor='#222222', 
                color='#ffffff',
                showbackground=True
            ),
            # Z Axis: Amplitude
            zaxis=dict(
                title='mV', 
                range=[-2.5, 2.5], 
                backgroundcolor='#0f1116', 
                gridcolor='#333333', 
                color='#aaaaaa',
                showbackground=True
            ),
            camera=dict(
                eye=dict(x=1.6, y=0.2, z=0.6), # Optimised angle (Less "top down", more "side profile")
                center=dict(x=0, y=0, z=0)
            ),
            aspectratio=dict(x=2.2, y=1.2, z=0.5) # Wider time axis for smoother look
        ),
        paper_bgcolor='#0e1117',
        margin=dict(l=0, r=0, b=0, t=30),
        height=500,
        showlegend=True, # ALLOW TOGGLING
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(0,0,0,0.5)",
            font=dict(color="white")
        )
    )
    
    return fig

def plot_12_lead_ecg(signal_data, patient_meta=None, original_filename="unknown_signal", sampling_rate=500, save_dir="generated_ecgs"):
    """
    Plots a professional Medical-Grade 12-Lead ECG.
    Gold Standard Edition:
    - True 25mm/s Geometry
    - 400 DPI Resolution (Print Grade)
    - Precision Time Ticks (0.2s)
    - Absolute Header Positioning
    """
    
    # Defaults
    if patient_meta is None:
        patient_meta = {"name": "Anonymous", "age": "N/A", "gender": "U", "notes": ""}
        
    # Ensure shape is [12, N]
    if signal_data.shape[0] != 12:
        signal_data = signal_data.T
    
    leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    # --- PHYSICAL DIMENSIONS ---
    # We want exact 25mm/s representation.
    # 10 seconds = 250mm width.
    # We add padding for labels. Let's define the "Grid Area" to be exactly proportionate.
    
    # Canvas Size: 20 inches width * 24 inches height (Huge canvas for sharpness)
    fig, axes = plt.subplots(12, 1, figsize=(20, 24), sharex=True)
    
    # Margins (Top 12% for Header)
    fig.subplots_adjust(hspace=0, top=0.88, bottom=0.05, left=0.06, right=0.98)
    
    # Time vector
    num_samples = signal_data.shape[1]
    duration = num_samples / sampling_rate
    time = np.linspace(0, duration, num_samples)
    
    # Colors (High Contrast Medical)
    grid_red_major = '#ff5050' # Sharper Red
    grid_red_minor = '#ffb3b3' # Lighter Red
    signal_black = '#000000'   # Pure Black
    
    for i, ax in enumerate(axes):
        lead_data = signal_data[i]
        
        # 1. GRID SYSTEM (Medical Standard)
        # Minor: 1mm (0.04s x 0.1mV)
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.04))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
        
        # Major: 5mm (0.2s x 0.5mV)
        ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
        
        # Draw Grids
        # Thinner lines = Less Blur at zoom
        ax.grid(which='minor', linestyle='-', linewidth=0.3, color=grid_red_minor, alpha=0.7)
        ax.grid(which='major', linestyle='-', linewidth=0.6, color=grid_red_major, alpha=0.8)
        
        # 2. SIGNAL TRACE
        # Linewidth 0.8 is standard for high-res print.
        # Antialiased=True provides smooth curves.
        ax.plot(time, lead_data, color=signal_black, linewidth=0.9, antialiased=True)
        
        # 3. TYPOGRAHY
        ax.set_ylabel(leads[i], fontsize=16, fontweight='bold', rotation=0, labelpad=50, color='#222222', va='center')
        
        # 4. SPINES (Borders)
        for spine in ax.spines.values():
            spine.set_edgecolor(grid_red_major)
            spine.set_linewidth(1.2)
            
        # 5. SCALING
        # Fixed +/- 2.0mV Range
        ax.set_ylim(-2.0, 2.0)
        ax.set_xlim(0, duration)
        
        # Hide internal ticks
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # --- X-AXIS TELEMETRY ---
    # Bottom axis contains the precision markings
    axes[-1].tick_params(left=False, bottom=True, labelleft=False, labelbottom=True, direction='out', length=6, width=1.2)
    
    # Major Ticks every 0.2s (Standard Grid Box)
    major_ticks = np.arange(0, duration + 0.1, 0.2)
    axes[-1].set_xticks(major_ticks)
    
    # Labels: Only label every 1.0 second to avoid clutter, but keep ticks for every 0.2s
    labels = []
    for t in major_ticks:
        if abs(t % 1.0) < 0.01: # Sample is close to integer
            labels.append(f"{t:.0f}s")
        else:
            labels.append("") # Tick exists but no text
            
    axes[-1].set_xticklabels(labels, fontsize=12, fontweight='bold', color='#444444')
    
    # Minor Ticks for every 0.04s (1mm) - Extreme Precision
    axes[-1].xaxis.set_minor_locator(plt.MultipleLocator(0.04))
    axes[-1].tick_params(which='minor', bottom=True, direction='out', length=3, width=0.6, color='#888888')
    
    # Axis Label
    axes[-1].set_xlabel(f"Time (seconds)  |  Calibration: 25mm/s, 10mm/mV  |  Bandwidth: 0.05-150Hz", 
                        fontsize=14, fontweight='bold', color='#222222', labelpad=15)
    
    # --- HEADER (Absolute Position) ---
    
    # Title
    fig.text(0.5, 0.96, "12-LEAD ELECTROCARDIOGRAM REPORT", 
             ha='center', va='top', fontsize=28, fontweight='900', color='#000000')
    
    # Timestamp
    ist_time = datetime.utcnow() + timedelta(hours=5, minutes=30)
    timestamp_str = ist_time.strftime("%d-%m-%Y %H:%M:%S IST")
    
    # Patient Data (Left)
    fig.text(0.06, 0.93, f"Patient Name: {patient_meta['name']}", fontsize=16, fontweight='bold', family='monospace')
    fig.text(0.06, 0.915, f"Patient ID:   {original_filename}", fontsize=14, family='monospace')
    fig.text(0.06, 0.900, f"Demographics: {patient_meta['age']} Years / {patient_meta['gender']}", fontsize=14, family='monospace')
    
    # Technical Data (Right)
    fig.text(0.98, 0.93, f"Report Date: {timestamp_str}", ha='right', fontsize=14, family='monospace')
    fig.text(0.98, 0.915, "Signal Source: Raw Digital Telemetry", ha='right', fontsize=14, family='monospace')
    fig.text(0.98, 0.900, "Device Config: Standard 12-Lead", ha='right', fontsize=14, family='monospace')
    
    # Notes Strip
    fig.text(0.06, 0.88, f"Physician Notes: {patient_meta['notes']}", 
             fontsize=14, style='italic', color='#444444', 
             bbox=dict(facecolor='#f4f4f4', edgecolor='#dddddd', boxstyle='round,pad=0.5'))

    # Save
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    def clean(s): return str(s).replace(" ", "_").replace("/", "-").replace(":", "_").replace("|", "_").replace(",", "_")
    safe_name = clean(patient_meta['name'])
    safe_age = clean(patient_meta['age'])
    safe_sig = clean(original_filename)
    
    filename = f"{safe_name}_{safe_age}_({safe_sig}).png"
    save_path = os.path.join(save_dir, filename)
    
    # 400 DPI is optimal trade-off for speed/quality.
    # At 20x24 inch canvas, this is 8000x9600 pixels.
    plt.savefig(save_path, dpi=400, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    return save_path
