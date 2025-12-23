import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from utils.plotting import plot_12_lead_ecg
    import inspect
    print("--- DEBUG INFO ---")
    print(f"File: {inspect.getfile(plot_12_lead_ecg)}")
    print(f"Signature: {inspect.signature(plot_12_lead_ecg)}")
    print("------------------")
except Exception as e:
    print(e)
