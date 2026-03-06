"""
V3.0 Consensus Engine — Dual-Hierarchy Cardiac Diagnosis System
================================================================
Runs Engine A (Visual) and Engine B (Signal) hierarchies in parallel.
Compares top-1 diagnoses for consensus:
  - Agreement    → "High Consensus Diagnosis" badge
  - Disagreement → "Manual Review" flag
"""

import os
import sys

# Setup Path
current_dir = os.path.dirname(os.path.abspath(__file__))
production_dir = os.path.dirname(current_dir)
sys.path.append(production_dir)

from utils.backend import CardiacPredictor
from utils.visual_backend import VisualPredictor


class CardiacSystem:
    """
    Unified Dual-Engine Cardiac Diagnosis System.
    
    Orchestrates:
      - Engine B (Signal): Hierarchical signal classifier (Router → Specialist)
      - Engine A (Visual): Hierarchical visual classifier (Router → Specialist)
    
    Consensus Logic:
      - If both engines agree on the top-1 diagnosis → "High Consensus Diagnosis"
      - If engines disagree → "Manual Review"
    """
    
    def __init__(self, models_dir="models", data_dir="data"):
        """
        Initialize both prediction engines.
        
        Args:
            models_dir: Relative path to models directory from production root.
            data_dir: Relative path to data directory from production root.
        """
        print("=" * 60)
        print("Initializing CardiacSystem V3.0 — Dual Hierarchy")
        print("=" * 60)
        
        # Engine B: Signal Hierarchy
        try:
            self.signal_engine = CardiacPredictor(models_dir=models_dir)
            self.signal_available = True
            print("✅ Engine B (Signal) — Online")
        except Exception as e:
            self.signal_engine = None
            self.signal_available = False
            print(f"⚠️  Engine B (Signal) — Offline: {e}")
        
        # Engine A: Visual Hierarchy
        try:
            self.visual_engine = VisualPredictor(models_dir=models_dir, data_dir=data_dir)
            self.visual_available = True
            print("✅ Engine A (Visual) — Online")
        except Exception as e:
            self.visual_engine = None
            self.visual_available = False
            print(f"⚠️  Engine A (Visual) — Offline: {e}")
        
        print("=" * 60)
    
    def predict_signal(self, signal_numpy):
        """Run Engine B (Signal) hierarchy only."""
        if not self.signal_available:
            return {"error": "Engine B (Signal) not available"}
        return self.signal_engine.predict(signal_numpy)
    
    def predict_visual(self, image_input, patient_metadata=None):
        """Run Engine A (Visual) hierarchy only."""
        if not self.visual_available:
            return {"error": "Engine A (Visual) not available"}
        return self.visual_engine.predict(image_input, patient_metadata=patient_metadata)
    
    def predict_consensus(self, signal_numpy=None, image_input=None, patient_metadata=None):
        """
        Run both engines in parallel and compute consensus.
        
        Args:
            signal_numpy: Raw ECG signal array [12, 5000] or [5000, 12].
            image_input: PIL Image or path to ECG image.
            patient_metadata: Optional dict with 'name', 'age', 'gender'.
            
        Returns:
            dict with:
              - signal_result: Engine B prediction dict
              - visual_result: Engine A prediction dict
              - consensus: "High Consensus Diagnosis" or "Manual Review"
              - consensus_diagnosis: Agreed diagnosis (if consensus) or None
              - signal_diagnosis: str
              - visual_diagnosis: str
        """
        signal_result = None
        visual_result = None
        
        # Run Engine B (Signal)
        if signal_numpy is not None and self.signal_available:
            try:
                signal_result = self.signal_engine.predict(signal_numpy)
            except Exception as e:
                signal_result = {"error": str(e)}
        
        # Run Engine A (Visual)
        if image_input is not None and self.visual_available:
            try:
                visual_result = self.visual_engine.predict(image_input, patient_metadata=patient_metadata)
            except Exception as e:
                visual_result = {"error": str(e)}
        
        # Extract diagnoses
        signal_diag = signal_result.get("diagnosis") if signal_result and "error" not in signal_result else None
        visual_diag = visual_result.get("diagnosis") if visual_result and "error" not in visual_result else None
        
        # Consensus Logic
        if signal_diag and visual_diag:
            if signal_diag == visual_diag:
                consensus = "High Consensus Diagnosis"
                consensus_diagnosis = signal_diag
            else:
                consensus = "Manual Review"
                consensus_diagnosis = None
        elif signal_diag:
            consensus = "Single Engine (Signal Only)"
            consensus_diagnosis = signal_diag
        elif visual_diag:
            consensus = "Single Engine (Visual Only)"
            consensus_diagnosis = visual_diag
        else:
            consensus = "No Diagnosis Available"
            consensus_diagnosis = None
        
        return {
            "signal_result": signal_result,
            "visual_result": visual_result,
            "consensus": consensus,
            "consensus_diagnosis": consensus_diagnosis,
            "signal_diagnosis": signal_diag,
            "visual_diagnosis": visual_diag,
        }
