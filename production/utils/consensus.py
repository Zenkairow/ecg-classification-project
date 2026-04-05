"""
V3.5 Consensus Engine — Dual-Hierarchy Cardiac Diagnosis System
================================================================
Runs Engine A (Visual) and Engine B (Signal) hierarchies in parallel.

Supports two consensus modes:
  1. Deterministic AGC (Original): Compares top-1 diagnoses.
  2. Conformal AGC (v3.5): Compares Prediction Sets via set intersection,
     producing a 4-level confidence hierarchy:
       - High Consensus:   top-1 diagnoses match across both engines
       - Soft Consensus:   top-1 of one engine exists in the other's prediction set
       - Partial Overlap:  prediction sets share at least one class (but not top-1)
       - Manual Review:    zero intersection between prediction sets
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
    Unified Dual-Engine Cardiac Diagnosis System (v3.5-experimental).
    
    Orchestrates:
      - Engine B (Signal): Hierarchical signal classifier (Router → Specialist)
      - Engine A (Visual): Hierarchical visual classifier (Router → Specialist)
    
    Consensus Modes:
      - Standard:     top-1 vs top-1 (v3.0 behavior)
      - Uncertainty:  prediction set intersection with MC-Dropout (v3.5)
    """
    
    def __init__(self, models_dir="models", data_dir="data"):
        """
        Initialize both prediction engines.
        
        Args:
            models_dir: Relative path to models directory from production root.
            data_dir: Relative path to data directory from production root.
        """
        print("=" * 60)
        print("Initializing CardiacSystem V3.5 — Dual Hierarchy + Conformal")
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
    
    def predict_signal(self, signal_numpy, uncertainty_mode=False, mc_samples=50, alpha=0.1):
        """Run Engine B (Signal) hierarchy only."""
        if not self.signal_available:
            return {"error": "Engine B (Signal) not available"}
        return self.signal_engine.predict(
            signal_numpy,
            uncertainty_mode=uncertainty_mode,
            mc_samples=mc_samples,
            alpha=alpha,
        )
    
    def predict_visual(self, image_input, patient_metadata=None, uncertainty_mode=False, mc_samples=50, alpha=0.1):
        """Run Engine A (Visual) hierarchy only."""
        if not self.visual_available:
            return {"error": "Engine A (Visual) not available"}
        return self.visual_engine.predict(
            image_input,
            patient_metadata=patient_metadata,
            uncertainty_mode=uncertainty_mode,
            mc_samples=mc_samples,
            alpha=alpha,
        )
    
    # ── Deterministic AGC (Original v3.0) ─────────────────────────
    def _compute_deterministic_consensus(self, signal_diag, visual_diag):
        """
        Original top-1 vs top-1 agreement logic.
        Returns (consensus_label, consensus_diagnosis).
        """
        if signal_diag and visual_diag:
            if signal_diag == visual_diag:
                return "High Consensus Diagnosis", signal_diag
            else:
                return "Manual Review", None
        elif signal_diag:
            return "Single Engine (Signal Only)", signal_diag
        elif visual_diag:
            return "Single Engine (Visual Only)", visual_diag
        else:
            return "No Diagnosis Available", None

    # ── Conformal AGC (New v3.5) ──────────────────────────────────
    def _compute_conformal_consensus(self, signal_result, visual_result):
        """
        Prediction Set-based consensus using MC-Dropout conformal outputs.
        
        Consensus Hierarchy:
          1. HIGH CONSENSUS:     Top-1 diagnoses match
          2. SOFT CONSENSUS:     Top-1 of Engine B is in Engine A's prediction set, or vice versa
          3. PARTIAL OVERLAP:    Prediction sets share >= 1 class, but not the other engine's top-1
          4. MANUAL REVIEW:      Zero intersection between prediction sets
          
        Also reports:
          - set_intersection:    Classes found in both prediction sets
          - combined_uncertainty: Mean of both engines' epistemic uncertainty
        """
        signal_diag = signal_result.get("diagnosis")
        visual_diag = visual_result.get("diagnosis")
        
        signal_set = set(signal_result.get("prediction_set_labels", []))
        visual_set = set(visual_result.get("prediction_set_labels", []))
        
        # Set intersection
        intersection = signal_set & visual_set
        
        # Combined epistemic uncertainty
        signal_eu = signal_result.get("epistemic_uncertainty", 0.0)
        visual_eu = visual_result.get("epistemic_uncertainty", 0.0)
        combined_uncertainty = (signal_eu + visual_eu) / 2.0
        
        # Determine consensus level
        if signal_diag and visual_diag and signal_diag == visual_diag:
            # Level 1: Both engines' top-1 predictions match exactly
            consensus_level = "High Consensus"
            consensus_diagnosis = signal_diag
            
        elif signal_diag in visual_set or visual_diag in signal_set:
            # Level 2: One engine's top-1 is in the other's prediction set
            # Choose the diagnosis that appears in both contexts
            if signal_diag in visual_set:
                consensus_diagnosis = signal_diag
            else:
                consensus_diagnosis = visual_diag
            consensus_level = "Soft Consensus"
            
        elif len(intersection) > 0:
            # Level 3: Sets overlap, but neither top-1 is in the other set
            # Select the class with highest combined mean probability
            best_class = None
            best_prob = -1.0
            signal_probs = signal_result.get("mean_probs", {})
            visual_probs = visual_result.get("mean_probs", {})
            
            for cls in intersection:
                combined_p = signal_probs.get(cls, 0) + visual_probs.get(cls, 0)
                if combined_p > best_prob:
                    best_prob = combined_p
                    best_class = cls
            
            consensus_level = "Partial Overlap"
            consensus_diagnosis = best_class
            
        else:
            # Level 4: Zero intersection — complete disagreement
            consensus_level = "Manual Review"
            consensus_diagnosis = None
        
        return {
            "consensus_level": consensus_level,
            "consensus_diagnosis": consensus_diagnosis,
            "set_intersection": sorted(list(intersection)),
            "signal_prediction_set": sorted(list(signal_set)),
            "visual_prediction_set": sorted(list(visual_set)),
            "combined_epistemic_uncertainty": combined_uncertainty,
        }
    
    def predict_consensus(self, signal_numpy=None, image_input=None,
                          patient_metadata=None, uncertainty_mode=False,
                          mc_samples=50, alpha=0.1):
        """
        Run both engines and compute consensus.
        
        Args:
            signal_numpy: Raw ECG signal array [12, 5000] or [5000, 12].
            image_input: PIL Image or path to ECG image.
            patient_metadata: Optional dict with 'name', 'age', 'gender'.
            uncertainty_mode: If True, use MC-Dropout Conformal Prediction AGC.
            mc_samples: Number of stochastic forward passes (T=50).
            alpha: Significance level for prediction sets (default: 0.1).
            
        Returns:
            dict with signal_result, visual_result, consensus info.
            In uncertainty_mode, also includes conformal_consensus with
            set intersection analysis and epistemic uncertainty metrics.
        """
        signal_result = None
        visual_result = None
        
        # Run Engine B (Signal)
        if signal_numpy is not None and self.signal_available:
            try:
                signal_result = self.signal_engine.predict(
                    signal_numpy,
                    uncertainty_mode=uncertainty_mode,
                    mc_samples=mc_samples,
                    alpha=alpha,
                )
            except Exception as e:
                signal_result = {"error": str(e)}
        
        # Run Engine A (Visual)
        if image_input is not None and self.visual_available:
            try:
                visual_result = self.visual_engine.predict(
                    image_input,
                    patient_metadata=patient_metadata,
                    uncertainty_mode=uncertainty_mode,
                    mc_samples=mc_samples,
                    alpha=alpha,
                )
            except Exception as e:
                visual_result = {"error": str(e)}
        
        # Extract top-1 diagnoses
        signal_diag = signal_result.get("diagnosis") if signal_result and "error" not in signal_result else None
        visual_diag = visual_result.get("diagnosis") if visual_result and "error" not in visual_result else None
        
        # --- Deterministic AGC (always computed for backward compat) ---
        consensus, consensus_diagnosis = self._compute_deterministic_consensus(signal_diag, visual_diag)
        
        result = {
            "signal_result": signal_result,
            "visual_result": visual_result,
            "consensus": consensus,
            "consensus_diagnosis": consensus_diagnosis,
            "signal_diagnosis": signal_diag,
            "visual_diagnosis": visual_diag,
        }
        
        # --- Conformal AGC (v3.5 extension) ---
        if uncertainty_mode and signal_result and visual_result:
            has_signal_sets = "prediction_set_labels" in (signal_result or {})
            has_visual_sets = "prediction_set_labels" in (visual_result or {})
            
            if has_signal_sets and has_visual_sets:
                conformal = self._compute_conformal_consensus(signal_result, visual_result)
                result["conformal_consensus"] = conformal
                
                # Override top-level consensus with conformal result
                result["consensus"] = conformal["consensus_level"]
                result["consensus_diagnosis"] = conformal["consensus_diagnosis"]
        
        return result
