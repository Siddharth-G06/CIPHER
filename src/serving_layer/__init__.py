# CIPHER — src/serving_layer package
"""
serving_layer
-------------
Serving-layer components for the CIPHER fraud-detection pipeline.

Public exports
--------------
FlaggedTransaction   : Dataclass carrying every artefact produced for a
                       single flagged transaction (scores, features, SHAP
                       explanation, drift state, model version).
ReportGenerator      : Template-Method base class that generates professional
                       PDF investigation reports via ReportLab Platypus.
"""

from src.serving_layer.report_generator import FlaggedTransaction, ReportGenerator

__all__ = [
    "FlaggedTransaction",
    "ReportGenerator",
]
