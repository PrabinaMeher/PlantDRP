__version__     = "1.0.0"
__author__      = "Meher PK, Pradhan UK, Gupta A, Kumar S, Kumari A, Das R"
__affiliation__ = "ICAR-IASRI, New Delhi, India"
__email__       = "meherprabin@yahoo.com"
__all__         = ["Predictor"]

def __getattr__(name):
    if name == "Predictor":
        from .predictor import Predictor
        return Predictor
    raise AttributeError(f"module 'plantdrp' has no attribute {name!r}")
