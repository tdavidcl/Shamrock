try:
    from .pyshamrock import *
except ImportError:
    # the one above will fail if pyshamrock is declared by the embed python interpreter
    # so we try to import it from the global namespace
    from pyshamrock import *
        
print(f"pyshamrock imported from {__file__}")