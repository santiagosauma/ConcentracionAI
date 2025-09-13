# Dashboard Pages Module
# Cada archivo contiene una página específica del dashboard

from .exploration import render_exploration_page
from .prediction import render_prediction_page  
from .model_analysis import render_model_analysis_page
from .whatif import render_whatif_page

__all__ = [
    'render_exploration_page',
    'render_prediction_page', 
    'render_model_analysis_page',
    'render_whatif_page'
]
