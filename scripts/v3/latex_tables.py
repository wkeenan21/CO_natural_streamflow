import pandas as pd
import numpy as np

# Your input dictionary
USE_DICT = {
    'irrigation': -0.6, 
    'municipal': -0.3, 
    'interbasin': -1,
    'industrial': -1, 
    'hydropower': 0, 
    'intrabasin': -1, 
    'transbasin': 1,
}

# 1. Convert the dictionary into a Pandas DataFrame
df = pd.DataFrame(list(USE_DICT.items()), columns=['Water Use', 'Consumptive Use Factor'])

# 2. Generate the LaTeX code (using booktabs for professional formatting)
latex_code = df.to_latex(
    index=False, 
    caption="Consumptive Use Factors by Water Use Type", 
    label="tab:use_factors",
)

print(latex_code)