import traceback
try:
    from src.explainability.gradcam import XAI_Visualizer
    print("OK")
except Exception as e:
    with open("error.txt", "w") as f:
        traceback.print_exc(file=f)
