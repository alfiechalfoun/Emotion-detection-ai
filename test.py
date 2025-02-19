from graphviz import Digraph

# Create a Directed Graph for Critical Path Diagram
dot = Digraph(comment="Critical Path Diagram for Emotion Detection AI")

# Add Nodes
dot.node("Start", "Start")
dot.node("Capture", "Capture Frame")
dot.node("Detect", "Detect Face")
dot.node("Crop", "Crop Face")
dot.node("Resize", "Resize Image\n(48x48 MaxPooling)")
dot.node("Predict", "Predict Emotion\n(CNN Model)")
dot.node("Update", "Update Bounding Box Color")
dot.node("Display", "Display Frame")
dot.node("End", "End")

# Add Edges for Task Flow
dot.edge("Start", "Capture", label="Start Capture")
dot.edge("Capture", "Detect", label="Capture Frame")
dot.edge("Detect", "Crop", label="Detect Face")
dot.edge("Crop", "Resize", label="Crop Face")
dot.edge("Resize", "Predict", label="Resize Image")
dot.edge("Predict", "Update", label="Predict Emotion")
dot.edge("Update", "Display", label="Update Box Color")
dot.edge("Display", "End", label="Display Frame")

# Render and Save Diagram
dot.render('critical_path_diagram', format='png', cleanup=False)
print("Critical Path Diagram saved as 'critical_path_diagram.png'.")
