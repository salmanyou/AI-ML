import matplotlib.pyplot as plt
import pandas as pd

# Data for the confusion matrix
data = {
    "Class": ["Print Text", "Book Text", "Digital Screen", "Posters"],
    "Print Text": [140, 0, 0, 0],
    "Book Text": [0, 133, 0, 0],
    "Digital Screen": [0, 0, 138, 0],
    "Posters": [0, 0, 0, 127]
}

# Create a DataFrame for better visualization
df = pd.DataFrame(data)

# Plotting the table
fig, ax = plt.subplots(figsize=(8, 4))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')

# Styling the table
table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width(col=list(range(len(df.columns))))

# Save the figure
plt.savefig(r"C:\Users\Elite\Downloads\Snip Speak\OCR Model\Chars-74k dataset\confusion_matrix_table.png")
plt.show()
