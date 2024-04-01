import tkinter as tk
import subprocess

# Function to open Python file 1
def open_file1():
    subprocess.Popen(["python", "ml.py"])

# Function to open Python file 2
def open_file2():
    subprocess.Popen(["python", "eml.py"])

# Create the main window
window = tk.Tk()
window.title("Color Detection")

# Add a label for color detection heading
heading_label = tk.Label(window, text="Color Detection", font=("Arial", 14))
heading_label.pack(pady=10)  # Add some padding above the heading label

# Create Button 1 to load image
button1 = tk.Button(window, text="Load Image", command=open_file1)
button1.pack(pady=5)  # Add some padding between the heading label and the button

# Create Button 2 to open camera
button2 = tk.Button(window, text="Open Camera", command=open_file2)
button2.pack(pady=5)  # Add some padding between the buttons

# Run the main event loop
window.mainloop()
