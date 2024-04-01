import cv2
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import webbrowser

# Load the CSV data
data = pd.read_csv(r'C:\Users\dbsha\OneDrive\Desktop\gmit\intern\python-project-color-detection\colors.csv', header=None, names=['Color', 'Label', 'Hex', 'R', 'G', 'B'])

# Preprocess the data
X = data[['R', 'G', 'B']]
y = data['Label']

# Train a KNN model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X, y)

# Initialize variables for detected colors
detected_colors = []

# Function to classify color on mouse click
def classify_color(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel_color = frame[y, x]
        detect_color_for_pixel(pixel_color)

# Function to detect color for a specific pixel
def detect_color_for_pixel(pixel_color):
    # Predict color label using the trained KNN model
    color_label = model.predict([pixel_color])[0]
    print(f"Detected color: {color_label.capitalize()}")
    color_rgb = f"R={pixel_color[2]} G={pixel_color[1]} B={pixel_color[0]}"
    detected_colors.append((color_label, color_rgb))
    update_color_window()

# Function to update the color window with detected colors
def update_color_window():
    color_window = np.zeros((500, 600, 3), np.uint8)
    button_height = 30
    for i, (color_name, color_rgb) in enumerate(detected_colors[-10:]):  # Display last 10 detected colors
        cv2.putText(color_window, f"{color_name}: {color_rgb}", (20, 50 + 50 * i), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.rectangle(color_window, (20, 50 + 50 * i), (150, 50 + 50 * i + button_height), (0, 0, 255), -1)
        cv2.putText(color_window, "Search", (40, 80 + 50 * i), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('Detected Color', color_window)

# Function to search for color on website

def search_color(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Determine which button is clicked
        index = (y - 50) // 50
        color_label = detected_colors[-index][0]
        search_color_on_website(color_label)

def search_color_on_website(color_label):
    search_url = f"https://rgbcolorcode.com/color/{color_label.lower().replace(' ', '-')}"
    webbrowser.open_new_tab(search_url)
# Add event handling for the search button


# Open default camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # Read frame from camera
    if not ret:
        print("Error: Failed to capture frame")
        break

    # Display the resulting frame
    cv2.imshow('Color Detection', frame)
    
    # Set mouse callback function for color detection
    cv2.setMouseCallback('Color Detection', classify_color)
    cv2.namedWindow('Detected Color', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Detected Color', search_color)
    
    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
