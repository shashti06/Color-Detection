import pandas as pd
import cv2
import numpy as np
import webbrowser

class MyKNeighborsClassifier:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for x in X_test:
            distances = []
            for i, x_train in enumerate(self.X_train):
                distance = np.sqrt(np.sum((x - x_train) ** 2))
                distances.append((distance, self.y_train[i]))
            distances.sort()
            neighbors = distances[:self.n_neighbors]
            counts = {}
            for _, label in neighbors:
                counts[label] = counts.get(label, 0) + 1
            predicted_label = max(counts, key=counts.get)
            predictions.append(predicted_label)
        return predictions

# Step 1: Load the CSV data
data = pd.read_csv(r'C:\Users\dbsha\OneDrive\Desktop\gmit\intern\python-project-color-detection\colors.csv', header=None, names=['Color', 'Label', 'Hex', 'R', 'G', 'B'])

# Step 2: Preprocess the data
X = data[['R', 'G', 'B']]
y = data['Label']

# Step 3: Train a machine learning model (using custom KNN)
model = MyKNeighborsClassifier(n_neighbors=5)
model.fit(X.values, y.values)

# Step 4: Load and process the image
def classify_color(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = img[y, x]
        B, G, R = pixel[0], pixel[1], pixel[2]
        color_name = model.predict([[R, G, B]])
        color_rgb = f"R={R} G={G} B={B}"
        detected_colors.append((color_name[0], color_rgb))
        update_color_window()

def update_color_window():
    color_window[:] = (0, 0, 0)
    button_height = 30
    for i, (color_name, color_rgb) in enumerate(detected_colors[-10:]):  # Display last 10 detected colors
        cv2.putText(color_window, f"{color_name}: {color_rgb}", (20, 50 + 50 * i), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.rectangle(color_window, (20, 50 + 50 * i), (150, 50 + 50 * i + button_height), (0, 0, 255), -1)
        cv2.putText(color_window, "Search", (40, 80 + 50 * i), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

def search_color(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Determine which button is clicked
        index = (y - 50) // 50
        color_label = detected_colors[-index][0]
        search_color_on_website(color_label)

def search_color_on_website(color_label):
    search_url = f"https://rgbcolorcode.com/color/{color_label.lower().replace(' ', '-')}"
    webbrowser.open_new_tab(search_url)

# Get image location from user input
img_path = input("Enter the path of the image: ")
img = cv2.imread(img_path)
if img is None:
    print("Error: Could not read the image")
    exit()

color_window = np.zeros((500, 600, 3), np.uint8)
detected_colors = []

cv2.namedWindow('image')
cv2.namedWindow('Detected Color', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('image', classify_color)
cv2.setMouseCallback('Detected Color', search_color)  # Add event handling for the search button

# Step 5: Predict the color on mouse click
while True:
    cv2.imshow('image', img)
    cv2.imshow('Detected Color', color_window)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
