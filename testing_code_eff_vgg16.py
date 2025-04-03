from tensorflow.keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.preprocessing import image
from joblib import dump, load
import pandas as pd
import numpy as np

model = load_model('hybrid_vgg16_efficientnet_model.h5')

test_image="1.jpg"
test_image = image.load_img(test_image, target_size=(224, 224))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = np.argmax(model.predict(test_image))
result = result.item()
print(result)

# Load the Excel file
file_path = "pesticides.xlsx"  # Ensure this is the correct path
df = pd.read_excel(file_path, sheet_name="Sheet1")

# Ensure column names are correct (strip spaces & convert to lowercase)
df.rename(columns=lambda x: x.strip().lower(), inplace=True)

# Ensure "result" column is an integer
df["result"] = df["result"].astype(int)

result = int(result)  # Convert to integer for matching

# Disease mapping
disease_mapping = {
    0: "Tomato - Bacterial spot",
    1: "Tomato - Early blight",
    2: "Tomato - Healthy",
    3: "Tomato - Late blight",
    4: "Tomato - Leaf Mold",
    5: "Tomato - Septoria leaf spot",
    6: "Tomato - Spider mites (Two-spotted spider mite)",
    7: "Tomato - Target Spot",
    8: "Tomato - Tomato mosaic virus",
    9: "Tomato - Tomato Yellow Leaf Curl Virus",
}

disease_name = disease_mapping.get(result, "Unknown Disease")

# Debugging: Print unique values in 'result' column
print(f"Predicted Result: {result}")
print("Unique values in 'result' column:", df["result"].unique())

# Find the matching row
row = df[df["result"] == result]
print(f"Matching Row:\n{row}")

if not row.empty:
    tonic = row["tonic"].values[0]
    bio_fertilizer = row["biological fertilizers"].values[0]
    chemical = row["chemical"].values[0]
else:
    print("No matching row found for result:", result)
    tonic = bio_fertilizer = chemical = "Normal Category"

# Print output for verification
print(f"Disease Identified: {disease_name}")
print(f"Tonic: {tonic}")
print(f"Biological Fertilizers: {bio_fertilizer}")
print(f"Chemical: {chemical}")

context = {
    'class': disease_name,
    'tonic': tonic,
    'bio_fertilizer': bio_fertilizer,
    'chemical': chemical
}
