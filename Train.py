import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import joblib

CSV_PATH = "gait_pose_sequences.csv"  # Replace with your CSV path
MODEL_PATH = "mlp_model.pkl"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "label_encoder.pkl"

# Load the CSV
df = pd.read_csv(CSV_PATH)

# Separate features (all columns except 'label') and labels
X = df.drop(columns=['label']).values  # Shape (3371, 3960)
y = df['label'].values  # Shape (3371, )

# Encode the labels as integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Scale the features for better training convergence
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and test sets, keep class distribution with stratify
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Define and train MLP classifier
clf = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=1000, random_state=42)
clf.fit(X_train, y_train)

# Evaluate on test set
y_pred = clf.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Save model, scaler, and encoder to disk
joblib.dump(clf, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
joblib.dump(label_encoder, ENCODER_PATH)

print(f"\nModel, scaler, and label encoder saved successfully.")
