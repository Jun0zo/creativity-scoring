import pickle
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

with open("labels.pkl", "w") as f:
    labels = pickle.dump(f)

with open("predictions.pkl", "w") as f:
    predictions = pickle.dump(f)

# Calculate the confusion matrix
confusion = confusion_matrix(labels, predictions)

# Calculate accuracy
accuracy = accuracy_score(labels, predictions)

# Print accuracy
print(f"\nAccuracy: {accuracy * 100:.2f}%")

# Print confusion matrix
print("Confusion Matrix:")
print(pd.DataFrame(confusion, columns=["Predicted Negative", "Predicted Positive"], index=["Actual Negative", "Actual Positive"]))


# Calculate and print classification report
target_names = ["Negative", "Positive/"]
print("\nClassification Report:")
print(classification_report(labels, predictions, target_names=target_names))
