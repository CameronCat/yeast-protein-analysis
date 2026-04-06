import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("Loading yeast dataset...")
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "samanemami/yeastcsv",
    "yeast.csv"
)

print("\nDataset Overview:")
print(f"Shape: {df.shape}")
print(f"\nFirst 5 records:\n{df.head()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nClass distribution:\n{df.iloc[:, -1].value_counts()}")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nTraining Random Forest Classifier...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)

y_pred = rf_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5)
print(f"\nCross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

fig = plt.figure(figsize=(18, 12))

ax1 = plt.subplot(2, 3, 1)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

ax1.barh(feature_importance['feature'], feature_importance['importance'])
ax1.set_xlabel('Importance')
ax1.set_title('Feature Importance')
ax1.invert_yaxis()

ax2 = plt.subplot(2, 3, 2)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
            xticklabels=le.classes_, yticklabels=le.classes_)
ax2.set_xlabel('Predicted')
ax2.set_ylabel('Actual')
ax2.set_title('Confusion Matrix')

ax3 = plt.subplot(2, 3, 3)
class_counts = pd.Series(y).value_counts()
ax3.bar(range(len(class_counts)), class_counts.values)
ax3.set_xticks(range(len(class_counts)))
ax3.set_xticklabels(class_counts.index, rotation=45, ha='right')
ax3.set_xlabel('Class')
ax3.set_ylabel('Count')
ax3.set_title('Class Distribution')

ax4 = plt.subplot(2, 3, 4)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)
scatter = ax4.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap='viridis', alpha=0.6)
ax4.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
ax4.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
ax4.set_title('PCA - First 2 Components')
plt.colorbar(scatter, ax=ax4)

ax5 = plt.subplot(2, 3, 5)
corr_matrix = X.corr()
sns.heatmap(corr_matrix, cmap='coolwarm', center=0, ax=ax5,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
ax5.set_title('Feature Correlation Matrix')

ax6 = plt.subplot(2, 3, 6)
metrics = ['Train Accuracy', 'Test Accuracy', 'Mean CV Score']
scores = [
    rf_model.score(X_train_scaled, y_train),
    accuracy,
    cv_scores.mean()
]
bars = ax6.bar(metrics, scores, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
ax6.set_ylabel('Score')
ax6.set_title('Model Performance Metrics')
ax6.set_ylim([0, 1])
for bar in bars:
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('yeast_analysis_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n Analysis complete. Visualization saved as 'yeast_analysis_results.png'")
