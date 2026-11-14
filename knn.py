import sklearn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# Check library versions
print("sklearn: %s" % sklearn.__version__)
print("numpy: %s" % np.__version__)
print("pandas: %s" % pd.__version__)
print("seaborn: %s" % sns.__version__)

# 1. Load data (pointing directly to data_knn.txt in your current folder)
col_names = ["x1", "x2", "x3", "x4", "class"]
df = pd.read_table("data_knn.txt", sep=",", header=None, names=col_names)

print(df.tail(3))
df.info()

# 2. Visualize data
sns.pairplot(df, hue="class")
plt.show()

class_1 = df[df["class"] == 0].shape[0]
class_2 = df[df["class"] == 1].shape[0]
label = [class_1, class_2]
plt.pie(label, labels=["class_1", "class_2"], startangle=90, autopct="%1.1f%%")
plt.tight_layout()
plt.show()

# 3. Define features and target
X = df.iloc[:, :-1].to_numpy()
Y = df.iloc[:, -1].to_numpy()
print("features", X)
print("target", Y)

# 4. Handle missing data
print("Missing values per column:\n", df.isnull().sum())
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer = imputer.fit(X)
X = imputer.transform(X)

# 5. Feature standardization
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

# 6. Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0
)
print("X_train", X_train.shape)
print("Y_train", y_train.shape)
print("X_test", X_test.shape)
print("Y_test", y_test.shape)

# 7. Build and train KNN classifier
knn = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
knn.fit(X_train, y_train)

# 8. Predict and check accuracy
Y_predict = knn.predict(X_test)
print(Y_predict)
Accuracy = accuracy_score(Y_predict, y_test)
print("Accuracy Score:", Accuracy)

# 9. Predict on a new unseen data point
new_point = np.array([[3.1111, 8.8282, 4.111, -2.34422]])
new_prediction = knn.predict(sc_X.transform(new_point))
print("Predicted value on new data:", new_prediction)

