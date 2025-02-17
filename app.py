import mlflow
import mlflow.sklearn
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Tải dữ liệu Titanic
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    df.drop(columns=["Name", "Ticket", "Cabin"], inplace=True)
    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
    return df

df = load_data()

# Khởi tạo MLFlow
mlflow.start_run()

# Ghi lại thông tin mô hình
mlflow.log_param("model_type", "Random Forest")
mlflow.log_param("test_size", 0.3)
mlflow.log_param("cross_val_folds", 5)

# Tiền xử lý dữ liệu
encoder = LabelEncoder()
# Huấn luyện encoder với tất cả giá trị có thể có trong 'Sex' và 'Embarked'
encoder.fit(pd.concat([df["Sex"], df["Embarked"]]))  # Kết hợp dữ liệu huấn luyện và các nhãn mới nếu có

df["Sex"] = encoder.transform(df["Sex"])  # Mã hóa dữ liệu
df["Embarked"] = encoder.transform(df["Embarked"])

X = df.drop("Survived", axis=1)
y = df["Survived"]

# Chia tập dữ liệu
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Hiển thị thông tin về tỷ lệ chia dữ liệu
st.subheader("📊 Tỷ lệ phân chia tập dữ liệu (Train/Validation/Test)")
st.write(f"Tập huấn luyện (Train): {len(X_train)} mẫu")
st.write(f"Tập kiểm tra (Test): {len(X_test)} mẫu")
st.write(f"Tập kiểm thử (Validation): {len(X_valid)} mẫu")

# Biểu đồ phân bố các tập dữ liệu
st.subheader("📊 Biểu đồ phân bố các tập dữ liệu")
fig, ax = plt.subplots(figsize=(6, 4))
sns.histplot(y_train, kde=True, color="blue", label="Train", ax=ax)
sns.histplot(y_valid, kde=True, color="orange", label="Validation", ax=ax)
sns.histplot(y_test, kde=True, color="green", label="Test", ax=ax)
ax.set_title("Phân bố các giá trị trong các tập dữ liệu")
ax.set_xlabel("Survived")
ax.set_ylabel("Số lượng")
ax.legend()
st.pyplot(fig)

# Kiểm tra dữ liệu thiếu
st.subheader("🔍 Kiểm tra dữ liệu thiếu (Missing Data)")
st.write(df.isnull().sum())

# Biểu đồ phân tích dữ liệu thiếu
st.subheader("📊 Biểu đồ phân tích dữ liệu thiếu")
fig, ax = plt.subplots(figsize=(8, 4))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis", ax=ax)
ax.set_title("Biểu đồ dữ liệu thiếu")
st.pyplot(fig)

# Chuẩn hóa dữ liệu (Normalization/Standardization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Biểu đồ phân phối dữ liệu sau khi chuẩn hóa
st.subheader("📊 Biểu đồ phân phối dữ liệu sau khi chuẩn hóa")
fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(X_scaled, kde=True, ax=ax)
ax.set_title("Phân phối dữ liệu sau khi chuẩn hóa")
st.pyplot(fig)

# Kiểm tra phân phối dữ liệu
st.subheader("🔍 Kiểm tra phân phối dữ liệu")
st.write("Phân phối dữ liệu trên tập huấn luyện (Train):")
st.write(y_train.value_counts())

st.write("Phân phối dữ liệu trên tập kiểm thử (Validation):")
st.write(y_valid.value_counts())

st.write("Phân phối dữ liệu trên tập kiểm tra (Test):")
st.write(y_test.value_counts())

# Biểu đồ phân phối dữ liệu
st.subheader("📊 Biểu đồ phân phối dữ liệu")
fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(y_train, kde=True, color="blue", label="Train", ax=ax)
sns.histplot(y_valid, kde=True, color="orange", label="Validation", ax=ax)
sns.histplot(y_test, kde=True, color="green", label="Test", ax=ax)
ax.set_title("Phân phối dữ liệu trên các tập dữ liệu")
ax.set_xlabel("Survived")
ax.set_ylabel("Số lượng")
ax.legend()
st.pyplot(fig)

# Huấn luyện mô hình Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Cross-Validation trên tập training
cross_val_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

# Ghi lại kết quả cross-validation vào MLFlow
mlflow.log_metric("cross_val_mean_accuracy", cross_val_scores.mean())

# Hiển thị kết quả cross-validation
st.subheader("🧑‍🏫 Kết Quả Cross-Validation")
st.write(f"Điểm chính xác trung bình của Cross-Validation: {cross_val_scores.mean():.2f}")

# Hiển thị biểu đồ phân phối các điểm Cross-Validation
st.subheader("📊 Phân phối các điểm Cross-Validation")
fig, ax = plt.subplots(figsize=(8, 4))
sns.boxplot(y=cross_val_scores, ax=ax)
ax.set_title("Phân phối điểm Cross-Validation")
ax.set_ylabel("Điểm chính xác")
st.pyplot(fig)

# Huấn luyện mô hình trên toàn bộ tập huấn luyện và đánh giá trên tập kiểm thử
model.fit(X_train, y_train)
y_pred_train = model.predict(X_train)
y_pred_valid = model.predict(X_valid)
y_pred_test = model.predict(X_test)

# Tính toán độ chính xác trên các tập dữ liệu
accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_valid = accuracy_score(y_valid, y_pred_valid)
accuracy_test = accuracy_score(y_test, y_pred_test)

# Ghi lại độ chính xác vào MLFlow
mlflow.log_metric("train_accuracy", accuracy_train)
mlflow.log_metric("valid_accuracy", accuracy_valid)
mlflow.log_metric("test_accuracy", accuracy_test)

# Lưu mô hình vào MLFlow
mlflow.sklearn.log_model(model, "random_forest_model")

# Lấy AUC
fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)

# Hiển thị kết quả trên Streamlit
st.subheader("📊 Kết Quả Huấn Luyện và Kiểm Thử")
st.write(f"Độ chính xác trên tập huấn luyện: {accuracy_train:.2f}")
st.write(f"Độ chính xác trên tập kiểm thử (Validation): {accuracy_valid:.2f}")
st.write(f"Độ chính xác trên tập kiểm tra (Test): {accuracy_test:.2f}")

# Biểu đồ so sánh độ chính xác trên các tập dữ liệu
st.subheader("📊 Biểu đồ so sánh độ chính xác trên các tập dữ liệu")
accuracy_data = [accuracy_train, accuracy_valid, accuracy_test]
labels = ['Train', 'Validation', 'Test']

fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(x=labels, y=accuracy_data, ax=ax, palette='viridis')
ax.set_title('So sánh Độ Chính Xác trên các Tập Dữ Liệu')
ax.set_ylabel('Độ Chính Xác')
st.pyplot(fig)

# --- Biểu đồ Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred_test)
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
ax.set_xlabel("Dự đoán")
ax.set_ylabel("Thực tế")
ax.set_title("Ma Trận Nhầm Lẫn (Confusion Matrix)")
st.pyplot(fig)

# --- Biểu đồ ROC ---
st.subheader("📊 Biểu đồ ROC")
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, color='blue', lw=2, label=f'Random Forest (AUC = {roc_auc:.2f})')
ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('Tỷ lệ sai âm (False Positive Rate)')
ax.set_ylabel('Tỷ lệ đúng dương (True Positive Rate)')
ax.set_title('Biểu đồ ROC')
ax.legend(loc='lower right')
st.pyplot(fig)

# Hiển thị một số dòng dữ liệu từ tập Titanic
st.subheader("📊 Một số dòng dữ liệu từ tập Titanic")
st.write(df.head())  # Hiển thị 5 dòng đầu tiên của dữ liệu

# Dự đoán trên một mẫu dữ liệu mới
sample_data = {
    "Pclass": [3],
    "Sex": "female",  # Dữ liệu mẫu chưa mã hóa
    "Age": [30],
    "SibSp": [1],
    "Parch": [0],
    "Fare": [7.25],
    "Embarked": "C"  # Dữ liệu mẫu chưa mã hóa
}

sample_df = pd.DataFrame(sample_data)

# Xử lý nhãn 'Sex' và 'Embarked' nếu có lỗi (nhãn không có trong dữ liệu huấn luyện)
try:
    sample_df["Sex"] = encoder.transform(sample_df["Sex"])  # Chuyển 'Sex' thành giá trị mã hóa
except ValueError:
    # Nếu gặp lỗi, mã hóa giá trị mới thành một giá trị mặc định (ví dụ: gán giá trị 0)
    sample_df["Sex"] = 0  # Hoặc bất kỳ giá trị mặc định nào bạn muốn

try:
    sample_df["Embarked"] = encoder.transform(sample_df["Embarked"])  # Chuyển 'Embarked' thành giá trị mã hóa
except ValueError:
    # Nếu gặp lỗi, mã hóa giá trị mới thành một giá trị mặc định (ví dụ: gán giá trị 0)
    sample_df["Embarked"] = 0  # Hoặc bất kỳ giá trị mặc định nào bạn muốn

# Tiền xử lý và chuẩn hóa dữ liệu mẫu trước khi dự đoán
sample_scaled = scaler.transform(sample_df)

# Dự đoán
sample_prediction = model.predict(sample_scaled)

# Hiển thị kết quả dự đoán
st.subheader("🔮 Dự đoán trên mẫu dữ liệu mới")
st.write(f"Khả năng sống sót: {'Survived' if sample_prediction[0] == 1 else 'Did not survive'}")

# Kết thúc chạy MLFlow
mlflow.end_run()

# cd "C:\Users\Dell\OneDrive\Pictures\Documents\Code\python\OpenCV\HMVPYTHON\BaiThucHanh1"
# streamlit run app.py
