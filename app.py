import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Tải dữ liệu Titanic
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    # Xóa các cột không cần thiết
    df.drop(columns=["Name", "Ticket", "Cabin"], inplace=True)
    # Điền giá trị thiếu cho độ tuổi và nơi khởi hành
    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
    return df

df = load_data()

# Tiêu đề chính
st.title("Phân Tích Dữ Liệu Titanic 🚢")

# Hiển thị dữ liệu
st.subheader("📌 Một số dòng dữ liệu từ tập Titanic")
st.table(df.head())

# Biểu đồ phân phối các giá trị thiếu trong dữ liệu
st.subheader("📊 Biểu đồ phân phối giá trị thiếu trong dữ liệu")
fig, ax = plt.subplots(figsize=(8, 4))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis', ax=ax)
ax.set_title("Phân phối giá trị thiếu trong các cột")
st.pyplot(fig)

# Biểu đồ phân phối giới tính
st.subheader("📊 Phân phối giới tính của hành khách")
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x="Sex", data=df, palette="pastel", ax=ax)
ax.set_title("Phân phối giới tính của hành khách")
st.pyplot(fig)

# Biểu đồ phân phối hạng vé
st.subheader("📊 Phân phối hạng vé")
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x="Pclass", data=df, palette="coolwarm", ax=ax)
ax.set_title("Phân phối hạng vé của hành khách")
st.pyplot(fig)

# Biểu đồ phân phối nơi khởi hành (Embarked)
st.subheader("📊 Phân phối nơi khởi hành của hành khách")
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x="Embarked", data=df, palette="muted", ax=ax)
ax.set_title("Phân phối nơi khởi hành của hành khách")
st.pyplot(fig)

# --- Tiền xử lý dữ liệu ---
# Chuyển đổi các biến phân loại thành số
encoder = LabelEncoder()
df["Sex"] = encoder.fit_transform(df["Sex"])
df["Embarked"] = encoder.fit_transform(df["Embarked"])

# --- Chia dữ liệu: 70% cho training, 15% cho validation, 15% cho test ---
X = df.drop("Survived", axis=1)  # Xử lý tất cả các cột, trừ cột 'Survived'
y = df["Survived"]  # Cột 'Survived' là nhãn mục tiêu

# Chia dữ liệu thành các tập Train, Validation, Test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Hiển thị thông tin về tỷ lệ chia dữ liệu
st.subheader("📊 Tỷ lệ phân chia tập dữ liệu (Train/Validation/Test)")
st.write(f"Tập huấn luyện (Train): {len(X_train)} mẫu")
st.write(f"Tập kiểm tra (Test): {len(X_test)} mẫu")
st.write(f"Tập kiểm thử (Validation): {len(X_valid)} mẫu")

# --- Biểu đồ phân bố các tập dữ liệu ---
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

# --- Huấn luyện mô hình Random Forest ---
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Cross-Validation trên tập training
cross_val_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

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
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Hiển thị kết quả huấn luyện và kiểm thử
st.subheader("📊 Kết Quả Huấn Luyện và Kiểm Thử")
st.write(f"Độ chính xác trên tập kiểm thử: {accuracy:.2f}")

# --- Biểu đồ Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
ax.set_xlabel("Dự đoán")
ax.set_ylabel("Thực tế")
ax.set_title("Ma Trận Nhầm Lẫn (Confusion Matrix)")
st.pyplot(fig)

# --- Biểu đồ ROC ---
fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:,1])
roc_auc = auc(fpr, tpr)

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

# --- Dự đoán trên mẫu dữ liệu ---
st.subheader("🧑‍💻 Dự đoán trên mẫu dữ liệu")
sample = X_test.sample(1)
st.write("Mẫu dữ liệu:")
st.write(sample)

prediction = model.predict(sample)
st.write(f"Dự đoán sống sót: {'Sống' if prediction[0] == 1 else 'Chết'}")


# cd "C:\Users\Dell\OneDrive\Pictures\Documents\Code\python\OpenCV\HMVPYTHON\BaiThucHanh1"
# streamlit run app.py
