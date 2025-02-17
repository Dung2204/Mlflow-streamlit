import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

def train_and_log_model():
    # Tải dữ liệu
    data_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(data_url)
    
    # Xử lý dữ liệu
    df.drop(columns=["Name", "Ticket", "Cabin"], inplace=True)
    df.drop_duplicates(inplace=True)
    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
    df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)
    
    # Chia dữ liệu theo tỷ lệ 70/15/15
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    feature_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']
    X_train, y_train = train_df[feature_cols], train_df['Survived']
    X_valid, y_valid = valid_df[feature_cols], valid_df['Survived']
    X_test, y_test = test_df[feature_cols], test_df['Survived']
    
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)
        
        # Cross Validation
        X_train_valid = pd.concat([X_train, X_valid])
        y_train_valid = pd.concat([y_train, y_valid])
        cv_results = cross_validate(model, X_train_valid, y_train_valid, cv=5, scoring='accuracy', return_train_score=True)
        
        # Log kết quả Cross Validation
        mlflow.log_metric("cross_val_train_accuracy_mean", np.mean(cv_results['train_score']))
        mlflow.log_metric("cross_val_valid_accuracy_mean", np.mean(cv_results['test_score']))
        
        # Huấn luyện trên tập train+valid
        model.fit(X_train_valid, y_train_valid)
        
        # Kiểm tra trên tập test
        y_test_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        mlflow.log_metric("test_accuracy", test_accuracy)
        
        # Lưu mô hình vào MLflow
        model_uri = mlflow.sklearn.log_model(model, "random_forest_model").model_uri
        mlflow.register_model(model_uri, "titanic_model")
        
        print("Quá trình huấn luyện hoàn tất!")

if __name__ == "__main__":
    train_and_log_model()
