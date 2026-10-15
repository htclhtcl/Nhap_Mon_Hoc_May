import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- 1. Class KNN từ scratch ---
class KNN:
    def __init__(self, k=3, metric='euclidean', p=2):
        self.k = k
        self.metric = metric
        self.p = p # cho Minkowski

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def _compute_distance(self, x1, x2):
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2)**2))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        elif self.metric == 'minkowski':
            return np.sum(np.abs(x1 - x2)**self.p)**(1/self.p)
        return 0

    def predict(self, X):
        X = np.array(X)
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)

    def _predict_single(self, x):
        # Tính khoảng cách từ x đến tất cả điểm trong train set
        distances = [self._compute_distance(x, x_train) for x_train in self.X_train]
        # Lấy k chỉ số gần nhất
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Bầu chọn đa số
        most_common = max(set(k_nearest_labels), key=k_nearest_labels.count)
        return most_common

# --- 2. Giao diện Streamlit ---
st.set_page_config(page_title="KNN Visualization", layout="wide")
st.title("Ứng dụng học thuật toán KNN")

# Sidebar
st.sidebar.header("Cấu hình mô hình")
k_val = st.sidebar.slider("Chọn K", 1, 20, 3)
metric = st.sidebar.selectbox("Distance Metric", ["euclidean", "manhattan", "minkowski"])
test_size = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2)
use_scaler = st.sidebar.checkbox("Chuẩn hóa dữ liệu (StandardScaler)")

# Load dữ liệu
data_source = st.sidebar.radio("Chọn nguồn dữ liệu", ["Iris Dataset", "Upload CSV"])
df = None
if data_source == "Iris Dataset":
    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Dữ liệu đã upload:")
            st.write(df.head())
            target_col = st.sidebar.selectbox("Chọn cột nhãn (Label)", df.columns)
            feature_cols = st.sidebar.multiselect("Chọn các cột đặc trưng", [c for c in df.columns if c != target_col])
            if feature_cols:
                df = df[feature_cols + [target_col]]
                df.columns = feature_cols + ['target']
        except Exception as e:
            st.error(f"Lỗi: {e}")

if df is not None:
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    if use_scaler:
        X = StandardScaler().fit_transform(X)
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # --- Tabs ---
    tab1, tab2, tab3 = st.tabs(["Trực quan", "Huấn luyện & Đánh giá", "Dự đoán thử"])

    with tab1:
        st.subheader("Scatter Plot 2D")
        # Truy cập vào dữ liệu của cột cuối bằng cách dùng df[df.columns[-1]]
        fig = px.scatter(df, x=df.columns[0], y=df.columns[1], color=df[df.columns[-1]].astype(str))
        st.plotly_chart(fig)

    with tab2:
        if st.button("Train Mô hình"):
            # Scratch KNN
            my_knn = KNN(k=k_val, metric=metric)
            my_knn.fit(X_train, y_train)
            y_pred = my_knn.predict(X_test)
            
            # Sklearn comparison
            sk_knn = KNeighborsClassifier(n_neighbors=k_val, metric=metric)
            sk_knn.fit(X_train, y_train)
            sk_pred = sk_knn.predict(X_test)
            
            st.write(f"Accuracy (Scratch): {accuracy_score(y_test, y_pred):.2f}")
            st.write(f"Accuracy (Sklearn): {accuracy_score(y_test, sk_pred):.2f}")
            
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))
            
            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', ax=ax)
            st.pyplot(fig)

    with tab3:
        st.write("Nhập thông số để dự đoán:")
        inputs = []
        for col in df.columns[:-1]:
            inputs.append(st.number_input(f"Nhập {col}", value=0.0))
        
        if st.button("Dự đoán"):
            my_knn = KNN(k=k_val, metric=metric)
            my_knn.fit(X_train, y_train)
            res = my_knn.predict([inputs])
            st.success(f"Kết quả dự đoán loài: {res[0]}")