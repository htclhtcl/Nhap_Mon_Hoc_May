import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_blobs

st.title("Demo Thuật toán KNN (K-Nearest Neighbors)")

# 1. Tạo dữ liệu giả lập
st.sidebar.header("Cấu hình KNN")
n_neighbors = st.sidebar.slider("Chọn số lượng K (hàng xóm):", 1, 15, 3)

X, y = make_blobs(n_samples=100, centers=2, random_state=42, cluster_std=2.0)

# 2. Huấn luyện mô hình
clf = KNeighborsClassifier(n_neighbors=n_neighbors)
clf.fit(X, y)

# 3. Vẽ biểu đồ
fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
ax.set_title(f"KNN với K = {n_neighbors}")
st.pyplot(fig)

# 4. Dự đoán điểm mới
st.subheader("Dự đoán điểm mới")
new_x = st.slider("Tọa độ X", -10.0, 10.0, 0.0)
new_y = st.slider("Tọa độ Y", -10.0, 10.0, 0.0)

prediction = clf.predict([[new_x, new_y]])
st.write(f"Điểm ({new_x}, {new_y}) thuộc nhóm: {'Đỏ' if prediction[0] == 1 else 'Xanh'}")