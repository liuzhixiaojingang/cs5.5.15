import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import joblib
from torchvision.models import vgg11
import torch
from torchvision import transforms
from sklearn.decomposition import PCA
import io
import matplotlib.pyplot as plt

# 设置页面标题和布局
st.set_page_config(page_title="烧伤分类系统", layout="wide")
st.title("烧伤分类系统")

# 初始化模型和转换
@st.cache_resource
def load_models():
    # 加载预训练的VGG11模型
    model = vgg11(pretrained=True)
    model.eval()
    
    # 加载预训练的MLP模型
    mlp_model = joblib.load('best_mlp_model.pkl')
    
    # 初始化PCA
    pca = PCA(n_components=20)
    
    return model, mlp_model, pca

vgg_model, mlp_model, pca = load_models()

# 图像预处理
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 状态管理
if 'features' not in st.session_state:
    st.session_state.features = None
if 'pca_features' not in st.session_state:
    st.session_state.pca_features = None
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None

# 侧边栏
with st.sidebar:
    st.header("操作面板")
    uploaded_file = st.file_uploader("上传烧伤图片", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        st.session_state.uploaded_image = Image.open(uploaded_file)
        st.image(st.session_state.uploaded_image, caption="上传的图片", use_column_width=True)

# 主界面
col1, col2 = st.columns([1, 1])

with col1:
    st.header("特征提取")
    
    if st.button("提取特征", disabled=st.session_state.uploaded_image is None):
        if st.session_state.uploaded_image is not None:
            # 预处理图像
            img_tensor = preprocess(st.session_state.uploaded_image).unsqueeze(0)
            
            # 提取特征
            with torch.no_grad():
                features = vgg_model.features(img_tensor)
                features = vgg_model.avgpool(features)
                features = torch.flatten(features, 1)
                features = vgg_model.classifier[:4](features)  # 获取倒数第二层特征
            
            st.session_state.features = features.numpy().flatten()
            st.success("特征提取完成！")
    
    if st.session_state.features is not None:
        st.write(f"提取的特征维度: {st.session_state.features.shape[0]}")
        
        # 下载特征按钮
        buffer = io.BytesIO()
        np.save(buffer, st.session_state.features)
        buffer.seek(0)
        st.download_button(
            label="下载特征",
            data=buffer,
            file_name="extracted_features.npy",
            mime="application/octet-stream"
        )

with col2:
    st.header("降维与预测")
    
    if st.button("降维至20维", disabled=st.session_state.features is None):
        if st.session_state.features is not None:
            # 使用PCA降维
            pca_features = pca.fit_transform(st.session_state.features.reshape(1, -1))
            st.session_state.pca_features = pca_features.flatten()
            st.success("降维完成！")
    
    if st.session_state.pca_features is not None:
        st.write(f"降维后的特征维度: {st.session_state.pca_features.shape[0]}")
        
        # 可视化降维结果
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(range(len(st.session_state.pca_features)), st.session_state.pca_features)
        ax.set_title("降维后的20维特征")
        ax.set_xlabel("特征维度")
        ax.set_ylabel("特征值")
        st.pyplot(fig)
    
    if st.button("预测分类", disabled=st.session_state.pca_features is None):
        if st.session_state.pca_features is not None:
            # 使用MLP模型进行预测
            prediction = mlp_model.predict(st.session_state.pca_features.reshape(1, -1))
            st.session_state.prediction = prediction[0]
            st.success("预测完成！")
    
    if st.session_state.prediction is not None:
        st.subheader("预测结果")
        st.write(f"预测类别: {st.session_state.prediction}")
        
        # 类别解释
        class_descriptions = {
            0: "正常皮肤",
            1: "浅二度烫伤",
            2: "深二度烫伤",
            3: "三度烫伤",
            4: "电击烧伤",
            5: "火焰烧伤"
        }
        st.write(f"类别描述: {class_descriptions.get(st.session_state.prediction, '未知类别')}")

# 融合结果显示
if st.session_state.features is not None and st.session_state.pca_features is not None:
    st.header("特征融合结果")
    
    # 创建DataFrame显示特征
    df = pd.DataFrame({
        "原始特征": st.session_state.features[:20],  # 只显示前20维
        "降维特征": np.concatenate([st.session_state.pca_features, np.zeros(20 - len(st.session_state.pca_features))])
    })
    
    st.dataframe(df.style.format("{:.4f}"), height=400)
    
    # 可视化对比
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["原始特征"], label="原始特征(前20维)")
    ax.plot(df["降维特征"], label="降维特征")
    ax.legend()
    ax.set_title("原始特征与降维特征对比")
    st.pyplot(fig)