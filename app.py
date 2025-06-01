import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import random

# Page configuration
st.set_page_config(
    page_title="Restaurant Clustering Analysis",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    
    .cluster-card {
        border: 2px solid #667eea;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
    }
    
    .prediction-result {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Generate sample data
@st.cache_data
def generate_sample_data():
    np.random.seed(42)
    
    cities = ['Jakarta', 'Surabaya', 'Bandung', 'Semarang', 'Yogyakarta', 'Medan']
    cuisines = ['Indonesian', 'Chinese', 'Western', 'Japanese', 'Korean', 'Italian', 'Fast Food']
    parking_options = ['Ada', 'Tidak Ada', 'Terbatas']
    
    data = []
    for i in range(500):
        data.append({
            'Restaurant_Name': f'Restaurant_{i+1}',
            'City': random.choice(cities),
            'Rating': round(random.uniform(3.0, 5.0), 1),
            'Price': random.randint(15000, 150000),
            'Cuisine': random.choice(cuisines),
            'Capacity': random.randint(20, 200),
            'Parking': random.choice(parking_options),
            'Delivery': random.choice(['Ya', 'Tidak'])
        })
    
    return pd.DataFrame(data)

# Clustering function
@st.cache_data
def perform_clustering(df):
    # Prepare features for clustering
    features = df[['Rating', 'Price', 'Capacity']].copy()
    
    # Add encoded features
    features['Delivery_Yes'] = (df['Delivery'] == 'Ya').astype(int)
    features['Parking_Ada'] = (df['Parking'] == 'Ada').astype(int)
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(features_scaled)
    
    df_clustered = df.copy()
    df_clustered['Cluster'] = clusters
    
    return df_clustered, kmeans, scaler

# Prediction function
def predict_cluster(input_data, kmeans, scaler):
    # Prepare input for prediction
    features = np.array([[
        input_data['rating'],
        input_data['price'],
        input_data['capacity'],
        1 if input_data['delivery'] == 'Ya' else 0,
        1 if input_data['parking'] == 'Ada' else 0
    ]])
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Predict cluster
    cluster = kmeans.predict(features_scaled)[0]
    
    # Calculate confidence (simplified)
    distances = kmeans.transform(features_scaled)[0]
    confidence = max(0, 100 - min(distances) * 20)
    
    return cluster, confidence

# Cluster definitions
cluster_info = {
    0: {
        'name': 'Cluster 1 - Restoran Premium',
        'description': 'Restoran high-end dengan rating tinggi dan harga mahal.',
        'target_market': 'Eksekutif & VIP',
        'strategy': 'Luxury Experience'
    },
    1: {
        'name': 'Cluster 2 - Restoran Menengah',
        'description': 'Restoran kelas menengah dengan rating baik dan harga terjangkau.',
        'target_market': 'Keluarga & Pekerja',
        'strategy': 'Fokus Kualitas & Service'
    },
    2: {
        'name': 'Cluster 3 - Restoran Ekonomis',
        'description': 'Restoran dengan harga terjangkau dan fokus pada volume penjualan.',
        'target_market': 'Pelajar & Pekerja Muda',
        'strategy': 'Efisiensi & Volume'
    },
    3: {
        'name': 'Cluster 4 - Restoran Spesialis',
        'description': 'Restoran dengan menu spesialis atau makanan etnik tertentu.',
        'target_market': 'Food Enthusiast',
        'strategy': 'Diferensiasi Menu'
    }
}

# Sidebar navigation
st.sidebar.title("🍽️ Restaurant Clustering")
page = st.sidebar.selectbox(
    "Pilih Halaman:",
    ["Background", "Data Visualization", "Clustering Results", "Prediction Form"]
)

# Load data
df = generate_sample_data()
df_clustered, kmeans_model, scaler_model = perform_clustering(df)

# PAGE 1: BACKGROUND
if page == "Background":
    st.markdown("""
    <div class="main-header">
        <h1>🍽️ Restaurant Clustering Analysis</h1>
        <p>Analisis clustering untuk mengelompokkan restoran berdasarkan karakteristik bisnis</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Tujuan Project
        
        Project ini bertujuan untuk:
        - Mengelompokkan restoran berdasarkan karakteristik bisnis
        - Membantu pengusaha memahami posisi kompetitif
        - Memberikan rekomendasi strategi bisnis
        - Memprediksi cluster untuk restoran baru
        """)
        
    with col2:
        st.markdown("""
        ### 📊 Data yang Dianalisis
        
        - **Rating**: Penilaian pelanggan (1.0 - 5.0)
        - **Harga**: Rata-rata harga makanan (IDR)
        - **Jenis Makanan**: Kategori cuisine
        - **Kapasitas**: Jumlah tempat duduk
        - **Fasilitas**: Parkir dan delivery
        """)
    
    # Dataset overview
    st.markdown("### 📈 Overview Dataset")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>500</h3>
            <p>Total Restoran</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>6</h3>
            <p>Kota</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>7</h3>
            <p>Jenis Cuisine</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>4</h3>
            <p>Cluster</p>
        </div>
        """, unsafe_allow_html=True)

# PAGE 2: DATA VISUALIZATION  
elif page == "Data Visualization":
    st.markdown("""
    <div class="main-header">
        <h1>📊 Data Visualization</h1>
        <p>Eksplorasi visual dari dataset restoran</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Rating distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Distribusi Rating")
        fig_rating = px.histogram(df, x='Rating', nbins=20, 
                                title="Distribusi Rating Restoran")
        fig_rating.update_layout(showlegend=False)
        st.plotly_chart(fig_rating, use_container_width=True)
    
    with col2:
        st.subheader("💰 Distribusi Harga")
        fig_price = px.histogram(df, x='Price', nbins=25,
                               title="Distribusi Harga Restoran")
        fig_price.update_layout(showlegend=False)
        st.plotly_chart(fig_price, use_container_width=True)
    
    # City and cuisine analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏙️ Restoran per Kota")
        city_counts = df['City'].value_counts()
        fig_city = px.bar(x=city_counts.index, y=city_counts.values,
                         title="Jumlah Restoran per Kota")
        st.plotly_chart(fig_city, use_container_width=True)
    
    with col2:
        st.subheader("🍜 Jenis Cuisine")
        cuisine_counts = df['Cuisine'].value_counts()
        fig_cuisine = px.pie(values=cuisine_counts.values, names=cuisine_counts.index,
                           title="Distribusi Jenis Cuisine")
        st.plotly_chart(fig_cuisine, use_container_width=True)
    
    # Correlation analysis
    st.subheader("🔗 Analisis Korelasi")
    fig_scatter = px.scatter(df, x='Rating', y='Price', size='Capacity',
                           color='Cuisine', title="Rating vs Price vs Capacity")
    st.plotly_chart(fig_scatter, use_container_width=True)

# PAGE 3: CLUSTERING RESULTS
elif page == "Clustering Results":
    st.markdown("""
    <div class="main-header">
        <h1>🎯 Clustering Results</h1>
        <p>Hasil analisis clustering dengan K-Means</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Cluster visualization
    st.subheader("📊 Visualisasi Cluster")
    
    fig_cluster = px.scatter(df_clustered, x='Rating', y='Price', 
                           color='Cluster', size='Capacity',
                           title="Hasil Clustering Restoran",
                           color_discrete_sequence=px.colors.qualitative.Set1)
    st.plotly_chart(fig_cluster, use_container_width=True)
    
    # Cluster analysis
    st.subheader("📈 Analisis per Cluster")
    
    for cluster_id in sorted(df_clustered['Cluster'].unique()):
        cluster_data = df_clustered[df_clustered['Cluster'] == cluster_id]
        info = cluster_info[cluster_id]
        
        with st.expander(f"🔍 {info['name']} ({len(cluster_data)} restoran)"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Rata-rata Rating", f"{cluster_data['Rating'].mean():.1f}")
                st.metric("Rata-rata Harga", f"Rp {cluster_data['Price'].mean():,.0f}")
            
            with col2:
                st.metric("Rata-rata Kapasitas", f"{cluster_data['Capacity'].mean():.0f}")
                delivery_pct = (cluster_data['Delivery'] == 'Ya').mean() * 100
                st.metric("Delivery (%)", f"{delivery_pct:.0f}%")
            
            with col3:
                st.write("**Target Market:**", info['target_market'])
                st.write("**Strategi:**", info['strategy'])
            
            st.write("**Deskripsi:**", info['description'])

# PAGE 4: PREDICTION FORM
elif page == "Prediction Form":
    st.markdown("""
    <div class="main-header">
        <h1>🔮 Prediction Form</h1>
        <p>Prediksi cluster untuk restoran baru</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            restaurant_name = st.text_input("Nama Restoran", placeholder="Warung Bu Sari")
            city = st.selectbox("Kota", ['Jakarta', 'Surabaya', 'Bandung', 'Semarang', 'Yogyakarta', 'Medan'])
            rating = st.slider("Rating", 1.0, 5.0, 4.0, 0.1)
            price = st.number_input("Harga Rata-rata (IDR)", min_value=5000, value=50000, step=5000)
        
        with col2:
            cuisine = st.selectbox("Jenis Makanan", 
                                 ['Indonesian', 'Chinese', 'Western', 'Japanese', 'Korean', 'Italian', 'Fast Food'])
            capacity = st.number_input("Kapasitas Tempat Duduk", min_value=10, value=50, step=5)
            parking = st.selectbox("Fasilitas Parkir", ['Ada', 'Tidak Ada', 'Terbatas'])
            delivery = st.selectbox("Layanan Delivery", ['Ya', 'Tidak'])
        
        submitted = st.form_submit_button("🔮 Prediksi Cluster", use_container_width=True)
        
        if submitted:
            if restaurant_name:
                # Prepare input data
                input_data = {
                    'rating': rating,
                    'price': price,
                    'capacity': capacity,
                    'delivery': delivery,
                    'parking': parking
                }
                
                # Make prediction
                predicted_cluster, confidence = predict_cluster(input_data, kmeans_model, scaler_model)
                cluster_details = cluster_info[predicted_cluster]
                
                # Display results
                st.markdown(f"""
                <div class="prediction-result">
                    <h2>🎯 Hasil Prediksi untuk "{restaurant_name}"</h2>
                    <h3>{cluster_details['name']}</h3>
                    <p>{cluster_details['description']}</p>
                    <div style='margin-top: 1rem;'>
                        <strong>Confidence Score: {confidence:.1f}%</strong><br>
                        <strong>Target Market: {cluster_details['target_market']}</strong><br>
                        <strong>Strategi Rekomendasi: {cluster_details['strategy']}</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Show similar restaurants
                similar_restaurants = df_clustered[df_clustered['Cluster'] == predicted_cluster].head(5)
                
                st.subheader("🔍 Restoran Serupa dalam Cluster yang Sama")
                st.dataframe(similar_restaurants[['Restaurant_Name', 'City', 'Rating', 'Price', 'Cuisine']], 
                           use_container_width=True)
            else:
                st.error("Mohon isi nama restoran!")

# Footer
st.markdown("---")
st.markdown("**Restaurant Clustering Analysis** - Dibuat dengan ❤️ menggunakan Streamlit")
