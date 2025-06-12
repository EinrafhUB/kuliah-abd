import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Konfigurasi Halaman ---
st.set_page_config(layout="wide", page_title="Dasbor Segmentasi Pelanggan")

# --- Fungsi Pemuatan Data ---
@st.cache_data
def load_data(file_name, index_col=None):
    try:
        df = pd.read_csv(file_name, index_col=index_col)
        return df
    except FileNotFoundError:
        st.error(f"File tidak ditemukan: {file_name}. Pastikan file berada di direktori yang benar.")
        return None
    except Exception as e:
        st.error(f"Gagal memuat file {file_name}: {e}")
        return None

# --- Judul Utama Dasbor ---
st.title("Dasbor Analisis Segmentasi Pelanggan")
st.markdown("Visualisasi interaktif hasil segmentasi pelanggan menggunakan PySpark GMM, mencakup Analisis Eksploratif hingga Profiling Cluster.")

# --- Memuat Data ---
df_pelanggan = load_data("hasil_segmentasi_spark/data_pelanggan_tersegmentasi.csv")
profil_cluster = load_data("hasil_segmentasi_spark/profil_cluster.csv", index_col=0)

if df_pelanggan is None or profil_cluster is None:
    st.stop()

# --- Tampilan Utama dengan Tabs ---
st.header("Analisis Data Pelanggan")
tab1, tab2, tab3 = st.tabs(["Analisis Eksploratif (EDA)", "Analisis & Profil Segmen", "Telusuri Data Mentah"])

# --- Tab 1: Analisis Eksploratif (EDA) ---
with tab1:
    st.subheader("Analisis Umum Data Pelanggan (Sebelum Clustering)")
    st.markdown("Bagian ini menampilkan wawasan awal dari keseluruhan data pelanggan untuk memahami karakteristik umum dan hubungan antar variabel.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("*Total Pengeluaran per Kategori*")
        spend_dist_pd = df_pelanggan[['Wines', 'Fruits', 'Meat', 'Fish', 'Sweets', 'Gold']].sum().sort_values(ascending=False)
        fig_bar, ax_bar = plt.subplots(figsize=(8, 6))
        sns.barplot(x=spend_dist_pd.index, y=spend_dist_pd.values, ax=ax_bar, palette="viridis")
        ax_bar.set_title('Total Pengeluaran Berdasarkan Kategori Produk')
        ax_bar.set_ylabel('Total Pengeluaran')
        plt.xticks(rotation=45)
        st.pyplot(fig_bar)

    with col2:
        st.markdown("*Heatmap Korelasi*")
        fig_heatmap, ax_heatmap = plt.subplots(figsize=(8, 6))
        corr = df_pelanggan[['Age', 'Income', 'Spending', 'Seniority']].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_heatmap)
        ax_heatmap.set_title('Korelasi Antar Variabel Kunci')
        st.pyplot(fig_heatmap)

    st.markdown("---")
    st.markdown("*Hubungan Antar Variabel Kunci*")
    fig_scatter = plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_pelanggan, x='Income', y='Spending', hue='Education', style='Has_child', s=50, alpha=0.7)
    plt.title('Hubungan Antara Income dan Spending')
    plt.xlabel('Pendapatan (Income)')
    plt.ylabel('Total Pengeluaran (Spending)')
    plt.legend(title='Demografi')
    st.pyplot(fig_scatter)


# --- Tab 2: Analisis & Profil Segmen ---
with tab2:
    st.subheader("Analisis Mendalam Hasil Segmentasi")

    st.markdown("*Profil Karakteristik Segmen*")
    st.dataframe(profil_cluster.style.format("{:,.0f}"))

    st.markdown("---")
    st.subheader("Visualisasi Perbandingan Profil Belanja")

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("*Radar Chart Profil Belanja*")
        categories = ['Wines', 'Fruits', 'Meat', 'Fish', 'Sweets', 'Gold']
        radar_fig = go.Figure()
        for cluster_name in profil_cluster.index:
            values = profil_cluster.loc[cluster_name, categories].values
            radar_fig.add_trace(go.Scatterpolar(r=values, theta=categories, fill='toself', name=cluster_name))
        radar_fig.update_layout(title_text="DNA Belanja per Cluster", height=500)
        st.plotly_chart(radar_fig, use_container_width=True)

    with col4:
        st.markdown("*Komposisi Pengeluaran per Cluster*")
        cluster_profile_pct = profil_cluster[categories].divide(profil_cluster['Spending'], axis=0) * 100
        fig_stacked, ax_stacked = plt.subplots(figsize=(8,6))
        cluster_profile_pct.plot(kind='bar', stacked=True, colormap='viridis', ax=ax_stacked)
        ax_stacked.set_title('Komposisi Persentase Pengeluaran')
        ax_stacked.set_ylabel('Persentase dari Total Pengeluaran (%)')
        ax_stacked.set_xlabel('Cluster')
        ax_stacked.legend().set_visible(False)
        st.pyplot(fig_stacked)

    st.markdown("---")
    st.markdown("*Visualisasi 3D Gabungan*")
    fig_3d = go.Figure()
    for cluster_name in sorted(df_pelanggan['Cluster'].unique()):
        cluster_data = df_pelanggan[df_pelanggan['Cluster'] == cluster_name]
        fig_3d.add_trace(go.Scatter3d(
            x=cluster_data['Income'], y=cluster_data['Seniority'], z=cluster_data['Spending'],
            mode='markers', marker=dict(size=5, opacity=0.8), name=cluster_name,
            hovertemplate='<b>Cluster:</b> ' + cluster_name + '<br>' +
                          '<b>Income:</b> %{x:,.0f}<br>' +
                          '<b>Seniority:</b> %{y:.1f} bulan<br>' +
                          '<b>Spending:</b> %{z:,.0f}<extra></extra>'
        ))
    fig_3d.update_layout(
        title_text="Posisi Relatif Segmen Pelanggan",
        scene=dict(xaxis_title='Pendapatan', yaxis_title='Senioritas', zaxis_title='Pengeluaran'),
        height=700, legend_title_text='Cluster'
    )
    st.plotly_chart(fig_3d, use_container_width=True)


# --- Tab 3: Telusuri Data ---
with tab3:
    st.subheader("Data Mentah Pelanggan yang Telah Disegmentasi")
    st.markdown("Gunakan filter di bawah ini untuk menelusuri data pelanggan.")
    
    # Membuat filter interaktif
    cluster_filter = st.multiselect(
        'Filter berdasarkan Cluster:',
        options=sorted(df_pelanggan['Cluster'].unique()),
        default=sorted(df_pelanggan['Cluster'].unique())
    )
    
    filtered_df = df_pelanggan[df_pelanggan['Cluster'].isin(cluster_filter)]
    st.dataframe(filtered_df)


# --- Sidebar ---
st.sidebar.header("Informasi Proyek")
st.sidebar.info("Dasbor ini memvisualisasikan hasil segmentasi pelanggan dari notebook PySpark.")
st.sidebar.markdown("- *Clustering*: Gaussian Mixture Model (GMM)")
st.sidebar.markdown("- *Visualisasi*: Streamlit, Plotly, Matplotlib")
st.sidebar.write("---")
st.sidebar.write("Proyek Akhir Analitik Big Data.")