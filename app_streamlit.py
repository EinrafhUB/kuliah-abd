import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import os

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(layout="wide", page_title="Dasbor Segmentasi Pelanggan")

# --- Fungsi untuk Memuat Data yang Telah Diproses ---
@st.cache_data # Menggunakan cache Streamlit untuk optimasi performa
def load_processed_data(file_name, base_path="hasil_segmentasi_spark/"):
    """Memuat data dari file CSV yang dihasilkan oleh skrip PySpark."""
    file_path = os.path.join(base_path, file_name)
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error(f"File tidak ditemukan: {file_path}. Pastikan skrip PySpark batch telah dijalankan dan outputnya tersimpan di direktori yang benar.")
        return None
    except Exception as e:
        st.error(f"Gagal memuat file {file_path}: {e}")
        return None

# --- Judul Utama Dasbor ---
st.title("Analisis dan Visualisasi Segmentasi Pelanggan")
st.markdown("""
Dasbor interaktif ini menampilkan hasil segmentasi pelanggan yang dilakukan menggunakan PySpark MLlib (KMeans).
Data telah diproses secara batch, dan hasil analisisnya divisualisasikan di bawah ini untuk membantu memahami karakteristik masing-masing segmen pelanggan.
""")

# --- Memuat Semua Data yang Dibutuhkan ---
data_plot_3d_pd = load_processed_data("data_plot_3d.csv")
data_plot_pca_pd = load_processed_data("data_plot_pca.csv")
ringkasan_cluster_pd = load_processed_data("ringkasan_cluster.csv")
sampel_prediksi_cluster_pd = load_processed_data("sampel_prediksi_cluster.csv")


# --- Definisi Nama Cluster (SESUAIKAN DENGAN HASIL ANALISIS ANDA DARI PYSPARK CELL 18) ---
# Pemetaan ini HARUS konsisten dengan yang Anda finalisasi di skrip PySpark (Cell 18)
# setelah menganalisis karakteristik cluster dari output Cell 17.
# Contoh berdasarkan output yang Anda berikan sebelumnya:
cluster_names_map = {
    3: "Stars",
    2: "High potential", # Ingat, cluster ini sangat kecil di contoh data Anda
    1: "Need attention",
    0: "Leaky bucket"
}

# Menerapkan nama cluster ke DataFrame Pandas jika belum ada kolom 'Cluster_Name'
# (Biasanya sudah dilakukan di skrip PySpark saat menyimpan, tapi ini untuk jaga-jaga)
if data_plot_3d_pd is not None and 'Cluster_Name' not in data_plot_3d_pd.columns:
    if 'Cluster' in data_plot_3d_pd.columns:
        data_plot_3d_pd['Cluster_Name'] = data_plot_3d_pd['Cluster'].map(cluster_names_map).fillna("Cluster_" + data_plot_3d_pd['Cluster'].astype(str))
    else:
        st.warning("Kolom 'Cluster' tidak ditemukan di data_plot_3d.csv untuk pemetaan nama.")

if data_plot_pca_pd is not None and 'Cluster_Name' not in data_plot_pca_pd.columns:
    if 'Cluster' in data_plot_pca_pd.columns:
        data_plot_pca_pd['Cluster_Name'] = data_plot_pca_pd['Cluster'].map(cluster_names_map).fillna("Cluster_" + data_plot_pca_pd['Cluster'].astype(str))
    else:
        st.warning("Kolom 'Cluster' tidak ditemukan di data_plot_pca.csv untuk pemetaan nama.")

if sampel_prediksi_cluster_pd is not None and 'Cluster_Name' not in sampel_prediksi_cluster_pd.columns:
    if 'Cluster' in sampel_prediksi_cluster_pd.columns:
        sampel_prediksi_cluster_pd['Cluster_Name'] = sampel_prediksi_cluster_pd['Cluster'].map(cluster_names_map).fillna("Cluster_" + sampel_prediksi_cluster_pd['Cluster'].astype(str))
    else:
        st.warning("Kolom 'Cluster' tidak ditemukan di sampel_prediksi_cluster.csv untuk pemetaan nama.")


# --- Tampilan Ringkasan Statistik Cluster ---
st.header("Ringkasan Karakteristik Segmen Pelanggan")
if ringkasan_cluster_pd is not None:
    ringkasan_display = ringkasan_cluster_pd.copy()
    if 'Cluster' in ringkasan_display.columns:
        ringkasan_display['Nama Segmen'] = ringkasan_display['Cluster'].map(cluster_names_map).fillna("Cluster " + ringkasan_display['Cluster'].astype(str))
        # Atur 'Nama Segmen' sebagai indeks untuk tampilan yang lebih baik
        try:
            ringkasan_display = ringkasan_display.set_index('Nama Segmen')
            # Pilih kolom yang relevan untuk ditampilkan (opsional, bisa semua)
            kolom_tampil = ['Cluster_Size', 'avg(Income)', 'avg(Spending)', 'avg(Seniority)', 'avg(Age)',
                            'avg(Wines)', 'avg(Meat)', 'avg(Fruits)', 'avg(Sweets)', 'avg(Fish)', 'avg(Gold)']
            # Filter kolom yang ada di DataFrame untuk menghindari error
            kolom_tampil_valid = [kol for kol in kolom_tampil if kol in ringkasan_display.columns]
            st.dataframe(ringkasan_display[kolom_tampil_valid].style.format("{:,.2f}"))
        except KeyError:
             st.dataframe(ringkasan_display.style.format("{:,.2f}")) # Tampilkan apa adanya jika set_index gagal

    st.markdown("""
    **Interpretasi Tabel:**
    Tabel ini merangkum karakteristik rata-rata untuk setiap segmen pelanggan yang diidentifikasi.
    - **Cluster_Size**: Jumlah pelanggan dalam segmen.
    - **avg(Income)**: Rata-rata pendapatan tahunan.
    - **avg(Spending)**: Rata-rata total pengeluaran untuk produk.
    - **avg(Seniority)**: Rata-rata lama menjadi pelanggan (dalam bulan).
    - **avg(Age)**: Rata-rata usia pelanggan.
    - *Kolom lainnya menunjukkan rata-rata pengeluaran per kategori produk.*

    Gunakan informasi ini untuk memahami profil setiap segmen dan memvalidasi penamaan segmen Anda.
    """)
else:
    st.warning("File 'ringkasan_cluster.csv' tidak ditemukan atau gagal dimuat. Tidak dapat menampilkan ringkasan statistik.")

# --- Tampilan Sampel Data dengan Prediksi Cluster ---
st.header("Contoh Data Pelanggan dan Segmennya")
if sampel_prediksi_cluster_pd is not None:
    st.dataframe(sampel_prediksi_cluster_pd.head(10)) # Tampilkan 10 sampel pertama
    st.caption(f"Menampilkan sampel {len(sampel_prediksi_cluster_pd)} pelanggan beserta fitur input utama dan segmen cluster yang telah diprediksi.")
else:
    st.warning("File 'sampel_prediksi_cluster.csv' tidak ditemukan atau gagal dimuat.")


# --- Kontainer untuk Visualisasi dengan Tabs ---
st.header("Visualisasi Interaktif Segmen Pelanggan")

tab1, tab2, tab3 = st.tabs(["Plot 3D (Plotly)", "Plot 3D (Matplotlib)", "Plot PCA 2D"])

with tab1:
    st.subheader("Plot Scatter 3D (Income, Seniority, Spending) - Menggunakan Plotly")
    if data_plot_3d_pd is not None and 'Cluster_Name' in data_plot_3d_pd.columns:
        try:
            plotly_3d_fig = go.Figure()
            # Pastikan ada urutan konsisten untuk warna legenda
            sorted_cluster_names = sorted(data_plot_3d_pd['Cluster_Name'].unique())

            for cluster_label_name in sorted_cluster_names:
                df_plot_subset = data_plot_3d_pd[data_plot_3d_pd.Cluster_Name == cluster_label_name]
                plotly_3d_fig.add_trace(go.Scatter3d(
                    x=df_plot_subset['Income'], y=df_plot_subset['Seniority'], z=df_plot_subset['Spending'],
                    mode='markers', marker_size=5, marker_line_width=0.5, name=str(cluster_label_name)
                ))
            plotly_3d_fig.update_traces(hovertemplate='Income: %{x:,.0f} <br>Seniority: %{y:.1f} <br>Spending: %{z:,.0f}')
            plotly_3d_fig.update_layout(
                title="Visualisasi Segmen Pelanggan dalam Ruang 3D",
                width=800, height=700, autosize=True, showlegend=True,
                scene=dict(
                    xaxis=dict(title='Pendapatan (Income)'),
                    yaxis=dict(title='Senioritas (bulan)'),
                    zaxis=dict(title='Total Pengeluaran (Spending)')
                ),
                legend_title_text='Segmen Pelanggan'
            )
            st.plotly_chart(plotly_3d_fig, use_container_width=True)
        except Exception as e:
            st.error(f"Gagal membuat plot 3D Plotly: {e}")
    else:
        st.warning("Data untuk plot 3D Plotly (data_plot_3d.csv) tidak tersedia atau kolom 'Cluster_Name' hilang.")

with tab2:
    st.subheader("Plot Scatter 3D (Income, Seniority, Spending) - Menggunakan Matplotlib")
    if data_plot_3d_pd is not None and 'Cluster_Name' in data_plot_3d_pd.columns:
        try:
            fig_mpl_3d = plt.figure(figsize=(10, 8))
            ax_mpl_3d = fig_mpl_3d.add_subplot(111, projection='3d')
            unique_cluster_names_mpl = sorted(data_plot_3d_pd['Cluster_Name'].unique())
            colors_mpl_3d = plt.cm.get_cmap('tab10', len(unique_cluster_names_mpl))

            for i, cluster_label_name in enumerate(unique_cluster_names_mpl):
                df_plot_subset = data_plot_3d_pd[data_plot_3d_pd.Cluster_Name == cluster_label_name]
                ax_mpl_3d.scatter(
                    df_plot_subset['Income'], df_plot_subset['Seniority'], df_plot_subset['Spending'],
                    color=colors_mpl_3d(i), label=str(cluster_label_name), s=20, alpha=0.6
                )
            ax_mpl_3d.set_xlabel('Pendapatan (Income)')
            ax_mpl_3d.set_ylabel('Senioritas (bulan)')
            ax_mpl_3d.set_zlabel('Total Pengeluaran (Spending)')
            ax_mpl_3d.set_title('Visualisasi Segmen Pelanggan dalam Ruang 3D')
            ax_mpl_3d.legend(title="Segmen Pelanggan")
            st.pyplot(fig_mpl_3d)
        except Exception as e:
            st.error(f"Gagal membuat plot 3D Matplotlib: {e}")
    else:
        st.warning("Data untuk plot 3D Matplotlib (data_plot_3d.csv) tidak tersedia atau kolom 'Cluster_Name' hilang.")


with tab3:
    st.subheader("Plot PCA 2D (Hasil Reduksi Dimensi dari Fitur Clustering)")
    if data_plot_pca_pd is not None and 'Cluster_Name' in data_plot_pca_pd.columns:
        try:
            fig_pca_2d = plt.figure(figsize=(9, 6))
            unique_cluster_names_pca = sorted(data_plot_pca_pd['Cluster_Name'].unique())
            colors_pca = plt.cm.get_cmap('viridis', len(unique_cluster_names_pca))

            for i, cluster_label_name in enumerate(unique_cluster_names_pca):
                cluster_subset_data = data_plot_pca_pd[data_plot_pca_pd.Cluster_Name == cluster_label_name]
                plt.scatter(
                    cluster_subset_data['pca_x'], cluster_subset_data['pca_y'],
                    label=str(cluster_label_name), color=colors_pca(i), alpha=0.7, s=35
                )
            plt.title('Visualisasi Segmen Pelanggan dalam Ruang PCA 2D')
            plt.xlabel('Komponen PCA 1')
            plt.ylabel('Komponen PCA 2')
            plt.legend(title="Segmen Pelanggan")
            plt.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig_pca_2d)
        except Exception as e:
            st.error(f"Gagal membuat plot PCA 2D: {e}")
    else:
        st.warning("Data untuk plot PCA 2D (data_plot_pca.csv) tidak tersedia atau kolom 'Cluster_Name' hilang.")

# --- Informasi Tambahan di Sidebar ---
st.sidebar.header("Informasi Proyek")
st.sidebar.info("""
Aplikasi dasbor ini dibuat untuk memvisualisasikan hasil segmentasi pelanggan.
- **Sumber Data**: `marketing_campaign.csv`.
- **Pemrosesan Data & Clustering**: Apache Spark & PySpark MLlib (KMeans).
- **Visualisasi**: Streamlit, Plotly, Matplotlib.

""")
st.sidebar.markdown("---")
st.sidebar.write("Dibuat sebagai bagian dari tugas pemrosesan data besar.")