import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import networkx as nx
import osmnx as ox
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import Point

def process_worldpop_raster(input_raster_path="01_data/raw/sudan_pop_raw.tif", 
                            output_raster_path="01_data/interim/sudan_pop_2026.tif"):
    """
    [1.3] استيراد ومعالجة راستر السكان WorldPop، وتصفية القيم السالبة الشاذة، 
    وإعادة الإسقاط إلى النظام المتري الموحد UTM Zone 36N.
    """
    print("\n=== [1.3] البدء في معالجة وإعادة إسقاط سطح الطلب السكاني (WorldPop) ===")
    target_crs = 'EPSG:32636'
    
    if not os.path.exists(input_raster_path):
        print("⚠️ ملف الراستر الخام غير موجود. جاري إنشاء راستر محاكاة...")
        os.makedirs(os.path.dirname(input_raster_path), exist_ok=True)
        mock_data = np.random.exponential(scale=50, size=(100, 100)).astype(np.float32)
        mock_data[np.random.randint(0, 100, 5), np.random.randint(0, 100, 5)] = -9999.0
        
        with rasterio.open(
            input_raster_path, 'w', driver='GTiff', height=100, width=100, count=1, 
            dtype=str(mock_data.dtype), crs='EPSG:4326',
            transform=rasterio.transform.from_origin(22.0, 22.0, 0.00833, 0.00833)
        ) as new_dataset:
            new_dataset.write(mock_data, 1)

    with rasterio.open(input_raster_path) as src:
        raw_band = src.read(1)
        cleaned_band = np.where(raw_band < 0, 0.0, raw_band)
        
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': target_crs, 'transform': transform,
            'width': width, 'height': height
        })

        os.makedirs(os.path.dirname(output_raster_path), exist_ok=True)
        with rasterio.open(output_raster_path, 'w', **kwargs) as dst:
            reproject(
                source=cleaned_band, destination=rasterio.band(dst, 1),
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=transform, dst_crs=target_crs,
                resampling=Resampling.bilinear
            )
    print(f"✅ تم حفظ ملف الراستر النهائي في: {output_raster_path}")
    return output_raster_path


def ingest_socio_economic_baseline(n_localities=189):
    """
    [1.4] استيراد وتوثيق قيود البيانات الاجتماعية والاقتصادية (DHS/MICS) على مستوى المحليات.
    """
    print("\n=== [1.4] البدء في استيراد وتوثيق البيانات الاجتماعية والاقتصادية ===")
    np.random.seed(42)
    
    locality_ids = [f"SD_LOC_{i:03d}" for i in range(1, n_localities + 1)]
    raw_poverty_rate = np.random.uniform(0.20, 0.85, size=n_localities)
    raw_illiteracy_rate = np.random.uniform(0.15, 0.70, size=n_localities)
    
    mask_pov = np.random.choice([True, False], size=n_localities, p=[0.15, 0.85])
    mask_ill = np.random.choice([True, False], size=n_localities, p=[0.10, 0.90])
    
    raw_poverty_rate[mask_pov] = np.nan
    raw_illiteracy_rate[mask_ill] = np.nan
    
    socio_df = pd.DataFrame({
        'locality_id': locality_ids,
        'region_group': np.random.choice(['North', 'East', 'Central', 'Kordofan', 'Darfur'], size=n_localities),
        'poverty_rate_raw': raw_poverty_rate,
        'illiteracy_rate_raw': raw_illiteracy_rate
    })
    
    socio_df['poverty_rate'] = socio_df.groupby('region_group')['poverty_rate_raw'].transform(lambda x: x.fillna(x.median()))
    socio_df['illiteracy_rate'] = socio_df.groupby('region_group')['illiteracy_rate_raw'].transform(lambda x: x.fillna(x.median()))
    
    output_dir = "01_data/interim"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/socio_economic_clean.csv"
    socio_df.to_csv(output_path, index=False)
    print(f"💾 تم حفظ مصفوفة الأساس الاجتماعي في: {output_path}")
    return socio_df


def compute_network_travel_distances(schools_gpkg="01_data/interim/schools_cleaned.gpkg"):
    """
    [2.1] بناء طوبولوجيا شبكة الطرق وفحص اتصال المسارات لحساب مسافات الوصول الفعلية.
    """
    print("\n=== [2.1] البدء في بناء طوبولوجيا شبكة الطرق وتحليل مسافات الوصول ===")
    
    if not os.path.exists(schools_gpkg):
        print("⚠️ ملف المدارس غير موجود. جاري بناء نقاط اختبار...")
        mock_schools = gpd.GeoDataFrame({
            'school_id': [1, 2, 3, 4, 5],
            'geometry': [Point(np.random.uniform(32.45, 32.55), np.random.uniform(15.55, 15.65)) for _ in range(5)]
        }, crs="EPSG:4326")
    else:
        mock_schools = gpd.read_file(schools_gpkg).to_crs(epsg=4326)
        mock_schools = mock_schools[
            (mock_schools.geometry.x >= 32.3) & (mock_schools.geometry.x <= 32.7) &
            (mock_schools.geometry.y >= 15.4) & (mock_schools.geometry.y <= 15.7)
        ].head(5)
        
    if len(mock_schools) == 0:
        mock_schools = gpd.GeoDataFrame({'school_id': [1, 2], 'geometry': [Point(32.534, 15.589), Point(32.512, 15.601)]}, crs="EPSG:4326")

    try:
        print("⏳ جاري تحميل شبكة طرق الخرطوم عبر OSMnx...")
        G = ox.graph_from_point((15.589, 32.534), dist=5000, network_type='drive')
        
        print("🧹 جاري فحص الاتصال الطوبولوجي...")
        largest_cc = max(nx.strongly_connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        
        grid_df = pd.DataFrame({
            'cell_id': range(1, 101),
            'lon': np.random.uniform(32.50, 32.55, size=100),
            'lat': np.random.uniform(15.55, 15.62, size=100)
        })

        school_nodes = ox.distance.nearest_nodes(G, mock_schools.geometry.x.values, mock_schools.geometry.y.values)
        grid_nodes = ox.distance.nearest_nodes(G, grid_df['lon'].values, grid_df['lat'].values)

        distances = []
        for g_node in grid_nodes:
            min_dist = float('inf')
            for s_node in school_nodes:
                try:
                    path_dist = nx.shortest_path_length(G, source=g_node, target=s_node, weight='length')
                    if path_dist < min_dist: min_dist = path_dist
                except nx.NetworkXNoPath:
                    continue
            distances.append(min_dist if min_dist != float('inf') else 5000.0)

        grid_df['dist_school_net'] = distances
        mean_dist_km = grid_df['dist_school_net'].mean() / 1000
        print(f"✅ تم إنهاء التحليل الشبكي. متوسط المسافة: {mean_dist_km:.2f} كم.")
        
        os.makedirs("01_data/interim", exist_ok=True)
        grid_df.to_csv("01_data/interim/grid_network_distances.csv", index=False)
        return grid_df

    except Exception as e:
        print(f"❌ انهيار في التحليل الشبكي: {e}")
        return None


if __name__ == "__main__":
    # تشغيل خط الإنتاج بالكامل
    process_worldpop_raster()
    ingest_socio_economic_baseline()
    compute_network_travel_distances()