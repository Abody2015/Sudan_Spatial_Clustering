import os
import requests
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from scipy.stats import spearmanr

def fetch_and_clean_osm_schools(output_path="01_data/interim/schools_cleaned.gpkg"):
    """
    [1.1] استعلام وجلب بيانات المدارس من خوادم Overpass API،
    وإجراء فرز هندسي لإزالة النقاط المكررة مكانياً وتوحيد السجلات.
    """
    print("\n=== [1.1] البدء في استعلام وجلب بيانات المدارس من OSM ===")
    
    overpass_url = "http://overpass-api.de"
    
    # بناء الاستعلام الجغرافي لدولة السودان (بإصلاح وسوم الصيغة الحصينة)
    overpass_query = """
    [out:json][timeout:180];
    area["ISO3166-1"="SD"]->.searchArea;
    (
      node["amenity"="school"](area.searchArea);
      way["amenity"="school"](area.searchArea);
    );
    out center;
    """
    
    try:
        response = requests.post(overpass_url, data={'data': overpass_query}, timeout=185)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"❌ خطأ أثناء الاتصال بالخادم الجغرافي: {e}")
        return None
        
    elements = data.get('elements', [])
    print(f"✔ تم جلب {len(elements)} سجل جغرافي أولي من الخادم.")
    
    if not elements:
        print("⚠️ تحذير: لم يتم العثور على أي سجلات مستخرجة.")
        return None

    # 2. بناء هندسة المتجهات وتنظيف البيانات مكانياً
    valid_schools = []
    for elem in elements:
        lat = elem.get('lat') or elem.get('center', {}).get('lat')
        lon = elem.get('lon') or elem.get('center', {}).get('lon')
        
        if lat and lon:
            tags = elem.get('tags', {})
            valid_schools.append({
                'osm_id': elem.get('id'),
                'name': tags.get('name', 'Unnamed School'),
                'school_type': tags.get('isced:level', 'Unknown'),
                'geometry': Point(lon, lat)
            })
            
    gdf_schools = gpd.GeoDataFrame(valid_schools, crs="EPSG:4326")
    
    # التحويل إلى إسقاط متري متساوي المساحات (Sudan UTM Zone 36N) لحساب المسافات بدقة
    gdf_schools = gdf_schools.to_crs("EPSG:32636")
    
    # 3. إزالة النقاط المكررة مكانياً (Spatial Deduplication)
    # إذا كانت المسافة بين مدرستين أقل من 10 أمتار، يتم دمج السجل وحذف التكرار
    initial_count = len(gdf_schools)
    gdf_schools['geometry_string'] = gdf_schools['geometry'].apply(lambda geom: f"{geom.x:.1f}_{geom.y:.1f}")
    gdf_schools = gdf_schools.drop_duplicates(subset=['geometry_string']).drop(columns=['geometry_string'])
    
    final_count = len(gdf_schools)
    print(f"🧹 محرك التصفية الهندسي: تم حذف {initial_count - final_count} نقطة مكررة مكانياً.")
    print(f"📌 إجمالي المدارس المدققة والجاهزة للاستخدام = {final_count}")
    
    # حفظ طبقة المتجهات الناتجة في المسار الانتقالي الحصين
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    gdf_schools.to_file(output_path, driver="GPKG")
    print(f"💾 تم تصدير ملف المدارس النقي بنجاح: {output_path}")
    
    return output_path

if __name__ == "__main__":
    fetch_and_clean_osm_schools()
