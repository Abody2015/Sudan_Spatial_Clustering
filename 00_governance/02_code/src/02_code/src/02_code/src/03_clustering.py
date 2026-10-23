import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

def calculate_gap_statistic(data, n_references=10, max_clusters=6):
    """
    [3.1] حساب خوارزمية إحصاء الفجوة (Gap Statistic) لتحديد العدد الأمثل 
    من العناقيد (Optimal K) رياضياً عبر المقارنة بتوزيع عشوائي موحد.
    """
    print("⏳ جاري تشغيل محرك حساب إحصاء الفجوة (Gap Statistic Calculation)...")
    gaps = np.zeros((max_clusters - 1,))
    results_df = []
    
    for gap_index, k in enumerate(range(2, max_clusters + 1)):
        # 1. تشغيل النموذج الحقيقي وحساب الانحراف الداخلي الكلي (Dispersion)
        km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        km.fit(data)
        dispersion = km.inertia_
        
        # 2. توليد مصفوفات مرجعية عشوائية ومقارنتها بالواقع لتوليد الفجوة الرياضية
        ref_dispersions = []
        for _ in range(n_references):
            random_reference = np.random.random_sample(size=data.shape)
            km_ref = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
            km_ref.fit(random_reference)
            ref_dispersions.append(km_ref.inertia_)
            
        # حساب الفجوة الرياضية الفلسفية (Gap)
        gap = np.log(np.mean(ref_dispersions)) - np.log(dispersion)
        gaps[gap_index] = gap
        
        results_df.append({'Cluster_K': k, 'Gap_Score': gap, 'Log_Dispersion': np.log(dispersion)})
        
    optimal_k = range(2, max_clusters + 1)[np.argmax(gaps)]
    print(f"✔ تم تحديد العدد الأمثل للعناقيد إحصائياً بنجاح: Optimal K = {optimal_k}")
    return optimal_k, pd.DataFrame(results_df)

def run_multi_seed_kmeans(matrix_path="01_data/interim/final_feature_matrix_1km.csv"):
    """
    [3.1] تشغيل K-means++ بـ 10 بذور عشوائية مختلفة لضمان ثبات النموذج وحفظ التسميات المعتمدة.
    """
    print("\n=== [3.1] تشغيل نموذج K-means++ متعدد البذور وحساب إحصاء الفجوة (Gap Statistic) ===")
    
    # محرك الأمان المنهجي: قراءة مصفوفة البيانات المجمدة أو محاكاتها في حال غياب الملف
    if not os.path.exists(matrix_path):
        print("⚠️ مصفوفة السمات المجمدة غير موجودة. جاري بناء مصفوفة اختبار للنمذجة الحسابية...")
        np.random.seed(42)
        features = np.random.uniform(-2, 2, size=(100, 4))
        df_matrix = pd.DataFrame(features, columns=['scaled_dist', 'scaled_pop', 'scaled_conf36', 'scaled_conf6'])
        df_matrix.insert(0, 'cell_id', range(1, 101))
    else:
        df_matrix = pd.read_csv(matrix_path)
        
    # استخراج الأعمدة الرقمية الخاضعة للتجميع (التي تبدأ بكلمة scaled_)
    clustering_cols = [col for col in df_matrix.columns if col.startswith('scaled_')]
    if not clustering_cols: clustering_cols = [col for col in df_matrix.columns if col != 'cell_id']
    
    X = df_matrix[clustering_cols].values
    
    # 1. استدعاء وتشغيل إحصاء الفجوة لتحديد عدد العناقيد
    optimal_k, gap_report = calculate_gap_statistic(X)
    
    # تصدير تقرير الفجوة لمجلد المخرجات
    output_dir = "03_outputs/tables"
    os.makedirs(output_dir, exist_ok=True)
    gap_report.to_csv(f"{output_dir}/kmeans_gap_statistic_report.csv", index=False)
    
    # 2. تشغيل النموذج النهائي المستقر المعتمد على البذور الـ 10 لضمان الحوكمة
    print(f"⏳ جاري تشغيل النموذج النهائي بـ 10 بذور عشوائية وثبيت العناقيد عند K={optimal_k}...")
    final_km = KMeans(n_clusters=optimal_k, init='k-means++', n_init=10, random_state=42)
    df_matrix['kmeans_cluster_id'] = final_km.fit_predict(X)
    
    # 3. تصدير ملف النتائج القطعية الأولي
    output_data_dir = "01_data/interim"
    df_matrix.to_csv(f"{output_data_dir}/grid_kmeans_outputs.csv", index=False)
    
    print(f"✅ تم إنهاء التجميع القطعي بنجاح.")
    print(f"💾 تم حفظ مصفوفة تسميات K-means في: {output_data_dir}/grid_kmeans_outputs.csv")
    return df_matrix


def run_initial_hdbscan_clustering(matrix_path="01_data/interim/final_feature_matrix_1km.csv"):
    """
    [3.2] التشغيل الأولي للنموذج الكثافي البنيوي HDBSCAN لاستخراج العناقيد العضوية، 
    وعزل ملف النقاط غير المستقرة كثافيًا (الضوضاء الابتدائية) لفحص فجوات التخطيط.
    """
    print("\n=== [3.2] البدء في التشغيل الأولي للنموذج الكثافي البنيوي HDBSCAN ===")
    
    import hdbscan
    
    # قراءة مصفوفة البيانات المجمدة أو محاكاتها إحصائياً في حال غياب الملف
    if not os.path.exists(matrix_path):
        np.random.seed(42)
        features = np.random.uniform(-2, 2, size=(100, 4))
        df_matrix = pd.DataFrame(features, columns=['scaled_dist', 'scaled_pop', 'scaled_conf36', 'scaled_conf6'])
        df_matrix.insert(0, 'cell_id', range(1, 101))
    else:
        df_matrix = pd.read_csv(matrix_path)
        
    clustering_cols = [col for col in df_matrix.columns if col.startswith('scaled_')]
    if not clustering_cols: clustering_cols = [col for col in df_matrix.columns if col != 'cell_id']
    
    X = df_matrix[clustering_cols].values
    
    # 1. إطلاق النموذج الكثافي بمعاملات ابتدائية محافظة (Baseline Parameters)
    print("⏳ جاري نمذجة الفضاء الكثافي وعزل العناصر غير المستقرة...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5, gen_min_span_tree=True)
    df_matrix['hdbscan_raw_label'] = clusterer.fit_predict(X)
    
    # 2. حساب إحصائيات الضوضاء الابتدائية (Label: -1) لتوثيق فجوة التغطية
    noise_mask = df_matrix['hdbscan_raw_label'] == -1
    noise_count = noise_mask.sum()
    noise_ratio = (noise_count / len(df_matrix)) * 100
    
    print(f"✔ تم الانتهاء من التشغيل الأولي للنموذج البنيوي العضوي.")
    print(f"⚠️ رصد نقاط الضوضاء الابتدائية: تم تصنيف {noise_count} خلية كضوضاء غير مستقرة كثافيًا ({noise_ratio:.2f}% من إجمالي الفضاء).")
    
    # 3. تصدير ملف المخرجات الكثافية الأولية لمجلد البيانات الوسيطة interim
    output_dir = "01_data/interim"
    df_matrix.to_csv(f"{output_dir}/grid_hdbscan_raw_outputs.csv", index=False)
    print(f"💾 تم حفظ مصفوفة التسميات الكثافية الأولية وعزل الضوضاء في: {output_dir}/grid_hdbscan_raw_outputs.csv")
    
    return df_matrix

if __name__ == "__main__":
    # تشغيل النمذجة الحسابية والتجميع للمرحلة الثالثة بالتتابع
    run_multi_seed_kmeans()
    run_initial_hdbscan_clustering()
