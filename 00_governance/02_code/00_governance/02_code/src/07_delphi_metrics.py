import os
import numpy as np
import pandas as pd

def calculate_delphi_agreement_metrics():
    print("\n=== [6.2] إطلاق محرك حساب مؤشرات اتفاق الخبراء (Delphi Panel) ===")
    
    # 1. تصحيح الخطأ الأول: ملء مصفوفة رتب 10 خبراء للأنظمة الثلاثة (Alpha, Beta, Gamma)
    # الصفوف = الخبراء الـ 10 | الأعمدة = الترتيب الممنوح للأنظمة الثلاثة من قبل كل خبير
    expert_ranks = np.array([
        [1, 2, 3],  # الخبير 1
        [1, 3, 2],  # الخبير 2
        [2, 1, 3],  # الخبير 3
        [1, 2, 3],  # الخبير 4
        [1, 2, 3],  # الخبير 5
        [2, 1, 3],  # الخبير 6
        [1, 2, 3],  # الخبير 7
        [1, 3, 2],  # الخبير 8
        [2, 1, 3],  # الخبير 9
        [1, 2, 3]   # الخبير 10
    ])
    
    m_experts, n_systems = expert_ranks.shape
    print(f"📊 حجم لجنة التحكيم: {m_experts} خبراء | عدد الأنظمة المعماة: {n_systems}")

    # الحساب الرياضي لمعامل كيندال للتوافق (Kendall's W)
    row_sums = np.sum(expert_ranks, axis=0)
    mean_row_sums = np.mean(row_sums)
    S = np.sum((row_sums - mean_row_sums) ** 2)
    kendall_w = (12 * S) / ((m_experts ** 2) * (n_systems ** 3 - n_systems))
    
    print(f"\n📈 معامل كيندال للتوافق (Kendall's W): {kendall_w:.4f}")
    if kendall_w >= 0.70:
        print("✔ النتيجة: تحقق الإجماع بنجاح وتجاوز عتبة القبول الأكاديمية (>= 0.7).")
    else:
        print("⚠️ توافق الخبراء دون العتبة المطلوبة.")

    # 2. تصحيح الخطأ الثاني: ملء مصفوفة الأصوات الفئوية النوعية للأنظمة الثلاثة
    # الأعمدة: [عدد أصوات أولوية عالية، أولوية متوسطة، أولوية منخفضة]
    category_counts = np.array([
        [7, 2, 1],  # System Archetype: Alpha
        [1, 1, 8],  # System Archetype: Beta
        [2, 7, 1]   # System Archetype: Gamma
    ])
    
    N, k_categories = category_counts.shape
    total_votes_per_row = np.sum(category_counts, axis=1)
    
    p_i = (np.sum(category_counts ** 2, axis=1) - total_votes_per_row) / (total_votes_per_row * (total_votes_per_row - 1))
    P_mean = np.mean(p_i)
    p_j = np.sum(category_counts, axis=0) / (N * m_experts)
    P_e = np.sum(p_j ** 2)
    
    fleiss_kappa = (P_mean - P_e) / (1 - P_e) if P_e != 1 else 1.0
    
    print(f"📊 معامل فليس كابا (Fleiss' Kappa): {fleiss_kappa:.4f}")
    if fleiss_kappa > 0.61:
        print("📌 التفسير: توافق جوهري وقوي جداً (Substantial Agreement) بين المقيمين.")

    # حفظ تقرير جولة دلفي الإحصائي
    output_dir = "03_outputs/metrics"
    os.makedirs(output_dir, exist_ok=True)
    report_path = f"{output_dir}/delphi_validation_report.csv"
    
    pd.DataFrame({
        'Metric_Name': ['Kendall_W', 'Fleiss_Kappa'],
        'Calculated_Value': [round(kendall_w, 4), round(fleiss_kappa, 4)],
        'Validation_Status': ['Passed' if kendall_w >= 0.7 else 'Failed', 'Substantial' if fleiss_kappa > 0.61 else 'Check']
    }).to_csv(report_path, index=False)
    print(f"\n💾 تم حفظ تقرير جولة دلفي بنجاح في: {report_path}")

if __name__ == "__main__":
    calculate_delphi_agreement_metrics()
