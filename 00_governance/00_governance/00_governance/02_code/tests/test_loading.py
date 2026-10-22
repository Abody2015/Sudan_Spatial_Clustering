import os
import pytest

def test_repository_structure():
    """يتحقق آلياً من التزام المستودع بهيكل الحوكمة الصارم لمنع تشتت الملفات"""
    mandatory_dirs = [
        "00_governance",
        "01_data/raw",
        "01_data/interim",
        "01_data/processed",
        "02_code/src",
        "02_code/tests",
        "03_outputs/figures",
        "03_outputs/tables",
        "04_manuscript"
    ]
    for directory in mandatory_dirs:
        assert os.path.exists(directory), f"Critical Governance Error: Required directory '{directory}' is missing."

def test_governance_files_exist():
    """يتحقق من وجود وثائق الحوكمة والتسجيل المسبق لضمان نزاهة الاستدلال"""
    critical_files = [
        "00_governance/README.md",
        "00_governance/environment.yml",
        "00_governance/risk_register.md",
        "00_governance/data_dictionary.md",
        "00_governance/irb_application_memo.md",
        "00_governance/informed_consent_form.md"
    ]
    for file_path in critical_files:
        assert os.path.exists(file_path), f"Methodological Blocker: Missing critical compliance file '{file_path}'."
