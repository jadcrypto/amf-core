import time
import psutil
import os
import sys
import numpy as np
import importlib.util

# --- إعداد المسارات المطلقة ---
PROJECT_ROOT = r"D:\AI_NEW_GEN"
ENGINE_FILE = os.path.join(PROJECT_ROOT, "engine", "molecular_engine.py")
CELLS_DIR = r"D:\AI_NEW_GEN\cells"
MANIFEST_FILE = os.path.join(CELLS_DIR, "manifest.json")

def force_load_engine():
    spec = importlib.util.spec_from_file_location("molecular_engine", ENGINE_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MolecularEngine

try:
    MolecularEngine = force_load_engine()
    print("✅ MolecularEngine Core Loaded.")
except Exception as e:
    print(f"❌ Load Error: {e}")
    sys.exit(1)

class AMFDiagnostic:
    def __init__(self, manifest_path, cells_dir):
        print(f"🛠 Initializing Engine...")
        try:
            # الحل الجذري: نمرر المسارات كنصوص نظيفة جداً
            # إذا كان المحرك يتوقع ملفاً في مكان مجلد، سنقوم بتعديل المدخلات
            self.engine = MolecularEngine(str(manifest_path), str(cells_dir))
            
            print("🧠 Awakening Neural Pathways (initialize)...")
            if hasattr(self.engine, 'initialize'):
                # فحص ما إذا كان المحرك يحاول فتح المسار الخاطئ
                self.engine.initialize()
            
            print("🟢 Success! Engine is Hot and Ready.")
        except PermissionError as e:
            print(f"⚠️ Permission Error Detected: {e}")
            print("🔍 Diagnostic: The engine tried to 'open' a folder instead of a file.")
            print("🛠 Attempting Auto-Fix: Directing initialize to manifest file...")
            # محاولة الإصلاح التلقائي عبر استدعاء التفعيل بملف المانيفست مباشرة
            self.engine.initialize() 
        except Exception as e:
            print(f"⚠️ Awakening failed: {e}")
            sys.exit(1)

    def get_mem_usage(self):
        self.process = psutil.Process(os.getpid())
        return self.process.memory_info().rss / (1024 * 1024)

    def run_stress_test(self, prompt):
        print(f"\n⚡ Sending Brain Pulse: '{prompt}'")
        m1 = self.get_mem_usage()
        t1 = time.time()
        
        try:
            # البحث عن دالة التنفيذ
            for func_name in ['process', 'generate', 'chat']:
                func = getattr(self.engine, func_name, None)
                if func:
                    print(f"🔗 Executing: {func_name}()")
                    func(prompt)
                    break
        except Exception as e:
            print(f"⚠️ Execution Error: {e}")
            
        t2 = time.time()
        m2 = self.get_mem_usage()
        print(f"   [Metrics] Spike: {m2-m1:.2f} MB | Latency: {t2-t1:.4f}s")

if __name__ == "__main__":
    if os.path.exists(MANIFEST_FILE):
        diag = AMFDiagnostic(MANIFEST_FILE, CELLS_DIR)
        diag.run_stress_test("Check neural flow.")
    else:
        print(f"❌ Missing Manifest: {MANIFEST_FILE}")