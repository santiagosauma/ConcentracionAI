import subprocess
import sys
import os

def main():
    """Ejecutar el dashboard de Streamlit"""
    print("🚢 Iniciando Dashboard del Titanic...")
    print("📊 Cargando aplicación Streamlit...")
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "main.py",
            "--server.port", "8503",
            "--server.headless", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Dashboard cerrado por el usuario")
    except Exception as e:
        print(f"❌ Error al ejecutar el dashboard: {e}")

if __name__ == "__main__":
    main()