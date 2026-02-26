#!/usr/bin/env python3
import sys
import os
from pathlib import Path

# Agregar el directorio src al path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))


def main():
    """Función principal para ejecutar el dashboard."""
    try:
        print("🚀 Iniciando Dashboard Flask Simplificado - AEP CertMaster")
        print("📍 URL: http://localhost:5033")
        print("⏹️  Presiona Ctrl+C para detener")
        print("-" * 60)

        # Importar y ejecutar el dashboard
        from dashboard.app import socketio, app

        # El servidor se ejecutará hasta que se presione Ctrl+C
        socketio.run(app, host='0.0.0.0', port=5033,
                     debug=False, use_reloader=False)

    except KeyboardInterrupt:
        print("\n👋 Dashboard detenido por el usuario")
    except Exception as e:
        print(f"❌ Error al iniciar el dashboard: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
