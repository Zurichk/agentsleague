"""
Dashboard Flask Simplificado - AEP CertMaster

Punto de entrada principal para ejecutar el dashboard Flask.
"""

from .app import socketio, app

if __name__ == '__main__':
    # Inicializar agentes (ya se hace en app.py)
    socketio.run(app, host='0.0.0.0', port=5033, debug=True)
