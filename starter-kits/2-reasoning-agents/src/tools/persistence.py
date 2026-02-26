"""
Herramienta de persistencia para AEP CertMaster.

Implementa persistencia de datos del estudiante usando SQLite local
con interfaz preparada para migración a Azure Cosmos DB.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logger import setup_logger

logger = setup_logger("tools.persistence")


class PersistenceTool:
    """
    Herramienta de persistencia para almacenar y recuperar datos del estudiante.

    Actualmente usa SQLite local, pero la interfaz está preparada para
    migración a Azure Cosmos DB.
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Inicializa la herramienta de persistencia.

        Args:
            db_path: Ruta al archivo de base de datos SQLite.
                    Si no se especifica, usa 'data/certmaster.db'
        """
        if db_path is None:
            # Crear directorio data si no existe
            data_dir = Path(__file__).parent.parent.parent / "data"
            data_dir.mkdir(exist_ok=True)
            db_path = str(data_dir / "certmaster.db")

        self.db_path = db_path
        self._init_database()

        logger.info(f"🗄️ PersistenceTool inicializado: {self.db_path}")

    def _init_database(self) -> None:
        """Inicializa las tablas de la base de datos."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Tabla para perfiles de estudiantes
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS student_profiles (
                    student_id TEXT PRIMARY KEY,
                    profile_data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Tabla para planes de estudio
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS study_plans (
                    plan_id TEXT PRIMARY KEY,
                    student_id TEXT NOT NULL,
                    plan_name TEXT,
                    certification TEXT,
                    plan_data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (student_id) REFERENCES student_profiles (student_id)
                )
            """)

            # Tabla para progreso de estudio
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS study_progress (
                    progress_id TEXT PRIMARY KEY,
                    student_id TEXT NOT NULL,
                    plan_id TEXT,
                    module_id TEXT NOT NULL,
                    completed BOOLEAN DEFAULT FALSE,
                    completion_date TIMESTAMP,
                    time_spent_minutes INTEGER DEFAULT 0,
                    score REAL,
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (student_id) REFERENCES student_profiles (student_id),
                    FOREIGN KEY (plan_id) REFERENCES study_plans (plan_id)
                )
            """)

            # Tabla para evaluaciones
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS assessments (
                    assessment_id TEXT PRIMARY KEY,
                    student_id TEXT NOT NULL,
                    plan_id TEXT,
                    assessment_data TEXT NOT NULL,
                    score REAL,
                    passed BOOLEAN,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (student_id) REFERENCES student_profiles (student_id),
                    FOREIGN KEY (plan_id) REFERENCES study_plans (plan_id)
                )
            """)

            # Tabla para historial de conversaciones
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversation_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT NOT NULL,
                    session_id TEXT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    agent_name TEXT,
                    phase TEXT,
                    prompt_tokens INTEGER DEFAULT 0,
                    completion_tokens INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (student_id) REFERENCES student_profiles (student_id)
                )
            """)

            # Tabla para métricas de sesión
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS session_metrics (
                    session_id TEXT PRIMARY KEY,
                    student_id TEXT NOT NULL,
                    total_interactions INTEGER DEFAULT 0,
                    prompt_tokens INTEGER DEFAULT 0,
                    completion_tokens INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    agents_used TEXT,
                    final_phase TEXT,
                    execution_time_seconds REAL DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (student_id) REFERENCES student_profiles (student_id)
                )
            """)

            # Tabla para registro de emails (recordatorios y notificaciones)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS email_registry (
                    student_id TEXT PRIMARY KEY,
                    email TEXT NOT NULL,
                    name TEXT,
                    notify_reminders BOOLEAN DEFAULT TRUE,
                    notify_progress BOOLEAN DEFAULT TRUE,
                    notify_assessments BOOLEAN DEFAULT TRUE,
                    active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (student_id) REFERENCES student_profiles (student_id)
                )
            """)

            # Tabla de usuarios para autenticación
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    name TEXT,
                    role TEXT DEFAULT 'student',
                    student_id TEXT,
                    active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP
                )
            """)

            conn.commit()

            self._ensure_study_plan_columns(cursor)

            conn.commit()

        logger.info("📊 Base de datos inicializada correctamente")

    @staticmethod
    def _ensure_study_plan_columns(cursor: sqlite3.Cursor) -> None:
        """Asegura columnas nuevas en study_plans para migraciones."""
        cursor.execute("PRAGMA table_info(study_plans)")
        existing_columns = {row[1] for row in cursor.fetchall()}

        if 'plan_name' not in existing_columns:
            cursor.execute("ALTER TABLE study_plans ADD COLUMN plan_name TEXT")

        if 'certification' not in existing_columns:
            cursor.execute(
                "ALTER TABLE study_plans ADD COLUMN certification TEXT"
            )

    def save_student_profile(self, student_id: str, profile_data: Dict[str, Any]) -> bool:
        """
        Guarda el perfil de un estudiante.

        Args:
            student_id: ID único del estudiante
            profile_data: Datos del perfil como diccionario

        Returns:
            True si se guardó correctamente, False en caso contrario
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Convertir datos a JSON
                profile_json = json.dumps(profile_data, default=str)

                # Insertar o actualizar
                cursor.execute("""
                    INSERT INTO student_profiles (student_id, profile_data, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(student_id) DO UPDATE SET
                        profile_data = excluded.profile_data,
                        updated_at = CURRENT_TIMESTAMP
                """, (student_id, profile_json))

                conn.commit()

            logger.info(f"✅ Perfil guardado para estudiante: {student_id}")
            return True

        except Exception as e:
            logger.error(
                f"❌ Error guardando perfil de estudiante {student_id}: {e}")
            return False

    def load_student_profile(self, student_id: str) -> Optional[Dict[str, Any]]:
        """
        Carga el perfil de un estudiante.

        Args:
            student_id: ID único del estudiante

        Returns:
            Datos del perfil como diccionario, o None si no existe
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    "SELECT profile_data FROM student_profiles WHERE student_id = ?",
                    (student_id,)
                )

                row = cursor.fetchone()
                if row:
                    return json.loads(row[0])

                return None

        except Exception as e:
            logger.error(
                f"❌ Error cargando perfil de estudiante {student_id}: {e}")
            return None

    def save_study_plan(self, plan_id: str, student_id: str, plan_data: Dict[str, Any]) -> bool:
        """
        Guarda un plan de estudio.

        Args:
            plan_id: ID único del plan
            student_id: ID del estudiante
            plan_data: Datos del plan como diccionario

        Returns:
            True si se guardó correctamente
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                plan_json = json.dumps(plan_data, default=str)
                plan_name = (
                    plan_data.get("plan_name")
                    or plan_data.get("certification")
                    or f"Plan {plan_id[:8]}"
                )
                certification = plan_data.get("certification", "")

                cursor.execute("""
                    INSERT INTO study_plans (
                        plan_id, student_id, plan_name, certification,
                        plan_data, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(plan_id) DO UPDATE SET
                        plan_name = excluded.plan_name,
                        certification = excluded.certification,
                        plan_data = excluded.plan_data,
                        updated_at = CURRENT_TIMESTAMP
                """, (
                    plan_id,
                    student_id,
                    plan_name,
                    certification,
                    plan_json,
                ))

                conn.commit()

            logger.info(
                f"✅ Plan de estudio guardado: {plan_id} para estudiante: {student_id}")
            return True

        except Exception as e:
            logger.error(f"❌ Error guardando plan de estudio {plan_id}: {e}")
            return False

    def load_study_plan(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """
        Carga un plan de estudio.

        Args:
            plan_id: ID único del plan

        Returns:
            Datos del plan como diccionario, o None si no existe
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    "SELECT plan_data, plan_name, certification "
                    "FROM study_plans WHERE plan_id = ?",
                    (plan_id,)
                )

                row = cursor.fetchone()
                if row:
                    data = json.loads(row[0])
                    if row[1] and not data.get('plan_name'):
                        data['plan_name'] = row[1]
                    if row[2] and not data.get('certification'):
                        data['certification'] = row[2]
                    return data

                return None

        except Exception as e:
            logger.error(f"❌ Error cargando plan de estudio {plan_id}: {e}")
            return None

    def list_study_plans(self, student_id: str) -> List[Dict[str, Any]]:
        """Lista planes de estudio de un estudiante (más recientes primero)."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT plan_id, student_id, plan_name, certification, "
                    "created_at, updated_at "
                    "FROM study_plans WHERE student_id = ? "
                    "ORDER BY updated_at DESC",
                    (student_id,)
                )
                rows = cursor.fetchall()

            return [
                {
                    'plan_id': row[0],
                    'student_id': row[1],
                    'plan_name': row[2] or row[3] or row[0],
                    'certification': row[3] or '',
                    'created_at': row[4],
                    'updated_at': row[5],
                }
                for row in rows
            ]
        except Exception as e:
            logger.error(
                f"❌ Error listando planes para estudiante {student_id}: {e}"
            )
            return []

    def update_study_plan_name(self, plan_id: str, plan_name: str) -> bool:
        """Actualiza el nombre visible de un plan de estudio."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE study_plans "
                    "SET plan_name = ?, updated_at = CURRENT_TIMESTAMP "
                    "WHERE plan_id = ?",
                    (plan_name, plan_id),
                )
                conn.commit()
                if cursor.rowcount == 0:
                    return False

                cursor.execute(
                    "SELECT plan_data FROM study_plans WHERE plan_id = ?",
                    (plan_id,)
                )
                row = cursor.fetchone()
                if not row:
                    return False

                plan_data = json.loads(row[0])
                plan_data['plan_name'] = plan_name
                cursor.execute(
                    "UPDATE study_plans "
                    "SET plan_data = ?, updated_at = CURRENT_TIMESTAMP "
                    "WHERE plan_id = ?",
                    (json.dumps(plan_data, default=str), plan_id),
                )
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"❌ Error actualizando nombre de plan {plan_id}: {e}")
            return False

    def set_study_session_completed(
        self,
        plan_id: str,
        session_id: str,
        completed: bool,
    ) -> bool:
        """Marca una sesión del plan como completada/no completada."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT plan_data FROM study_plans WHERE plan_id = ?",
                    (plan_id,)
                )
                row = cursor.fetchone()
                if not row:
                    return False

                plan_data = json.loads(row[0])
                sessions = plan_data.get('sessions', [])
                updated = False
                for session in sessions:
                    if session.get('session_id') == session_id:
                        session['completed'] = bool(completed)
                        updated = True
                        break

                if not updated:
                    return False

                plan_data['sessions'] = sessions
                cursor.execute(
                    "UPDATE study_plans "
                    "SET plan_data = ?, updated_at = CURRENT_TIMESTAMP "
                    "WHERE plan_id = ?",
                    (json.dumps(plan_data, default=str), plan_id),
                )
                conn.commit()

            return True
        except Exception as e:
            logger.error(
                f"❌ Error actualizando sesión {session_id} de plan {plan_id}: {e}"
            )
            return False

    def update_study_progress(
        self,
        student_id: str,
        module_id: str,
        completed: bool = False,
        time_spent_minutes: int = 0,
        score: Optional[float] = None,
        notes: Optional[str] = None,
        plan_id: Optional[str] = None
    ) -> bool:
        """
        Actualiza el progreso de estudio de un módulo.

        Args:
            student_id: ID del estudiante
            module_id: ID del módulo
            completed: Si el módulo está completado
            time_spent_minutes: Tiempo dedicado en minutos
            score: Puntuación obtenida (0-100)
            notes: Notas adicionales
            plan_id: ID del plan de estudio (opcional)

        Returns:
            True si se actualizó correctamente
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Generar ID único para el progreso
                progress_id = f"{student_id}_{module_id}"

                completion_date = "CURRENT_TIMESTAMP" if completed else None

                cursor.execute("""
                    INSERT INTO study_progress (
                        progress_id, student_id, plan_id, module_id,
                        completed, completion_date, time_spent_minutes, score, notes, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(progress_id) DO UPDATE SET
                        completed = excluded.completed,
                        completion_date = excluded.completion_date,
                        time_spent_minutes = excluded.time_spent_minutes,
                        score = excluded.score,
                        notes = excluded.notes,
                        updated_at = CURRENT_TIMESTAMP
                """, (progress_id, student_id, plan_id, module_id, completed,
                      completion_date, time_spent_minutes, score, notes))

                conn.commit()

            logger.info(
                f"✅ Progreso actualizado: {module_id} para estudiante: {student_id}")
            return True

        except Exception as e:
            logger.error(f"❌ Error actualizando progreso {module_id}: {e}")
            return False

    def get_study_progress(self, student_id: str, plan_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Obtiene el progreso de estudio de un estudiante.

        Args:
            student_id: ID del estudiante
            plan_id: ID del plan (opcional, para filtrar)

        Returns:
            Lista de registros de progreso
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                if plan_id:
                    cursor.execute(
                        "SELECT * FROM study_progress WHERE student_id = ? AND plan_id = ?",
                        (student_id, plan_id)
                    )
                else:
                    cursor.execute(
                        "SELECT * FROM study_progress WHERE student_id = ?",
                        (student_id,)
                    )

                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()

                return [dict(zip(columns, row)) for row in rows]

        except Exception as e:
            logger.error(
                f"❌ Error obteniendo progreso para estudiante {student_id}: {e}")
            return []

    def save_assessment(
        self,
        assessment_id: str,
        student_id: str,
        assessment_data: Dict[str, Any],
        score: Optional[float] = None,
        passed: Optional[bool] = None,
        plan_id: Optional[str] = None
    ) -> bool:
        """
        Guarda los resultados de una evaluación.

        Args:
            assessment_id: ID único de la evaluación
            student_id: ID del estudiante
            assessment_data: Datos de la evaluación
            score: Puntuación obtenida
            passed: Si pasó la evaluación
            plan_id: ID del plan de estudio

        Returns:
            True si se guardó correctamente
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                assessment_json = json.dumps(assessment_data, default=str)

                cursor.execute("""
                    INSERT INTO assessments (assessment_id, student_id, plan_id, assessment_data, score, passed)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (assessment_id, student_id, plan_id, assessment_json, score, passed))

                conn.commit()

            logger.info(
                f"✅ Evaluación guardada: {assessment_id} para estudiante: {student_id}")
            return True

        except Exception as e:
            logger.error(f"❌ Error guardando evaluación {assessment_id}: {e}")
            return False

    def get_student_stats(self, student_id: str) -> Dict[str, Any]:
        """
        Obtiene estadísticas generales del estudiante.

        Args:
            student_id: ID del estudiante

        Returns:
            Diccionario con estadísticas
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Contar módulos completados
                cursor.execute(
                    "SELECT COUNT(*) FROM study_progress WHERE student_id = ? AND completed = 1",
                    (student_id,)
                )
                completed_modules = cursor.fetchone()[0]

                # Tiempo total estudiado
                cursor.execute(
                    "SELECT SUM(time_spent_minutes) FROM study_progress WHERE student_id = ?",
                    (student_id,)
                )
                total_time = cursor.fetchone()[0] or 0

                # Evaluaciones tomadas
                cursor.execute(
                    "SELECT COUNT(*) FROM assessments WHERE student_id = ?",
                    (student_id,)
                )
                total_assessments = cursor.fetchone()[0]

                # Puntuación promedio
                cursor.execute(
                    "SELECT AVG(score) FROM assessments WHERE student_id = ? AND score IS NOT NULL",
                    (student_id,)
                )
                avg_score = cursor.fetchone()[0]

                return {
                    "student_id": student_id,
                    "completed_modules": completed_modules,
                    "total_study_time_minutes": total_time,
                    "total_assessments": total_assessments,
                    "average_score": avg_score,
                    # Evitar división por cero
                    "completion_rate": completed_modules / max(total_assessments, 1)
                }

        except Exception as e:
            logger.error(
                f"❌ Error obteniendo estadísticas para estudiante {student_id}: {e}")
            return {}

    # -------------------------------------------------------------------------
    # Historial de conversaciones
    # -------------------------------------------------------------------------

    def save_conversation_message(
        self,
        student_id: str,
        role: str,
        content: str,
        session_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        phase: Optional[str] = None,
        tokens: Optional[Dict[str, int]] = None
    ) -> bool:
        """
        Guarda un mensaje del historial de conversación.

        Args:
            student_id: ID del estudiante
            role: 'user' o 'assistant'
            content: Texto del mensaje
            session_id: ID de sesión (opcional)
            agent_name: Nombre del agente que respondió (opcional)
            phase: Fase del workflow en ese momento (opcional)
            tokens: Dict con prompt_tokens, completion_tokens, total_tokens (opcional)

        Returns:
            True si se guardó correctamente
        """
        try:
            t = tokens or {}
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO conversation_history
                        (student_id, session_id, role, content, agent_name, phase,
                         prompt_tokens, completion_tokens, total_tokens)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    student_id, session_id, role, content, agent_name, phase,
                    t.get('prompt_tokens', 0),
                    t.get('completion_tokens', 0),
                    t.get('total_tokens', 0)
                ))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"❌ Error guardando mensaje de conversación: {e}")
            return False

    def get_conversation_history(
        self,
        student_id: str,
        limit: int = 50,
        session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Recupera el historial de conversación de un estudiante.

        Args:
            student_id: ID del estudiante
            limit: Número máximo de mensajes a retornar
            session_id: Filtrar por sesión específica (opcional)

        Returns:
            Lista de mensajes ordenados por fecha ascendente
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                if session_id:
                    cursor.execute("""
                        SELECT id, role, content, agent_name, phase,
                               prompt_tokens, completion_tokens, total_tokens, created_at
                        FROM conversation_history
                        WHERE student_id = ? AND session_id = ?
                        ORDER BY created_at ASC
                        LIMIT ?
                    """, (student_id, session_id, limit))
                else:
                    cursor.execute("""
                        SELECT id, role, content, agent_name, phase,
                               prompt_tokens, completion_tokens, total_tokens, created_at
                        FROM conversation_history
                        WHERE student_id = ?
                        ORDER BY created_at DESC
                        LIMIT ?
                    """, (student_id, limit))
                columns = [d[0] for d in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"❌ Error recuperando historial de conversación: {e}")
            return []

    # -------------------------------------------------------------------------
    # Métricas de sesión
    # -------------------------------------------------------------------------

    def save_session_metrics(
        self,
        session_id: str,
        student_id: str,
        tokens: Dict[str, int],
        total_interactions: int = 1,
        agents_used: Optional[List[str]] = None,
        final_phase: Optional[str] = None,
        execution_time_seconds: float = 0.0
    ) -> bool:
        """
        Guarda o acumula las métricas de una sesión de conversación.

        Args:
            session_id: ID único de sesión
            student_id: ID del estudiante
            tokens: Dict con prompt_tokens, completion_tokens, total_tokens
            total_interactions: Número de interacciones en esta llamada
            agents_used: Lista de nombres de agentes usados
            final_phase: Fase final del workflow
            execution_time_seconds: Tiempo de ejecución

        Returns:
            True si se guardó correctamente
        """
        try:
            agents_json = json.dumps(agents_used or [])
            t = tokens or {}
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO session_metrics
                        (session_id, student_id, total_interactions, prompt_tokens,
                         completion_tokens, total_tokens, agents_used, final_phase,
                         execution_time_seconds)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        total_interactions = total_interactions + excluded.total_interactions,
                        prompt_tokens = prompt_tokens + excluded.prompt_tokens,
                        completion_tokens = completion_tokens + excluded.completion_tokens,
                        total_tokens = total_tokens + excluded.total_tokens,
                        agents_used = excluded.agents_used,
                        final_phase = excluded.final_phase,
                        execution_time_seconds = execution_time_seconds + excluded.execution_time_seconds,
                        updated_at = CURRENT_TIMESTAMP
                """, (
                    session_id, student_id, total_interactions,
                    t.get('prompt_tokens', 0),
                    t.get('completion_tokens', 0),
                    t.get('total_tokens', 0),
                    agents_json, final_phase, execution_time_seconds
                ))
                conn.commit()
            return True
        except Exception as e:
            logger.error(
                f"❌ Error guardando métricas de sesión {session_id}: {e}")
            return False

    def get_student_session_stats(self, student_id: str) -> Dict[str, Any]:
        """
        Obtiene estadísticas agregadas de sesiones de un estudiante.

        Returns:
            Dict con total_sessions, total_tokens, total_interactions, avg_tokens_per_session
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT COUNT(*) as sessions,
                           SUM(total_tokens) as total_tokens,
                           SUM(total_interactions) as total_interactions,
                           AVG(total_tokens) as avg_tokens
                    FROM session_metrics
                    WHERE student_id = ?
                """, (student_id,))
                row = cursor.fetchone()
                if row:
                    return {
                        'total_sessions': row[0] or 0,
                        'total_tokens': row[1] or 0,
                        'total_interactions': row[2] or 0,
                        'avg_tokens_per_session': round(row[3] or 0, 1)
                    }
                return {}
        except Exception as e:
            logger.error(f"❌ Error obteniendo estadísticas de sesión: {e}")
            return {}

    # -------------------------------------------------------------------------
    # Registro de emails
    # -------------------------------------------------------------------------

    def save_email(
        self,
        student_id: str,
        email: str,
        name: Optional[str] = None,
        notify_reminders: bool = True,
        notify_progress: bool = True,
        notify_assessments: bool = True
    ) -> bool:
        """
        Registra o actualiza el email de un estudiante.

        Args:
            student_id: ID del estudiante
            email: Dirección de email
            name: Nombre del estudiante (opcional)
            notify_reminders: Activar recordatorios de estudio
            notify_progress: Activar notificaciones de progreso
            notify_assessments: Activar notificaciones de evaluaciones

        Returns:
            True si se guardó correctamente
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO email_registry
                        (student_id, email, name, notify_reminders, notify_progress,
                         notify_assessments, active)
                    VALUES (?, ?, ?, ?, ?, ?, TRUE)
                    ON CONFLICT(student_id) DO UPDATE SET
                        email = excluded.email,
                        name = COALESCE(excluded.name, email_registry.name),
                        notify_reminders = excluded.notify_reminders,
                        notify_progress = excluded.notify_progress,
                        notify_assessments = excluded.notify_assessments,
                        active = TRUE,
                        updated_at = CURRENT_TIMESTAMP
                """, (student_id, email, name, notify_reminders,
                      notify_progress, notify_assessments))
                conn.commit()
            logger.info(
                f"✅ Email registrado para estudiante {student_id}: {email}")
            return True
        except Exception as e:
            logger.error(f"❌ Error guardando email {student_id}: {e}")
            return False

    def get_email(self, student_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene el registro de email de un estudiante.

        Returns:
            Dict con email, name, preferencias de notificación, o None si no existe
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT student_id, email, name, notify_reminders, notify_progress,
                           notify_assessments, active, created_at
                    FROM email_registry
                    WHERE student_id = ? AND active = TRUE
                """, (student_id,))
                row = cursor.fetchone()
                if row:
                    columns = [d[0] for d in cursor.description]
                    return dict(zip(columns, row))
                return None
        except Exception as e:
            logger.error(f"❌ Error obteniendo email para {student_id}: {e}")
            return None

    def get_all_active_emails(self, notification_type: str = 'reminders') -> List[Dict[str, Any]]:
        """
        Obtiene todos los emails activos con un tipo de notificación habilitado.

        Args:
            notification_type: 'reminders', 'progress', o 'assessments'

        Returns:
            Lista de registros de email activos
        """
        column_map = {
            'reminders': 'notify_reminders',
            'progress': 'notify_progress',
            'assessments': 'notify_assessments'
        }
        col = column_map.get(notification_type, 'notify_reminders')
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(f"""
                    SELECT student_id, email, name FROM email_registry
                    WHERE active = TRUE AND {col} = TRUE
                """)
                columns = [d[0] for d in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"❌ Error obteniendo lista de emails: {e}")
            return []

    # -------------------------------------------------------------------------
    # Autenticación de usuarios
    # -------------------------------------------------------------------------

    @staticmethod
    def _hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
        """Genera hash seguro de contraseña con salt."""
        if salt is None:
            salt = secrets.token_hex(32)
        pwd_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            iterations=260000
        ).hex()
        return pwd_hash, salt

    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        name: Optional[str] = None,
        role: str = 'student'
    ) -> Optional[str]:
        """
        Crea un nuevo usuario en la base de datos.

        Args:
            username: Nombre de usuario único
            email: Email único
            password: Contraseña en texto plano (se hashea internamente)
            name: Nombre completo (opcional)
            role: Rol del usuario ('student' o 'admin')

        Returns:
            user_id si se creó correctamente, None si ya existe o hay error
        """
        try:
            user_id = str(uuid.uuid4())
            student_id = f"student_{username.lower().replace(' ', '_')}"
            pwd_hash, salt = self._hash_password(password)

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO users
                        (user_id, username, email, password_hash, salt, name, role, student_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (user_id, username.strip(), email.strip().lower(),
                      pwd_hash, salt, name, role, student_id))
                conn.commit()

            # Crear perfil de estudiante asociado automáticamente
            self.save_student_profile(student_id, {
                'student_id': student_id,
                'name': name or username,
                'email': email.strip().lower(),
                'username': username.strip()
            })

            logger.info(f"✅ Usuario creado: {username} ({user_id})")
            return user_id

        except sqlite3.IntegrityError:
            logger.warning(
                f"⚠️ Usuario o email ya existe: {username} / {email}")
            return None
        except Exception as e:
            logger.error(f"❌ Error creando usuario {username}: {e}")
            return None

    def authenticate_user(
        self,
        username_or_email: str,
        password: str
    ) -> Optional[Dict[str, Any]]:
        """
        Autentica un usuario por username o email y contraseña.

        Returns:
            Dict con datos del usuario si las credenciales son correctas, None si no.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT user_id, username, email, password_hash, salt,
                           name, role, student_id, active
                    FROM users
                    WHERE (username = ? OR email = ?) AND active = TRUE
                """, (username_or_email.strip(), username_or_email.strip().lower()))

                row = cursor.fetchone()
                if not row:
                    return None

                user_id, username, email, stored_hash, salt, name, role, student_id, active = row

                # Verificar contraseña
                computed_hash, _ = self._hash_password(password, salt)
                if not secrets.compare_digest(computed_hash, stored_hash):
                    return None

                # Actualizar último login
                cursor.execute(
                    "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE user_id = ?",
                    (user_id,)
                )
                conn.commit()

            logger.info(f"✅ Autenticación exitosa: {username}")
            return {
                'user_id': user_id,
                'username': username,
                'email': email,
                'name': name or username,
                'role': role,
                'student_id': student_id
            }

        except Exception as e:
            logger.error(f"❌ Error autenticando usuario: {e}")
            return None

    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene los datos de un usuario por su ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT user_id, username, email, name, role, student_id, created_at, last_login
                    FROM users
                    WHERE user_id = ? AND active = TRUE
                """, (user_id,))
                row = cursor.fetchone()
                if row:
                    columns = [d[0] for d in cursor.description]
                    return dict(zip(columns, row))
                return None
        except Exception as e:
            logger.error(f"❌ Error obteniendo usuario {user_id}: {e}")
            return None

    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Obtiene los datos de un usuario por su username."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT user_id, username, email, name, role, student_id, created_at, last_login
                    FROM users
                    WHERE (username = ? OR email = ?) AND active = TRUE
                """, (username.strip(), username.strip().lower()))
                row = cursor.fetchone()
                if row:
                    columns = [d[0] for d in cursor.description]
                    return dict(zip(columns, row))
                return None
        except Exception as e:
            logger.error(f"❌ Error obteniendo usuario {username}: {e}")
            return None

    def list_users(self, role: Optional[str] = None) -> List[Dict[str, Any]]:
        """Lista todos los usuarios activos, opcionalmente filtrados por rol."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                if role:
                    cursor.execute("""
                        SELECT user_id, username, email, name, role, student_id, created_at, last_login
                        FROM users WHERE active = TRUE AND role = ?
                        ORDER BY created_at DESC
                    """, (role,))
                else:
                    cursor.execute("""
                        SELECT user_id, username, email, name, role, student_id, created_at, last_login
                        FROM users WHERE active = TRUE
                        ORDER BY created_at DESC
                    """)
                columns = [d[0] for d in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"❌ Error listando usuarios: {e}")
            return []

    def change_password(self, user_id: str, new_password: str) -> bool:
        """Cambia la contraseña de un usuario."""
        try:
            pwd_hash, salt = self._hash_password(new_password)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE users SET password_hash = ?, salt = ? WHERE user_id = ?",
                    (pwd_hash, salt, user_id)
                )
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"❌ Error cambiando contraseña: {e}")
            return False


# Instancia global para uso en toda la aplicación
persistence_tool = PersistenceTool()
