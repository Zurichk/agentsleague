"""
Herramienta de generación de calendarios para AEP CertMaster.

Crea archivos de calendario (.ics) con planes de estudio
y recordatorios para los estudiantes.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from src.utils.logger import setup_logger

logger = setup_logger("tools.calendar")


class CalendarTool:
    """
    Herramienta para generar archivos de calendario (.ics)
    con planes de estudio y recordatorios.
    """

    def __init__(self):
        """Inicializa la herramienta de calendario."""
        logger.info("📅 CalendarTool inicializado")

    def generate_study_plan_calendar(
        self,
        plan_data: Dict[str, any],
        student_email: str,
        output_path: Optional[str] = None
    ) -> str:
        """
        Genera un archivo .ics con el plan de estudio completo.

        Args:
            plan_data: Datos del plan de estudio
            student_email: Email del estudiante para notificaciones
            output_path: Ruta donde guardar el archivo (opcional)

        Returns:
            Ruta al archivo .ics generado
        """
        logger.info(
            f"📅 Generando calendario para plan de estudio del estudiante: {student_email}")

        if output_path is None:
            # Crear directorio de salida
            output_dir = Path(__file__).parent.parent.parent / \
                "data" / "calendars"
            output_dir.mkdir(parents=True, exist_ok=True)
            plan_id = plan_data.get("plan_id", "unknown")
            output_path = str(output_dir / f"study_plan_{plan_id}.ics")

        # Generar contenido ICS
        ics_content = self._generate_ics_content(plan_data, student_email)

        # Guardar archivo
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(ics_content)

        logger.info(f"✅ Calendario generado: {output_path}")
        return output_path

    def generate_reminder_calendar(
        self,
        reminders: List[Dict[str, any]],
        student_email: str,
        output_path: Optional[str] = None
    ) -> str:
        """
        Genera un archivo .ics con recordatorios.

        Args:
            reminders: Lista de recordatorios
            student_email: Email del estudiante
            output_path: Ruta donde guardar el archivo

        Returns:
            Ruta al archivo .ics generado
        """
        logger.info(
            f"📅 Generando calendario de recordatorios para: {student_email}")

        if output_path is None:
            output_dir = Path(__file__).parent.parent.parent / \
                "data" / "calendars"
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(output_dir / f"reminders_{timestamp}.ics")

        # Generar contenido ICS para recordatorios
        ics_content = self._generate_reminders_ics_content(
            reminders, student_email)

        # Guardar archivo
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(ics_content)

        logger.info(f"✅ Calendario de recordatorios generado: {output_path}")
        return output_path

    def _generate_ics_content(self, plan_data: Dict[str, any], student_email: str) -> str:
        """Genera el contenido ICS para un plan de estudio."""

        # Encabezado ICS
        ics_lines = [
            "BEGIN:VCALENDAR",
            "VERSION:2.0",
            "PRODID:-//AEP CertMaster//Study Plan Calendar//EN",
            "CALSCALE:GREGORIAN",
            "METHOD:PUBLISH"
        ]

        plan_id = plan_data.get("plan_id", "unknown")
        plan_title = plan_data.get("title", "Plan de Estudio AEP CertMaster")

        # Evento principal del plan
        start_date = plan_data.get("start_date", datetime.now())
        if isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date)

        end_date = plan_data.get("end_date", start_date + timedelta(days=30))
        if isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date)

        event_lines = [
            "BEGIN:VEVENT",
            f"UID:{plan_id}@aep-certmaster",
            f"DTSTAMP:{datetime.now().strftime('%Y%m%dT%H%M%SZ')}",
            f"DTSTART:{start_date.strftime('%Y%m%dT%H%M%SZ')}",
            f"DTEND:{end_date.strftime('%Y%m%dT%H%M%SZ')}",
            f"SUMMARY:{plan_title}",
            f"DESCRIPTION:Plan de estudio personalizado generado por AEP CertMaster",
            f"ORGANIZER:mailto:certmaster@aep.com",
            f"ATTENDEE:mailto:{student_email}",
            "STATUS:CONFIRMED",
            "CLASS:PUBLIC",
            "END:VEVENT"
        ]

        ics_lines.extend(event_lines)

        # Agregar sesiones individuales si están disponibles
        sessions = plan_data.get("sessions", [])
        for session in sessions:
            session_event = self._generate_session_event(
                session, student_email)
            if session_event:
                ics_lines.extend(session_event)

        # Cerrar calendario
        ics_lines.append("END:VCALENDAR")

        return "\r\n".join(ics_lines)

    def _generate_session_event(self, session: Dict[str, any], student_email: str) -> List[str]:
        """Genera un evento ICS para una sesión de estudio."""

        try:
            session_date = session.get("date")
            if isinstance(session_date, str):
                session_date = datetime.fromisoformat(session_date)

            if not session_date:
                return []

            session_id = session.get("session_id", str(uuid.uuid4()))
            title = session.get("title", "Sesión de Estudio")
            duration_hours = session.get("duration_hours", 1.0)

            end_time = session_date + timedelta(hours=duration_hours)

            return [
                "BEGIN:VEVENT",
                f"UID:{session_id}@aep-certmaster",
                f"DTSTAMP:{datetime.now().strftime('%Y%m%dT%H%M%SZ')}",
                f"DTSTART:{session_date.strftime('%Y%m%dT%H%M%SZ')}",
                f"DTEND:{end_time.strftime('%Y%m%dT%H%M%SZ')}",
                f"SUMMARY:{title}",
                f"DESCRIPTION:Sesión de estudio: {session.get('description', '')}",
                f"ORGANIZER:mailto:certmaster@aep.com",
                f"ATTENDEE:mailto:{student_email}",
                "STATUS:CONFIRMED",
                "CLASS:PUBLIC",
                "BEGIN:VALARM",
                "TRIGGER:-PT15M",  # 15 minutos antes
                "ACTION:DISPLAY",
                f"DESCRIPTION:Recordatorio: {title}",
                "END:VALARM",
                "END:VEVENT"
            ]

        except Exception as e:
            logger.warning(f"⚠️ Error generando evento para sesión: {e}")
            return []

    def _generate_reminders_ics_content(self, reminders: List[Dict[str, any]], student_email: str) -> str:
        """Genera contenido ICS para recordatorios."""

        ics_lines = [
            "BEGIN:VCALENDAR",
            "VERSION:2.0",
            "PRODID:-//AEP CertMaster//Reminders Calendar//EN",
            "CALSCALE:GREGORIAN",
            "METHOD:PUBLISH"
        ]

        for reminder in reminders:
            reminder_event = self._generate_reminder_event(
                reminder, student_email)
            if reminder_event:
                ics_lines.extend(reminder_event)

        ics_lines.append("END:VCALENDAR")

        return "\r\n".join(ics_lines)

    def _generate_reminder_event(self, reminder: Dict[str, any], student_email: str) -> List[str]:
        """Genera un evento ICS para un recordatorio."""

        try:
            reminder_time = reminder.get("datetime")
            if isinstance(reminder_time, str):
                reminder_time = datetime.fromisoformat(reminder_time)

            if not reminder_time:
                return []

            reminder_id = reminder.get("id", str(uuid.uuid4()))
            title = reminder.get("title", "Recordatorio AEP CertMaster")
            message = reminder.get("message", "")

            # Recordatorios son eventos de 15 minutos
            end_time = reminder_time + timedelta(minutes=15)

            return [
                "BEGIN:VEVENT",
                f"UID:{reminder_id}@aep-certmaster",
                f"DTSTAMP:{datetime.now().strftime('%Y%m%dT%H%M%SZ')}",
                f"DTSTART:{reminder_time.strftime('%Y%m%dT%H%M%SZ')}",
                f"DTEND:{end_time.strftime('%Y%m%dT%H%M%SZ')}",
                f"SUMMARY:{title}",
                f"DESCRIPTION:{message}",
                f"ORGANIZER:mailto:certmaster@aep.com",
                f"ATTENDEE:mailto:{student_email}",
                "STATUS:CONFIRMED",
                "CLASS:PUBLIC",
                "BEGIN:VALARM",
                "TRIGGER:-PT5M",  # 5 minutos antes
                "ACTION:DISPLAY",
                f"DESCRIPTION:{message}",
                "END:VALARM",
                "END:VEVENT"
            ]

        except Exception as e:
            logger.warning(f"⚠️ Error generando evento para recordatorio: {e}")
            return []


# Instancia global
calendar_tool = CalendarTool()
