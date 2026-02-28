"""
Herramienta de envío de emails para AEP CertMaster.

Implementa envío real via SMTP (Gmail, Outlook, etc.) usando las variables
de entorno SMTP_HOST / SMTP_PORT / SMTP_USER / SMTP_PASSWORD.
Si USE_REAL_EMAIL=false (o las variables no están configuradas) guarda los
emails como JSON en data/emails/ (modo simulado).
"""

from __future__ import annotations

import asyncio
import json
import os
import smtplib
import ssl
import time
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Dict, List, Optional

from src.utils.logger import setup_logger

logger = setup_logger("tools.email")


class EmailTool:
    """
    Herramienta para envío de emails via SMTP.

    Lee la configuración de las variables de entorno:
      SMTP_HOST         — servidor SMTP (ej. smtp.gmail.com)
      SMTP_PORT         — puerto (587 STARTTLS / 465 SSL)
      SMTP_USER         — cuenta remitente
      SMTP_PASSWORD     — contraseña / contraseña de aplicación
      SMTP_FROM_NAME    — nombre visible del remitente
      USE_REAL_EMAIL    — 'true' para envío real, 'false' para simulado
    """

    def __init__(self, config: Optional[Dict[str, str]] = None):
        cfg = config or {}

        self.smtp_host = cfg.get("smtp_host") or os.getenv("SMTP_HOST", "")
        self.smtp_port = int(cfg.get("smtp_port")
                             or os.getenv("SMTP_PORT", "587"))
        self.smtp_user = cfg.get("smtp_user") or os.getenv("SMTP_USER", "")
        self.smtp_password = cfg.get(
            "smtp_password") or os.getenv("SMTP_PASSWORD", "")
        self.sender_name = cfg.get("smtp_from_name") or os.getenv(
            "SMTP_FROM_NAME", "AEP CertMaster")
        self.sender_email = self.smtp_user

        use_real_env = os.getenv("USE_REAL_EMAIL", "false").lower()
        self.simulate_sending = (
            use_real_env != "true" or not self.smtp_user or not self.smtp_password
        )

        self.email_log_dir = Path(
            __file__).parent.parent.parent / "data" / "emails"
        self.email_log_dir.mkdir(parents=True, exist_ok=True)

        if self.simulate_sending:
            if use_real_env == "true" and (not self.smtp_user or not self.smtp_password):
                logger.warning(
                    "[EMAIL] AVISO: USE_REAL_EMAIL=true pero SMTP_USER/SMTP_PASSWORD no configurados. "
                    "Usando modo simulado. Configura las variables en .env"
                )
            else:
                logger.info(
                    "[EMAIL] EmailTool inicializado (modo: simulado - USE_REAL_EMAIL=false)")
        else:
            logger.info(
                f"[EMAIL] EmailTool inicializado (modo: REAL | {self.smtp_user} -> {self.smtp_host}:{self.smtp_port})"
            )

    async def send_welcome_email(
        self,
        recipient_email: str,
        recipient_name: str,
        certification: str,
    ) -> bool:
        subject = f"¡Bienvenido a AEP CertMaster, {recipient_name}!"
        html = f"""
        <html><body>
            <h2>¡Bienvenido a AEP CertMaster!</h2>
            <p>Hola <strong>{recipient_name}</strong>,</p>
            <p>¡Felicitaciones por dar el primer paso hacia tu certificación
            <strong>{certification}</strong>!</p>
            <p>Estoy aquí para guiarte con:</p>
            <ul>
                <li>📚 Rutas de aprendizaje personalizadas</li>
                <li>📋 Planes de estudio adaptados a tu disponibilidad</li>
                <li>🎯 Evaluaciones inteligentes</li>
                <li>✅ Seguimiento de constancia y hábitos de estudio</li>
                <li>📧 Recordatorios y soporte continuo</li>
            </ul>
            <p>¡Vamos a lograrlo juntos!</p>
            <br><p>Saludos,<br><strong>AEP CertMaster</strong></p>
        </body></html>
        """
        return await self._send_email(recipient_email, subject, html)

    async def send_study_plan_email(
        self,
        recipient_email: str,
        recipient_name: str,
        plan_data: Dict,
        calendar_attachment: Optional[str] = None,
    ) -> bool:
        plan_title = plan_data.get("title", "Tu Plan de Estudio Personalizado")
        subject = f"📋 {plan_title} - AEP CertMaster"
        html = f"""
        <html><body>
            <h2>{plan_title}</h2>
            <p>Hola <strong>{recipient_name}</strong>,</p>
            <p>¡Tu plan de estudio personalizado está listo!</p>
            <h3>📅 Resumen del Plan</h3>
            <ul>
                <li><strong>Duración total:</strong> {plan_data.get('total_weeks', 'N/A')} semanas</li>
                <li><strong>Horas semanales:</strong> {plan_data.get('weekly_hours', 'N/A')} horas</li>
                <li><strong>Módulos incluidos:</strong> {len(plan_data.get('modules', []))}</li>
            </ul>
            <h3>ðŸŽ¯ Próximos Pasos</h3>
            <ol>
                <li>Revisa el calendario adjunto para ver tu cronograma</li>
                <li>Importa el archivo .ics a tu calendario favorito</li>
                <li>Comienza con el primer módulo cuando estés listo</li>
            </ol>
            <p>¿Necesitas ajustes? Solo responde a este email.</p>
            <br><p>¡Ã‰xito en tu certificación!<br><strong>AEP CertMaster</strong></p>
        </body></html>
        """
        attachments = []
        if calendar_attachment:
            attachments.append({
                "path": calendar_attachment,
                "filename": "study_plan.ics",
                "content_type": "text/calendar",
            })
        return await self._send_email(recipient_email, subject, html, attachments)

    async def send_reminder_email(
        self,
        recipient_email: str,
        recipient_name: str,
        reminder_data: Dict,
    ) -> bool:
        module_title = reminder_data.get("module_title", "Sesión de estudio")
        subject = f"â° Recordatorio: {module_title}"
        html = f"""
        <html><body>
            <h2>¡Es hora de estudiar!</h2>
            <p>Hola <strong>{recipient_name}</strong>,</p>
            <p>Te recuerdo que tienes programada una sesión de estudio:</p>
            <div style="background:#f0f8ff;padding:15px;border-left:4px solid #007acc;">
                <h3>{module_title}</h3>
                <p>{reminder_data.get('description', '')}</p>
                <p><strong>Duración estimada:</strong>
                {reminder_data.get('duration_minutes', 60)} minutos</p>
            </div>
            <p>¿Listo para continuar hacia tu certificación?</p>
            <br><p>¡Tú puedes!<br><strong>AEP CertMaster</strong></p>
        </body></html>
        """
        return await self._send_email(recipient_email, subject, html)

    async def send_progress_report_email(
        self,
        recipient_email: str,
        recipient_name: str,
        progress_data: Dict,
    ) -> bool:
        subject = f"📅 Tu Progreso en AEP CertMaster - Semana {progress_data.get('week', 'N/A')}"
        completed = progress_data.get("completed_modules", 0)
        total = progress_data.get("total_modules", 1)
        pct = (completed / total) * 100 if total else 0
        next_goals = "".join(
            f"<li>{g}</li>" for g in progress_data.get("next_goals", [])
        )
        html = f"""
        <html><body>
            <h2>📅 Reporte de Progreso Semanal</h2>
            <p>Hola <strong>{recipient_name}</strong>,</p>
            <h3>ðŸŽ¯ Métricas Principales</h3>
            <ul>
                <li><strong>Módulos completados:</strong> {completed}/{total}</li>
                <li><strong>Porcentaje de avance:</strong> {pct:.1f}%</li>
                <li><strong>Horas estudiadas:</strong> {progress_data.get('study_hours', 0)} h</li>
                <li><strong>Puntuación promedio:</strong> {progress_data.get('average_score', 'N/A')}%</li>
            </ul>
            <h3>ðŸŽ¯ Próximos Objetivos</h3><ul>{next_goals}</ul>
            <p>¡Sigue así! Estás más cerca de tu certificación.</p>
            <br><p>Saludos,<br><strong>AEP CertMaster</strong></p>
        </body></html>
        """
        return await self._send_email(recipient_email, subject, html)

    # ------------------------------------------------------------------ #
    #  Métodos internos                                                    #
    # ------------------------------------------------------------------ #

    async def _send_email(
        self,
        recipient: str,
        subject: str,
        html_content: str,
        attachments: Optional[List[Dict[str, str]]] = None,
    ) -> bool:
        try:
            if self.simulate_sending:
                return await self._simulate_send_email(recipient, subject, html_content, attachments)
            try:
                return await self._real_send_email(
                    recipient,
                    subject,
                    html_content,
                    attachments,
                )
            except smtplib.SMTPException as smtp_error:
                logger.warning(
                    "⚠️ Error SMTP enviando email real a %s: %s. "
                    "Aplicando fallback a envío simulado.",
                    recipient,
                    smtp_error,
                )
                return await self._simulate_send_email(
                    recipient,
                    subject,
                    html_content,
                    attachments,
                )
            except OSError as os_error:
                logger.warning(
                    "⚠️ Error de red/encriptación enviando email real a %s: %s. "
                    "Aplicando fallback a envío simulado.",
                    recipient,
                    os_error,
                )
                return await self._simulate_send_email(
                    recipient,
                    subject,
                    html_content,
                    attachments,
                )
        except Exception as e:
            logger.error(f"âŒ Error enviando email a {recipient}: {e}")
            return False

    async def _real_send_email(
        self,
        recipient: str,
        subject: str,
        html_content: str,
        attachments: Optional[List[Dict[str, str]]] = None,
    ) -> bool:
        """Envío real via SMTP con STARTTLS (puerto 587) o SSL (puerto 465)."""

        def _build_message() -> MIMEMultipart:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = f"{self.sender_name} <{self.sender_email}>"
            msg["To"] = recipient
            msg.attach(MIMEText(html_content, "html", "utf-8"))
            return msg

        def _build_multipart_message() -> MIMEMultipart:
            msg = MIMEMultipart("mixed")
            msg["Subject"] = subject
            msg["From"] = f"{self.sender_name} <{self.sender_email}>"
            msg["To"] = recipient
            alt = MIMEMultipart("alternative")
            alt.attach(MIMEText(html_content, "html", "utf-8"))
            msg.attach(alt)
            for att in (attachments or []):
                att_path = Path(att["path"])
                if att_path.exists():
                    with open(att_path, "rb") as f:
                        part = MIMEBase("application", "octet-stream")
                        part.set_payload(f.read())
                    encoders.encode_base64(part)
                    part.add_header(
                        "Content-Disposition",
                        f'attachment; filename="{att.get("filename", att_path.name)}"',
                    )
                    msg.attach(part)
                else:
                    logger.warning(
                        f"âš ï¸  Adjunto no encontrado: {att_path}")
            return msg

        msg = _build_multipart_message() if attachments else _build_message()

        def _send_sync():
            if self.smtp_port == 465:
                # SSL directo
                context = ssl.create_default_context()
                with smtplib.SMTP_SSL(self.smtp_host, self.smtp_port, context=context) as server:
                    server.login(self.smtp_user, self.smtp_password)
                    server.sendmail(self.sender_email,
                                    recipient, msg.as_string())
            else:
                # STARTTLS (puerto 587)
                context = ssl.create_default_context()
                with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=15) as server:
                    server.ehlo()
                    server.starttls(context=context)
                    server.ehlo()
                    server.login(self.smtp_user, self.smtp_password)
                    server.sendmail(self.sender_email,
                                    recipient, msg.as_string())

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _send_sync)

        logger.info(f"✅ Email REAL enviado a {recipient}: {subject}")
        return True

    async def _simulate_send_email(
        self,
        recipient: str,
        subject: str,
        html_content: str,
        attachments: Optional[List[Dict[str, str]]] = None,
    ) -> bool:
        """Guarda el email como JSON (modo simulado)."""
        timestamp = str(int(time.time()))
        email_data = {
            "timestamp": timestamp,
            "recipient": recipient,
            "subject": subject,
            "content": html_content,
            "attachments": attachments or [],
        }
        email_file = self.email_log_dir / f"email_{timestamp}.json"
        with open(email_file, "w", encoding="utf-8") as f:
            json.dump(email_data, f, indent=2, ensure_ascii=False)

        await asyncio.sleep(0.1)
        logger.info(f"📧 Email simulado → {recipient}: {subject}")


# ---- Instancia global del módulo ----
email_tool = EmailTool()
