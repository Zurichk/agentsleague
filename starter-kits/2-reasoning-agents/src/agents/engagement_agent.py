"""
Engagement Agent — seguimiento de constancia y recordatorios.

Este agente mantiene la motivación del estudiante sin mecánicas de
recompensas extrínsecas. Su foco es la constancia, recordatorios accionables y
mensajes personalizados por contexto.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from .base_agent import AEPAgent
from src.models.schemas import AEPWorkflowContext
from src.tools import email_tool


class EngagementAgent(AEPAgent):
    """Agente de motivación y recordatorios personalizados."""

    def __init__(self) -> None:
        """Inicializa el Engagement Agent."""
        super().__init__(
            name="EngagementAgent",
            description=(
                "Especialista en seguimiento de constancia y motivación "
                "mediante recordatorios y recomendaciones accionables."
            ),
            capabilities=[
                "Seguimiento de constancia de estudio",
                "Recordatorios de sesiones pendientes",
                "Mensajes motivacionales personalizados",
                "Recomendaciones para sostener el ritmo semanal",
            ],
            max_tokens=1200,
            temperature=0.6,
        )

    async def execute(
        self,
        context: AEPWorkflowContext,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Genera recomendaciones y recordatorios de motivación.

        Args:
            context: Contexto del workflow del estudiante.
            **kwargs: Parámetros adicionales no usados.

        Returns:
            Diccionario con recomendaciones de seguimiento.
        """
        student = context.student
        self.log_reasoning(
            "INICIO",
            "Generando seguimiento de motivación",
            f"Estudiante: {student.name}",
        )

        reminders = self._build_reminders(context)
        recommendations = self._build_recommendations(context)
        motivational_message = self._build_motivational_message(context)

        await self._send_follow_up_email(context, reminders, motivational_message)

        result = {
            "reminders": reminders,
            "recommendations": recommendations,
            "motivational_message": motivational_message,
            "consistency_score": self._compute_consistency_score(context),
        }

        self.log_reasoning(
            "COMPLETADO",
            "Seguimiento generado",
            f"Recordatorios: {len(reminders)}",
        )
        return result

    def _build_reminders(
        self,
        context: AEPWorkflowContext,
    ) -> List[Dict[str, Any]]:
        """Construye recordatorios en función de sesiones pendientes."""
        reminders: List[Dict[str, Any]] = []
        today = datetime.now().date()

        if not context.study_plan:
            return reminders

        pending = [
            session for session in context.study_plan.sessions
            if not session.completed and session.session_date >= today
        ]

        if pending:
            next_session = min(
                pending, key=lambda session: session.session_date)
            days_until = (next_session.session_date - today).days
            if days_until == 0:
                reminders.append({
                    "type": "today_session",
                    "title": "Sesión programada para hoy",
                    "message": f"Hoy toca: {next_session.topic}",
                    "priority": "high",
                })
            else:
                reminders.append({
                    "type": "upcoming_session",
                    "title": "Próxima sesión",
                    "message": (
                        f"En {days_until} día(s): {next_session.topic}"
                    ),
                    "priority": "medium",
                })

        reminders.append({
            "type": "weekly_review",
            "title": "Cierre semanal",
            "message": "Reserva 20 minutos para repasar avances y ajustar la semana siguiente.",
            "priority": "low",
        })
        return reminders

    def _build_recommendations(
        self,
        context: AEPWorkflowContext,
    ) -> List[str]:
        """Genera recomendaciones de constancia concretas."""
        recommendations: List[str] = []

        if context.study_plan and context.study_plan.weekly_hours < 6:
            recommendations.append(
                "Aumenta a 6-8 horas semanales para acelerar la preparación."
            )

        recommendations.append(
            "Bloquea horario fijo de estudio y evita mover sesiones salvo causa mayor."
        )
        recommendations.append(
            "Al final de cada sesión, anota 3 conceptos clave y 1 duda concreta."
        )

        if context.student.level == "beginner":
            recommendations.append(
                "Prioriza módulos introductorios y mini-prácticas guiadas en sandbox."
            )

        return recommendations[:4]

    def _build_motivational_message(
        self,
        context: AEPWorkflowContext,
    ) -> str:
        """Construye un mensaje motivacional personalizado."""
        cert = context.student.target_certification or "tu certificación objetivo"
        topics = context.student.topics_of_interest or []
        topic_hint = topics[0] if topics else "tu ruta de estudio"

        return (
            f"Vas avanzando hacia {cert}. Enfócate esta semana en {topic_hint} "
            "y mantén sesiones constantes: la consistencia vence a la intensidad "
            "ocasional."
        )

    def _compute_consistency_score(self, context: AEPWorkflowContext) -> float:
        """Calcula una señal simple de constancia (0.0-1.0)."""
        if not context.study_plan or not context.study_plan.sessions:
            return 0.0

        total = len(context.study_plan.sessions)
        completed = sum(
            1 for session in context.study_plan.sessions if session.completed
        )
        return completed / max(total, 1)

    async def _send_follow_up_email(
        self,
        context: AEPWorkflowContext,
        reminders: List[Dict[str, Any]],
        motivational_message: str,
    ) -> None:
        """Envía email de seguimiento si hay correo del estudiante."""
        student = context.student
        if not getattr(student, "email", "") or not email_tool:
            return

        try:
            reminder_lines = "".join(
                f"<li><strong>{item['title']}:</strong> {item['message']}</li>"
                for item in reminders
            )
            html_content = (
                "<html><body>"
                f"<h2>Seguimiento de estudio para {student.name}</h2>"
                "<p>Resumen de recordatorios:</p>"
                f"<ul>{reminder_lines}</ul>"
                f"<p><strong>Motivación:</strong> {motivational_message}</p>"
                "<p>Equipo AEP CertMaster</p>"
                "</body></html>"
            )

            await email_tool.send_custom_email(
                recipient_email=student.email,
                subject="Seguimiento de estudio AEP CertMaster",
                html_content=html_content,
            )
        except Exception as exc:
            self.logger.warning(
                f"No se pudo enviar email de seguimiento: {exc}")
