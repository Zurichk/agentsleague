"""
Study Plan Agent — Genera planes de estudio personalizados.

Este agente toma las rutas de aprendizaje curadas por el Curator Agent
y genera un plan de estudio personalizado basado en la disponibilidad
horaria del estudiante, nivel de experiencia y objetivos de certificación.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta
from typing import Any, Dict, List

from .base_agent import AEPAgent
from src.models.schemas import (
    AEPLearningPath,
    AEPMilestone,
    AEPStudentProfile,
    AEPStudyPlan,
    AEPStudySession,
    AEPWorkflowContext,
)
from src.tools import calendar_tool, email_tool, persistence_tool


class StudyPlanAgent(AEPAgent):
    """
    Agente generador de planes de estudio personalizados.

    Analiza las rutas de aprendizaje curadas, considera la disponibilidad
    horaria del estudiante y genera un cronograma realista con sesiones
    de estudio, hitos y recordatorios automáticos.
    """

    def __init__(self) -> None:
        """Inicializa el Study Plan Agent."""
        super().__init__(
            name="StudyPlanAgent",
            description=(
                "Especialista en crear planes de estudio personalizados "
                "basados en rutas de aprendizaje curadas y disponibilidad del estudiante."
            ),
            capabilities=[
                "Analizar rutas de aprendizaje curadas",
                "Evaluar disponibilidad horaria del estudiante",
                "Generar cronogramas realistas",
                "Crear hitos y sesiones de estudio",
                "Ajustar planes según nivel de experiencia",
                "Generar recordatorios y calendarios",
            ],
            max_tokens=2048,
            temperature=0.4,  # Moderado para creatividad controlada
        )

    async def execute(
        self,
        context: AEPWorkflowContext,
        curated_paths: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Genera un plan de estudio personalizado.

        Args:
            context: Contexto del workflow del estudiante.
            curated_paths: Rutas curadas por el Curator Agent.
            **kwargs: Parámetros adicionales.

        Returns:
            Diccionario con el plan generado y metadatos.
        """
        self.log_reasoning(
            "INICIO",
            "Generando plan de estudio personalizado",
            f"Estudiante: {context.student.name}, Rutas curadas: {len(curated_paths)}",
        )

        # 1. Analizar rutas y calcular tiempo total
        total_hours, path_analysis = self._analyze_learning_paths(
            curated_paths)

        # 2. Determinar fechas del plan
        plan_dates = self._calculate_plan_dates(
            context.student.available_hours_per_week,
            total_hours,
            context.student.target_exam_date,
        )

        # 3. Generar sesiones de estudio
        study_sessions = self._generate_study_sessions(
            curated_paths,
            plan_dates,
            context.student.available_hours_per_week,
            context.student.level,
        )

        # 4. Crear hitos del plan
        milestones = self._generate_milestones(
            study_sessions,
            plan_dates,
            curated_paths,
        )

        # 5. Crear el plan completo
        study_plan = AEPStudyPlan(
            plan_id=f"plan_{uuid.uuid4().hex[:8]}",
            student_id=context.student.student_id,
            certification=context.student.target_certification,
            start_date=plan_dates["start_date"],
            target_exam_date=plan_dates["target_exam_date"],
            sessions=study_sessions,
            milestones=milestones,
            total_hours=total_hours,
            weekly_hours=context.student.available_hours_per_week,
        )

        # 6. Generar calendario ICS
        calendar_path = await self._generate_calendar(study_plan, context.student)

        # 7. Enviar notificación por email
        await self._send_plan_notification(study_plan, context.student)

        # 8. Persistir el plan
        await self._save_study_plan(study_plan)

        result = {
            "study_plan": study_plan,
            "calendar_path": calendar_path,
            "path_analysis": path_analysis,
            "total_sessions": len(study_sessions),
            "total_milestones": len(milestones),
            "estimated_completion_weeks": plan_dates["weeks_needed"],
        }

        self.log_reasoning(
            "COMPLETADO",
            f"Plan generado: {len(study_sessions)} sesiones, {len(milestones)} hitos",
            f"Duración: {plan_dates['weeks_needed']} semanas, {total_hours:.1f} horas totales",
        )

        return result

    def _analyze_learning_paths(
        self,
        curated_paths: List[Dict[str, Any]],
    ) -> tuple[float, Dict[str, Any]]:
        """
        Analiza las rutas curadas y calcula tiempo total estimado.

        Args:
            curated_paths: Rutas curadas por el Curator Agent.

        Returns:
            Tupla con horas totales y análisis detallado.
        """
        total_hours = 0.0
        path_details = []

        for path in curated_paths:
            # Estimar horas basado en módulos (cada módulo ~2-4 horas)
            modules_count = path.get(
                "modules_count", len(path.get("modules", [])))
            path_hours = modules_count * 3.0  # 3 horas promedio por módulo

            # Ajustar por dificultad
            difficulty_multiplier = {
                "beginner": 1.2,  # Más tiempo para principiantes
                "intermediate": 1.0,
                "advanced": 0.8,  # Menos tiempo para avanzados
            }.get(path.get("difficulty", "intermediate"), 1.0)

            adjusted_hours = path_hours * difficulty_multiplier
            total_hours += adjusted_hours

            path_details.append({
                "title": path.get("title", "Ruta sin título"),
                "modules_count": modules_count,
                "estimated_hours": adjusted_hours,
                "difficulty": path.get("difficulty", "intermediate"),
            })

        analysis = {
            "total_paths": len(curated_paths),
            "total_estimated_hours": total_hours,
            "average_hours_per_path": total_hours / len(curated_paths) if curated_paths else 0,
            "path_details": path_details,
        }

        return total_hours, analysis

    def _calculate_plan_dates(
        self,
        weekly_hours: float,
        total_hours: float,
        target_exam_date: date | None,
    ) -> Dict[str, Any]:
        """
        Calcula las fechas del plan de estudio.

        Args:
            weekly_hours: Horas disponibles por semana.
            total_hours: Horas totales estimadas.
            target_exam_date: Fecha objetivo del examen.

        Returns:
            Diccionario con fechas calculadas.
        """
        today = date.today()

        # Calcular semanas necesarias
        # +1 semana buffer
        weeks_needed = max(1, int(total_hours / weekly_hours) + 1)

        if target_exam_date:
            # Si hay fecha objetivo, calcular hacia atrás
            calculated_start = target_exam_date - timedelta(weeks=weeks_needed)
            start_date = max(today, calculated_start)
            actual_weeks = (target_exam_date - start_date).days // 7
        else:
            # Sin fecha objetivo, empezar hoy
            start_date = today
            target_exam_date = start_date + timedelta(weeks=weeks_needed)
            actual_weeks = weeks_needed

        return {
            "start_date": start_date,
            "target_exam_date": target_exam_date,
            "weeks_needed": actual_weeks,
            "buffer_weeks": weeks_needed - actual_weeks if target_exam_date else 0,
        }

    def _generate_study_sessions(
        self,
        curated_paths: List[Dict[str, Any]],
        plan_dates: Dict[str, Any],
        weekly_hours: float,
        student_level: str,
    ) -> List[AEPStudySession]:
        """
        Genera sesiones de estudio distribuidas en el tiempo.

        Args:
            curated_paths: Rutas curadas.
            plan_dates: Fechas del plan.
            weekly_hours: Horas semanales disponibles.
            student_level: Nivel del estudiante.

        Returns:
            Lista de sesiones de estudio.
        """
        sessions = []
        current_date = plan_dates["start_date"]

        # Convertir horas semanales a minutos diarios
        daily_minutes = int((weekly_hours * 60) / 7)  # Distribuir en 7 días

        # Ajustar por nivel de estudiante
        level_multiplier = {
            "beginner": 0.8,  # Sesiones más cortas para principiantes
            "intermediate": 1.0,
            "advanced": 1.2,  # Sesiones más largas para avanzados
        }.get(student_level, 1.0)

        session_minutes = int(daily_minutes * level_multiplier)

        # Generar sesiones para cada ruta
        for path in curated_paths:
            modules = path.get("modules", [])

            for i, module in enumerate(modules):
                # Crear sesión para este módulo
                session = AEPStudySession(
                    session_date=current_date,
                    topic=module.get("title", f"Módulo {i+1}"),
                    module_title=path.get("title", "Ruta de aprendizaje"),
                    duration_minutes=min(
                        session_minutes, 180),  # Máximo 3 horas
                    objectives=[
                        f"Completar módulo: {module.get('title', f'Módulo {i+1}')}",
                        "Tomar notas de conceptos clave",
                        "Completar ejercicios prácticos",
                    ],
                    completed=False,
                )

                sessions.append(session)

                # Avanzar fecha (distribuir sesiones)
                current_date += timedelta(days=1)

                # Si es fin de semana, saltar al lunes
                if current_date.weekday() >= 5:  # Sábado o domingo
                    current_date += timedelta(days=2)

        return sessions

    def _generate_milestones(
        self,
        sessions: List[AEPStudySession],
        plan_dates: Dict[str, Any],
        curated_paths: List[Dict[str, Any]],
    ) -> List[AEPMilestone]:
        """
        Genera hitos importantes del plan de estudio.

        Args:
            sessions: Sesiones generadas.
            plan_dates: Fechas del plan.
            curated_paths: Rutas curadas.

        Returns:
            Lista de hitos.
        """
        milestones = []
        total_sessions = len(sessions)
        start_date = plan_dates["start_date"]
        end_date = plan_dates["target_exam_date"]

        # Hito inicial
        milestones.append(AEPMilestone(
            title="Inicio del Plan de Estudio",
            target_date=start_date,
            description="Comenzar el primer módulo del plan de estudio",
            achieved=False,
        ))

        # Hitos por cada ruta completada
        sessions_completed = 0
        for i, path in enumerate(curated_paths):
            path_sessions = len(path.get("modules", []))
            sessions_completed += path_sessions

            # Calcular fecha proporcional
            progress_ratio = sessions_completed / total_sessions
            milestone_date = start_date + timedelta(
                days=int((end_date - start_date).days * progress_ratio)
            )

            milestones.append(AEPMilestone(
                title=f"Completar: {path.get('title', f'Ruta {i+1}')}",
                target_date=milestone_date,
                description=f"Finalizar todos los módulos de {path.get('title', f'Ruta {i+1}')}",
                achieved=False,
            ))

        # Hito final - preparación para examen
        exam_prep_date = end_date - timedelta(days=7)
        milestones.append(AEPMilestone(
            title="Preparación Final para Examen",
            target_date=exam_prep_date,
            description="Repaso general y práctica de examen",
            achieved=False,
        ))

        # Hito final - examen
        milestones.append(AEPMilestone(
            title="Examen de Certificación",
            target_date=end_date,
            description=f"Tomar el examen {curated_paths[0].get('certification', 'de certificación') if curated_paths else 'de certificación'}",
            achieved=False,
        ))

        return milestones

    async def _generate_calendar(
        self,
        study_plan: AEPStudyPlan,
        student: AEPStudentProfile,
    ) -> str:
        """
        Genera archivo de calendario ICS para el plan de estudio.

        Args:
            study_plan: Plan de estudio generado.
            student: Perfil del estudiante.

        Returns:
            Ruta del archivo de calendario generado.
        """
        try:
            # Convertir sesiones a formato de calendario
            plan_data = {
                "plan_id": study_plan.plan_id,
                "title": f"Plan de Estudio - {study_plan.certification}",
                "start_date": study_plan.start_date.isoformat(),
                "end_date": study_plan.target_exam_date.isoformat(),
                "sessions": [
                    {
                        "session_id": f"session_{i}",
                        "title": session.topic,
                        "date": session.session_date.isoformat(),
                        "duration_hours": session.duration_minutes / 60.0,
                        "description": f"Objetivos: {', '.join(session.objectives)}",
                    }
                    for i, session in enumerate(study_plan.sessions)
                ],
            }

            calendar_path = calendar_tool.generate_study_plan_calendar(
                plan_data=plan_data,
                student_email=student.email,
            )

            self.logger.info(f"Calendario generado: {calendar_path}")
            return calendar_path

        except Exception as e:
            self.logger.error(f"Error generando calendario: {e}")
            return ""

    async def _send_plan_notification(
        self,
        study_plan: AEPStudyPlan,
        student: AEPStudentProfile,
    ) -> None:
        """
        Envía notificación por email con el plan de estudio.

        Args:
            study_plan: Plan generado.
            student: Perfil del estudiante.
        """
        try:
            plan_data = {
                "title": f"Plan de Estudio - {study_plan.certification}",
                "total_weeks": (study_plan.target_exam_date - study_plan.start_date).days // 7,
                "weekly_hours": study_plan.weekly_hours,
                "total_sessions": len(study_plan.sessions),
                "start_date": study_plan.start_date.strftime("%d/%m/%Y"),
                "exam_date": study_plan.target_exam_date.strftime("%d/%m/%Y"),
            }

            success = await email_tool.send_study_plan_email(
                recipient_email=student.email,
                recipient_name=student.name,
                plan_data=plan_data,
            )

            if success:
                self.logger.info(f"Email de plan enviado a {student.email}")
            else:
                self.logger.warning(f"Error enviando email a {student.email}")

        except Exception as e:
            self.logger.error(f"Error enviando notificación: {e}")

    async def _save_study_plan(self, study_plan: AEPStudyPlan) -> None:
        """
        Guarda el plan de estudio en la base de datos.

        Args:
            study_plan: Plan a guardar.
        """
        try:
            # Convertir a diccionario para persistencia
            plan_dict = study_plan.model_dump()

            # Usar persistence tool para guardar
            success = persistence_tool.save_study_plan(
                plan_id=study_plan.plan_id,
                student_id=study_plan.student_id,
                plan_data=plan_dict,
            )

            if success:
                self.logger.info(
                    f"Plan guardado para estudiante {study_plan.student_id}")
            else:
                self.logger.error(
                    f"Error guardando plan para {study_plan.student_id}")

        except Exception as e:
            self.logger.error(f"Error guardando plan de estudio: {e}")
