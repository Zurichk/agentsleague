"""
Agente Asesor de Certificación - AEP CertMaster

Este agente implementa recomendaciones de certificación, evaluación de preparación
y guía personalizada hacia objetivos de certificación profesional.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from pydantic import BaseModel, Field

from .base_agent import AEPAgent
from src.models.schemas import (
    AEPWorkflowContext,
    AgentResponse,
    AEPStudentProfile,
    AEPStudyPlan,
    AEPStudySession,
    AEPCertificationPath
)
from src.tools.azure_openai_tool import AzureOpenAITool
from src.tools.certifications import certifications_tool
from src.tools.persistence import PersistenceTool


class CertificationLevel(Enum):
    """Niveles de certificación."""

    FOUNDATIONAL = "foundational"
    ASSOCIATE = "associate"
    PROFESSIONAL = "professional"
    EXPERT = "expert"
    MASTER = "master"


class CertificationProvider(Enum):
    """Proveedores de certificación principales."""

    AWS = "aws"
    AZURE = "azure"
    GOOGLE_CLOUD = "google_cloud"
    CISCO = "cisco"
    COMPTIA = "compTIA"
    PMI = "pmi"
    ISC2 = "isc2"
    OTHER = "other"


class ReadinessLevel(Enum):
    """Niveles de preparación."""

    NOT_READY = "not_ready"
    BEGINNING = "beginning"
    DEVELOPING = "developing"
    PROFICIENT = "proficient"
    READY = "ready"


class AEPCertificationRecommendation(BaseModel):
    """
    Recomendación de certificación.

    Attributes:
        certification_id: ID de la certificación recomendada.
        title: Título de la certificación.
        provider: Proveedor de la certificación.
        level: Nivel de la certificación.
        relevance_score: Puntuación de relevancia (0-100).
        readiness_level: Nivel de preparación del estudiante.
        estimated_prep_time: Tiempo estimado de preparación en semanas.
        prerequisites: Prerrequisitos necesarios.
        career_impact: Impacto en carrera profesional.
        recommended_path: Ruta recomendada para lograrla.
    """

    certification_id: str = Field(description="ID de la certificación")
    title: str = Field(description="Título de la certificación")
    provider: CertificationProvider = Field(description="Proveedor")
    level: CertificationLevel = Field(description="Nivel")
    relevance_score: int = Field(description="Relevancia (0-100)")
    readiness_level: ReadinessLevel = Field(description="Preparación")
    estimated_prep_time: int = Field(description="Semanas de preparación")
    prerequisites: List[str] = Field(
        default_factory=list,
        description="Prerrequisitos"
    )
    career_impact: str = Field(description="Impacto profesional")
    recommended_path: List[str] = Field(
        default_factory=list,
        description="Ruta recomendada"
    )


class AEPCertAdvisorAgent(AEPAgent):
    """
    Agente especializado en asesoramiento de certificaciones.

    Funcionalidades principales:
    - Recomendaciones personalizadas de certificación
    - Evaluación de preparación y brechas de conocimiento
    - Diseño de rutas de certificación
    - Análisis de impacto profesional
    - Seguimiento de progreso hacia certificaciones
    - Sugerencias de certificaciones complementarias
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa el agente asesor de certificación.

        Args:
            config: Configuración del agente.
        """
        super().__init__(
            name="AEPCertAdvisorAgent",
            description=(
                "Especialista en asesoramiento de certificaciones Microsoft "
                "y planificación de carrera profesional."
            ),
            capabilities=[
                "Analizar certificaciones disponibles",
                "Evaluar impacto profesional",
                "Recomendar rutas de certificación",
                "Seguimiento de progreso hacia certificaciones",
                "Sugerencias de certificaciones complementarias",
            ],
            max_tokens=config.get("max_tokens", 2048),
            temperature=config.get("temperature", 0.7),
        )
        self.openai_tool = AzureOpenAITool(config.get("azure_openai", {}))
        self.persistence_tool = PersistenceTool(config.get("persistence", {}))

        # Configuración específica del agente
        self.certification_database = self._load_certification_database()
        self._catalog_loaded_from_api = False
        self.career_impact_weights = config.get("career_impact_weights", {
            "salary_increase": 0.3,
            "job_opportunities": 0.3,
            "skill_relevance": 0.2,
            "industry_demand": 0.2
        })

        self.logger.info("Agente Asesor de Certificación inicializado")

    async def execute(
        self,
        context: AEPWorkflowContext,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Ejecuta asesoramiento de certificación usando Azure OpenAI.

        Args:
            context: Contexto del workflow del estudiante.
            **kwargs: Parámetros adicionales.

        Returns:
            Recomendaciones personalizadas de certificación.
        """
        try:
            student = context.student
            target_cert = student.target_certification or "General IT"
            topics = student.topics_of_interest or []
            level = getattr(student, "level", "intermediate")

            # Asesoramiento real con Azure OpenAI
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Eres un asesor experto en certificaciones IT (Microsoft) "
                        "Usa toda tu capacidad para construir recomendaciones estratégicas, comparativas y accionables con el máximo ajuste al perfil del estudiante. "
                        "Proporciona asesoramiento personalizado, práctico y accionable sobre rutas de certificación. "
                        "Basa las recomendaciones en el perfil real del estudiante. "
                        "Responde SOLO con JSON válido, sin texto adicional."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Genera un plan de asesoramiento personalizado para:\n"
                        f"- Certificación objetivo: {target_cert}\n"
                        f"- Nivel actual: {level}\n"
                        f"- Temas de interés: {', '.join(topics[:6]) if topics else 'No especificados'}\n\n"
                        "Responde con el siguiente JSON:\n"
                        "{\n"
                        '  "readiness_level": "<beginning|developing|ready>",\n'
                        '  "readiness_justification": "<explicación de 1-2 oraciones>",\n'
                        '  "recommendations": [\n'
                        "    {\n"
                        '      "certification": "nombre exacto de la certificación",\n'
                        '      "provider": "proveedor",\n'
                        '      "level": "Foundational|Associate|Professional|Expert",\n'
                        '      "reason": "razón específica por la que se recomienda",\n'
                        '      "estimated_time": "X semanas / X meses"\n'
                        "    }\n"
                        "  ],\n"
                        '  "next_steps": [\n'
                        '    "paso concreto 1",\n'
                        '    "paso concreto 2",\n'
                        '    "paso concreto 3"\n'
                        "  ],\n"
                        '  "career_impact": "<descripción del impacto profesional esperado>"\n'
                        "}"
                    ),
                },
            ]

            raw = await self._call_azure_openai(messages, temperature=0.5, max_tokens=1000)

            # Parsear respuesta
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                parts = cleaned.split("```")
                cleaned = parts[1] if len(parts) > 1 else cleaned
                if cleaned.lower().startswith("json"):
                    cleaned = cleaned[4:]
            cleaned = cleaned.strip()

            result = json.loads(cleaned)
            readiness_level = result.get("readiness_level", "developing")

            return {
                "success": True,
                "message": "Recomendaciones de certificación generadas",
                "target_certification": target_cert,
                "readiness_level": readiness_level,
                "readiness_justification": result.get("readiness_justification", ""),
                "recommendations": result.get("recommendations", []),
                "next_steps": result.get("next_steps", []),
                "career_impact": result.get("career_impact", ""),
            }

        except Exception as e:
            self.logger.error(f"Error en Cert Advisor Agent: {str(e)}")
            return {
                "success": False,
                "message": f"Error interno del agente: {str(e)}",
            }

    async def _get_certification_recommendations(
        self,
        context: dict
    ) -> AgentResponse:
        """
        Obtiene recomendaciones de certificación personalizadas.

        Args:
            context: Contexto de la solicitud.

        Returns:
            Recomendaciones de certificación.
        """
        student_id = context.request_data.get("student_id")
        career_goals = context.request_data.get("career_goals", [])
        current_experience = context.request_data.get("current_experience", {})
        preferred_providers = context.request_data.get(
            "preferred_providers", [])
        time_commitment = context.request_data.get(
            "time_commitment", "moderate")

        if not student_id:
            return AgentResponse(
                success=False,
                message="Se requiere student_id para recomendaciones"
            )

        # Obtener perfil del estudiante
        student_profile = await self.persistence_tool.get_student_profile(student_id)
        if not student_profile:
            return AgentResponse(
                success=False,
                message=f"Perfil de estudiante no encontrado: {student_id}"
            )

        # Sincronizar catálogo oficial antes de recomendar
        await self._refresh_certification_database_from_api()

        # Generar recomendaciones personalizadas
        recommendations = await self._generate_personalized_recommendations(
            student_profile,
            career_goals,
            current_experience,
            preferred_providers,
            time_commitment
        )

        # Ordenar por relevancia
        recommendations.sort(key=lambda x: x.relevance_score, reverse=True)

        return AgentResponse(
            success=True,
            message=f"Generadas {len(recommendations)} recomendaciones de certificación",
            data={
                "recommendations": [r.model_dump() for r in recommendations],
                "top_recommendation": recommendations[0].model_dump() if recommendations else None,
                "recommendation_summary": self._create_recommendation_summary(recommendations)
            }
        )

    async def _assess_certification_readiness(
        self,
        context: dict
    ) -> AgentResponse:
        """
        Evalúa la preparación del estudiante para una certificación específica.

        Args:
            context: Contexto con la certificación a evaluar.

        Returns:
            Evaluación de preparación.
        """
        student_id = context.request_data.get("student_id")
        certification_id = context.request_data.get("certification_id")

        if not all([student_id, certification_id]):
            return AgentResponse(
                success=False,
                message="Se requieren student_id y certification_id"
            )

        await self._refresh_certification_database_from_api()

        # Obtener perfil y certificación
        student_profile = await self.persistence_tool.get_student_profile(student_id)
        certification = self._get_certification_by_id(certification_id)

        if not student_profile:
            return AgentResponse(
                success=False,
                message=f"Perfil de estudiante no encontrado: {student_id}"
            )

        if not certification:
            return AgentResponse(
                success=False,
                message=f"Certificación no encontrada: {certification_id}"
            )

        # Realizar evaluación de preparación
        readiness_assessment = await self._perform_readiness_assessment(
            student_profile,
            certification
        )

        # Identificar brechas de conocimiento
        knowledge_gaps = await self._identify_knowledge_gaps(
            student_profile,
            certification
        )

        # Crear plan de preparación si es necesario
        prep_plan = None
        if readiness_assessment["readiness_level"] != ReadinessLevel.READY:
            prep_plan = await self._create_preparation_plan(
                student_profile,
                certification,
                knowledge_gaps
            )

        return AgentResponse(
            success=True,
            message="Evaluación de preparación completada",
            data={
                "readiness_assessment": readiness_assessment,
                "knowledge_gaps": knowledge_gaps,
                "preparation_plan": prep_plan.model_dump() if prep_plan else None,
                "estimated_prep_time": readiness_assessment.get("estimated_prep_weeks", 0)
            }
        )

    async def _create_certification_path(
        self,
        context: dict
    ) -> AgentResponse:
        """
        Crea una ruta de certificación completa.

        Args:
            context: Contexto con objetivos de certificación.

        Returns:
            Ruta de certificación diseñada.
        """
        student_id = context.request_data.get("student_id")
        target_certifications = context.request_data.get(
            "target_certifications", [])
        timeline_months = context.request_data.get("timeline_months", 12)
        budget_constraints = context.request_data.get("budget_constraints", {})

        if not student_id:
            return AgentResponse(
                success=False,
                message="Se requiere student_id para crear ruta"
            )

        # Obtener perfil del estudiante
        student_profile = await self.persistence_tool.get_student_profile(student_id)
        if not student_profile:
            return AgentResponse(
                success=False,
                message=f"Perfil de estudiante no encontrado: {student_id}"
            )

        # Crear ruta de certificación
        certification_path = await self._design_certification_path(
            student_profile,
            target_certifications,
            timeline_months,
            budget_constraints
        )

        # Guardar la ruta
        await self.persistence_tool.save_certification_path(certification_path)

        return AgentResponse(
            success=True,
            message="Ruta de certificación creada exitosamente",
            data={
                "certification_path": certification_path.model_dump(),
                "milestones": certification_path.milestones,
                "total_cost_estimate": certification_path.total_cost_estimate,
                "success_probability": certification_path.success_probability
            }
        )

    async def _analyze_career_impact(
        self,
        context: dict
    ) -> AgentResponse:
        """
        Analiza el impacto profesional de una certificación.

        Args:
            context: Contexto con la certificación a analizar.

        Returns:
            Análisis de impacto profesional.
        """
        certification_id = context.request_data.get("certification_id")
        industry = context.request_data.get("industry", "technology")
        experience_level = context.request_data.get("experience_level", "mid")

        if not certification_id:
            return AgentResponse(
                success=False,
                message="Se requiere certification_id para análisis"
            )

        await self._refresh_certification_database_from_api()

        certification = self._get_certification_by_id(certification_id)
        if not certification:
            return AgentResponse(
                success=False,
                message=f"Certificación no encontrada: {certification_id}"
            )

        # Realizar análisis de impacto
        career_impact = await self._calculate_career_impact(
            certification,
            industry,
            experience_level
        )

        return AgentResponse(
            success=True,
            message="Análisis de impacto profesional completado",
            data={
                "certification": certification.model_dump(),
                "career_impact": career_impact,
                "market_demand": career_impact.get("market_demand", {}),
                "salary_projection": career_impact.get("salary_projection", {}),
                "skill_alignment": career_impact.get("skill_alignment", {})
            }
        )

    async def _get_certification_progress(
        self,
        context: dict
    ) -> AgentResponse:
        """
        Obtiene el progreso del estudiante hacia certificaciones.

        Args:
            context: Contexto de la solicitud.

        Returns:
            Estado de progreso.
        """
        student_id = context.request_data.get("student_id")

        if not student_id:
            return AgentResponse(
                success=False,
                message="Se requiere student_id para seguimiento"
            )

        # Obtener rutas de certificación activas
        certification_paths = await self.persistence_tool.get_student_certification_paths(student_id)

        if not certification_paths:
            return AgentResponse(
                success=True,
                message="No hay rutas de certificación activas",
                data={"active_paths": 0}
            )

        # Calcular progreso para cada ruta
        progress_reports = []
        for path in certification_paths:
            progress = await self._calculate_path_progress(path)
            progress_reports.append(progress)

        return AgentResponse(
            success=True,
            message=f"Progreso calculado para {len(certification_paths)} rutas",
            data={
                "progress_reports": progress_reports,
                "overall_completion": self._calculate_overall_completion(progress_reports),
                "next_milestones": self._get_upcoming_milestones(progress_reports)
            }
        )

    def _load_certification_database(self) -> Dict[str, Dict[str, Any]]:
        """
        Carga la base de datos de certificaciones.

        Returns:
            Base de datos de certificaciones.
        """
        # En una implementación real, esto vendría de una base de datos
        # o archivo de configuración. Aquí incluimos algunas certificaciones comunes.
        return {
            "aws-saa": {
                "title": "AWS Solutions Architect Associate",
                "provider": "aws",
                "level": "associate",
                "prerequisites": [],
                "estimated_prep_weeks": 8,
                "cost_usd": 150,
                "validity_years": 3,
                "knowledge_areas": ["cloud_computing", "aws_services", "architecture_design"]
            },
            "azure-az900": {
                "title": "Microsoft Azure Fundamentals",
                "provider": "azure",
                "level": "foundational",
                "prerequisites": [],
                "estimated_prep_weeks": 4,
                "cost_usd": 99,
                "validity_years": 999,  # No expira
                "knowledge_areas": ["cloud_computing", "azure_services"]
            },
            "compTIA-a+": {
                "title": "CompTIA A+",
                "provider": "compTIA",
                "level": "associate",
                "prerequisites": [],
                "estimated_prep_weeks": 6,
                "cost_usd": 226,
                "validity_years": 3,
                "knowledge_areas": ["hardware", "software", "troubleshooting"]
            },
            "cisco-ccna": {
                "title": "Cisco Certified Network Associate",
                "provider": "cisco",
                "level": "associate",
                "prerequisites": [],
                "estimated_prep_weeks": 10,
                "cost_usd": 300,
                "validity_years": 3,
                "knowledge_areas": ["networking", "cisco_technologies", "routing_switching"]
            }
        }

    async def _generate_personalized_recommendations(
        self,
        student_profile: AEPStudentProfile,
        career_goals: List[str],
        current_experience: Dict[str, Any],
        preferred_providers: List[str],
        time_commitment: str
    ) -> List[AEPCertificationRecommendation]:
        """
        Genera recomendaciones personalizadas de certificación.

        Args:
            student_profile: Perfil del estudiante.
            career_goals: Objetivos profesionales.
            current_experience: Experiencia actual.
            preferred_providers: Proveedores preferidos.
            time_commitment: Compromiso de tiempo.

        Returns:
            Lista de recomendaciones.
        """
        recommendations = []

        # Analizar perfil del estudiante
        knowledge_areas = [
            area.area_name for area in student_profile.knowledge_areas]
        experience_years = current_experience.get("years", 0)
        current_role = current_experience.get("role", "")

        # Filtrar certificaciones relevantes
        relevant_certs = self._filter_relevant_certifications(
            knowledge_areas,
            career_goals,
            preferred_providers,
            experience_years
        )

        for cert_id, cert_data in relevant_certs.items():
            # Calcular relevancia
            relevance_score = await self._calculate_relevance_score(
                cert_data,
                student_profile,
                career_goals,
                current_experience
            )

            # Evaluar preparación
            readiness_level = await self._assess_readiness_level(
                student_profile,
                cert_data
            )

            # Ajustar por compromiso de tiempo
            time_adjustment = self._adjust_for_time_commitment(
                cert_data["estimated_prep_weeks"],
                time_commitment
            )

            recommendation = AEPCertificationRecommendation(
                certification_id=cert_id,
                title=cert_data["title"],
                provider=CertificationProvider(cert_data["provider"]),
                level=CertificationLevel(cert_data["level"]),
                relevance_score=min(100, relevance_score + time_adjustment),
                readiness_level=readiness_level,
                estimated_prep_time=cert_data["estimated_prep_weeks"],
                prerequisites=cert_data.get("prerequisites", []),
                career_impact=await self._get_career_impact_description(cert_data),
                recommended_path=self._get_recommended_path(
                    cert_data, student_profile)
            )

            recommendations.append(recommendation)

        return recommendations

    async def _calculate_relevance_score(
        self,
        cert_data: Dict[str, Any],
        student_profile: AEPStudentProfile,
        career_goals: List[str],
        current_experience: Dict[str, Any]
    ) -> int:
        """
        Calcula puntuación de relevancia para una certificación.

        Args:
            cert_data: Datos de la certificación.
            student_profile: Perfil del estudiante.
            career_goals: Objetivos profesionales.
            current_experience: Experiencia actual.

        Returns:
            Puntuación de relevancia (0-100).
        """
        score = 50  # Base score

        # Coincidencia con áreas de conocimiento
        cert_areas = set(cert_data.get("knowledge_areas", []))
        student_areas = set(
            area.area_name for area in student_profile.knowledge_areas)

        area_overlap = len(cert_areas.intersection(student_areas))
        if area_overlap > 0:
            score += 20 * min(area_overlap, 3)  # Máximo +60

        # Alineación con objetivos profesionales
        cert_title_lower = cert_data["title"].lower()
        for goal in career_goals:
            if goal.lower() in cert_title_lower:
                score += 15
                break

        # Nivel de experiencia apropiado
        experience_years = current_experience.get("years", 0)
        cert_level = cert_data["level"]

        level_appropriateness = self._check_level_appropriateness(
            experience_years,
            cert_level
        )
        score += level_appropriateness * 10

        return min(100, max(0, score))

    async def _assess_readiness_level(
        self,
        student_profile: AEPStudentProfile,
        cert_data: Dict[str, Any]
    ) -> ReadinessLevel:
        """
        Evalúa el nivel de preparación del estudiante.

        Args:
            student_profile: Perfil del estudiante.
            cert_data: Datos de la certificación.

        Returns:
            Nivel de preparación.
        """
        # Calcular promedio de proficiency en áreas relevantes
        cert_areas = cert_data.get("knowledge_areas", [])
        relevant_proficiencies = []

        for cert_area in cert_areas:
            for student_area in student_profile.knowledge_areas:
                if student_area.area_name == cert_area:
                    relevant_proficiencies.append(
                        student_area.proficiency_level)
                    break

        if not relevant_proficiencies:
            return ReadinessLevel.BEGINNING

        avg_proficiency = sum(relevant_proficiencies) / \
            len(relevant_proficiencies)

        if avg_proficiency >= 0.8:
            return ReadinessLevel.READY
        elif avg_proficiency >= 0.6:
            return ReadinessLevel.PROFICIENT
        elif avg_proficiency >= 0.4:
            return ReadinessLevel.DEVELOPING
        elif avg_proficiency >= 0.2:
            return ReadinessLevel.BEGINNING
        else:
            return ReadinessLevel.NOT_READY

    def _adjust_for_time_commitment(
        self,
        prep_weeks: int,
        time_commitment: str
    ) -> int:
        """
        Ajusta puntuación basado en compromiso de tiempo.

        Args:
            prep_weeks: Semanas de preparación.
            time_commitment: Compromiso de tiempo.

        Returns:
            Ajuste de puntuación.
        """
        if time_commitment == "high" and prep_weeks <= 6:
            return 10  # Bonus por certificaciones rápidas
        elif time_commitment == "moderate" and prep_weeks <= 12:
            return 5
        elif time_commitment == "low" and prep_weeks > 16:
            return -10  # Penalización por certificaciones muy largas
        return 0

    async def _perform_readiness_assessment(
        self,
        student_profile: AEPStudentProfile,
        certification: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Realiza evaluación completa de preparación.

        Args:
            student_profile: Perfil del estudiante.
            certification: Datos de la certificación.

        Returns:
            Evaluación de preparación.
        """
        readiness_level = await self._assess_readiness_level(
            student_profile,
            certification
        )

        # Calcular puntuación detallada
        detailed_scores = await self._calculate_detailed_readiness_scores(
            student_profile,
            certification
        )

        return {
            "readiness_level": readiness_level.value,
            "overall_score": sum(detailed_scores.values()) / len(detailed_scores),
            "detailed_scores": detailed_scores,
            "estimated_prep_weeks": certification.get("estimated_prep_weeks", 8),
            "recommended_actions": self._get_readiness_actions(readiness_level)
        }

    async def _identify_knowledge_gaps(
        self,
        student_profile: AEPStudentProfile,
        certification: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Identifica brechas de conocimiento.

        Args:
            student_profile: Perfil del estudiante.
            certification: Datos de la certificación.

        Returns:
            Lista de brechas identificadas.
        """
        gaps = []
        cert_areas = certification.get("knowledge_areas", [])

        for cert_area in cert_areas:
            student_area = None
            for area in student_profile.knowledge_areas:
                if area.area_name == cert_area:
                    student_area = area
                    break

            if not student_area or student_area.proficiency_level < 0.6:
                gap = {
                    "area": cert_area,
                    "current_level": student_area.proficiency_level if student_area else 0,
                    "required_level": 0.8,
                    "gap_size": 0.8 - (student_area.proficiency_level if student_area else 0),
                    "recommended_resources": self._get_resources_for_area(cert_area)
                }
                gaps.append(gap)

        return gaps

    async def _create_preparation_plan(
        self,
        student_profile: AEPStudentProfile,
        certification: Dict[str, Any],
        knowledge_gaps: List[Dict[str, Any]]
    ) -> AEPStudyPlan:
        """
        Crea un plan de preparación personalizado.

        Args:
            student_profile: Perfil del estudiante.
            certification: Datos de la certificación.
            knowledge_gaps: Brechas de conocimiento.

        Returns:
            Plan de estudio creado.
        """
        # Crear sesiones de estudio basadas en brechas
        study_sessions = []
        current_date = datetime.now()

        for gap in knowledge_gaps:
            # Crear sesiones para cerrar la brecha
            # 1-10 sesiones por brecha
            sessions_needed = max(1, int(gap["gap_size"] * 10))

            for i in range(sessions_needed):
                session = AEPStudySession(
                    session_id=f"prep_{certification['title'].replace(' ', '_').lower()}_{gap['area']}_{i}",
                    title=f"Preparación: {gap['area']} - Sesión {i+1}",
                    description=f"Trabajar en {gap['area']} para certificación {certification['title']}",
                    knowledge_area=gap["area"],
                    difficulty="intermediate",
                    estimated_duration=90,  # 1.5 horas
                    objectives=[f"Mejorar proficiency en {gap['area']}"],
                    resources=gap["recommended_resources"],
                    scheduled_date=current_date +
                    timedelta(days=i*7),  # Una sesión por semana
                    status="pending"
                )
                study_sessions.append(session)

        return AEPStudyPlan(
            plan_id=f"cert_prep_{student_profile.student_id}_{certification['title'].replace(' ', '_').lower()}",
            student_id=student_profile.student_id,
            title=f"Plan de Preparación: {certification['title']}",
            description=f"Plan personalizado para preparar {certification['title']}",
            target_certification=certification["title"],
            study_sessions=study_sessions,
            total_sessions=len(study_sessions),
            estimated_completion_weeks=len(
                study_sessions) // 2,  # 2 sesiones por semana
            created_at=datetime.now(),
            status="active"
        )

    async def _design_certification_path(
        self,
        student_profile: AEPStudentProfile,
        target_certifications: List[str],
        timeline_months: int,
        budget_constraints: Dict[str, Any]
    ) -> AEPCertificationPath:
        """
        Diseña una ruta completa de certificación.

        Args:
            student_profile: Perfil del estudiante.
            target_certifications: Certificaciones objetivo.
            timeline_months: Meses disponibles.
            budget_constraints: Restricciones de presupuesto.

        Returns:
            Ruta de certificación diseñada.
        """
        await self._refresh_certification_database_from_api()

        # Obtener datos de certificaciones objetivo
        cert_data = []
        for cert_id in target_certifications:
            cert = self._get_certification_by_id(cert_id)
            if cert:
                cert_data.append(cert)

        # Ordenar por prerrequisitos y dificultad
        ordered_certs = self._order_certifications_by_prerequisites(cert_data)

        # Crear milestones
        milestones = []
        current_date = datetime.now()
        total_cost = 0

        for i, cert in enumerate(ordered_certs):
            milestone_date = current_date + \
                timedelta(days=i * 30)  # Un mes por certificación

            milestone = {
                "milestone_id": f"milestone_{i+1}",
                "certification_id": cert["title"],
                "target_date": milestone_date,
                "prerequisites_completed": i > 0,
                "estimated_cost": cert.get("cost_usd", 0),
                "status": "pending"
            }
            milestones.append(milestone)
            total_cost += cert.get("cost_usd", 0)

        # Calcular probabilidad de éxito
        success_probability = self._calculate_success_probability(
            student_profile,
            ordered_certs,
            timeline_months
        )

        return AEPCertificationPath(
            path_id=f"path_{student_profile.student_id}_{datetime.now().strftime('%Y%m%d')}",
            student_id=student_profile.student_id,
            title=f"Ruta de Certificación: {len(target_certifications)} certificaciones",
            description="Ruta personalizada hacia objetivos de certificación",
            target_certifications=target_certifications,
            milestones=milestones,
            total_cost_estimate=total_cost,
            timeline_months=timeline_months,
            success_probability=success_probability,
            created_at=datetime.now(),
            status="active"
        )

    async def _calculate_career_impact(
        self,
        certification: Dict[str, Any],
        industry: str,
        experience_level: str
    ) -> Dict[str, Any]:
        """
        Calcula el impacto profesional de una certificación.

        Args:
            certification: Datos de la certificación.
            industry: Industria.
            experience_level: Nivel de experiencia.

        Returns:
            Análisis de impacto.
        """
        # En una implementación real, esto usaría datos de mercado
        # Aquí proporcionamos estimaciones basadas en conocimiento general

        base_salary_impact = {
            "aws-saa": {"entry": 5000, "mid": 8000, "senior": 12000},
            "azure-az900": {"entry": 3000, "mid": 5000, "senior": 7000},
            "compTIA-a+": {"entry": 2000, "mid": 4000, "senior": 6000},
            "cisco-ccna": {"entry": 4000, "mid": 7000, "senior": 10000}
        }

        cert_id = certification.get("title", "").lower().replace(" ", "-")
        salary_data = base_salary_impact.get(
            cert_id, {"entry": 3000, "mid": 5000, "senior": 7000})

        annual_increase = salary_data.get(experience_level, salary_data["mid"])

        return {
            "salary_projection": {
                "annual_increase_usd": annual_increase,
                # Asumiendo salario base de $50k
                "percentage_increase": (annual_increase / 50000) * 100,
                # Meses para recuperar inversión
                "break_even_months": certification.get("estimated_prep_weeks", 8) * 4
            },
            "market_demand": {
                "demand_level": "high" if certification.get("provider") in ["aws", "azure"] else "medium",
                "job_postings_increase": 25,
                "industry_relevance": industry
            },
            "skill_alignment": {
                "current_trends": ["cloud_computing", "automation", "security"],
                "future_proofing": 8,  # Años de relevancia
                "complementary_skills": ["linux", "networking", "programming"]
            },
            # Score basado en aumento salarial
            "career_impact_score": min(100, annual_increase // 100)
        }

    async def _calculate_path_progress(
        self,
        certification_path: AEPCertificationPath
    ) -> Dict[str, Any]:
        """
        Calcula el progreso en una ruta de certificación.

        Args:
            certification_path: Ruta de certificación.

        Returns:
            Reporte de progreso.
        """
        completed_milestones = len([
            m for m in certification_path.milestones
            if m.get("status") == "completed"
        ])

        total_milestones = len(certification_path.milestones)
        completion_percentage = (
            completed_milestones / total_milestones) * 100 if total_milestones > 0 else 0

        # Calcular milestones próximos
        upcoming_milestones = [
            m for m in certification_path.milestones
            if m.get("status") == "pending"
        ][:3]  # Próximos 3

        return {
            "path_id": certification_path.path_id,
            "completion_percentage": completion_percentage,
            "completed_milestones": completed_milestones,
            "total_milestones": total_milestones,
            "upcoming_milestones": upcoming_milestones,
            "estimated_completion_date": certification_path.created_at + timedelta(days=certification_path.timeline_months * 30),
            "on_track": completion_percentage >= (datetime.now() - certification_path.created_at).days / (certification_path.timeline_months * 30) * 100
        }

    def _filter_relevant_certifications(
        self,
        knowledge_areas: List[str],
        career_goals: List[str],
        preferred_providers: List[str],
        experience_years: int
    ) -> Dict[str, Dict[str, Any]]:
        """
        Filtra certificaciones relevantes.

        Args:
            knowledge_areas: Áreas de conocimiento.
            career_goals: Objetivos profesionales.
            preferred_providers: Proveedores preferidos.
            experience_years: Años de experiencia.

        Returns:
            Certificaciones filtradas.
        """
        relevant_certs = {}

        for cert_id, cert_data in self.certification_database.items():
            # Filtrar por proveedor preferido
            if preferred_providers and cert_data["provider"] not in preferred_providers:
                continue

            # Filtrar por nivel de experiencia apropiado
            if not self._is_experience_level_appropriate(experience_years, cert_data["level"]):
                continue

            # Verificar relevancia con áreas de conocimiento o objetivos
            cert_relevant = False

            # Verificar áreas de conocimiento
            cert_areas = cert_data.get("knowledge_areas", [])
            if any(area in knowledge_areas for area in cert_areas):
                cert_relevant = True

            # Verificar objetivos profesionales
            cert_title_lower = cert_data["title"].lower()
            if any(goal.lower() in cert_title_lower for goal in career_goals):
                cert_relevant = True

            if cert_relevant:
                relevant_certs[cert_id] = cert_data

        return relevant_certs

    def _is_experience_level_appropriate(
        self,
        experience_years: int,
        cert_level: str
    ) -> bool:
        """
        Verifica si el nivel de experiencia es apropiado para la certificación.

        Args:
            experience_years: Años de experiencia.
            cert_level: Nivel de la certificación.

        Returns:
            True si es apropiado.
        """
        level_requirements = {
            "foundational": (0, 2),
            "associate": (0, 5),
            "professional": (2, 10),
            "expert": (5, 999),
            "master": (10, 999)
        }

        min_years, max_years = level_requirements.get(cert_level, (0, 999))
        return min_years <= experience_years <= max_years

    def _check_level_appropriateness(
        self,
        experience_years: int,
        cert_level: str
    ) -> float:
        """
        Verifica qué tan apropiado es el nivel para la experiencia.

        Args:
            experience_years: Años de experiencia.
            cert_level: Nivel de la certificación.

        Returns:
            Puntuación de apropiación (0-1).
        """
        if cert_level == "foundational" and experience_years <= 1:
            return 1.0
        elif cert_level == "associate" and 0 <= experience_years <= 3:
            return 1.0
        elif cert_level == "professional" and 2 <= experience_years <= 7:
            return 1.0
        elif cert_level == "expert" and experience_years >= 5:
            return 1.0
        elif cert_level == "master" and experience_years >= 10:
            return 1.0
        else:
            return 0.5  # Parcialmente apropiado

    def _get_certification_by_id(self, cert_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene datos de certificación por ID."""
        return self.certification_database.get(cert_id)

    def _order_certifications_by_prerequisites(
        self,
        certifications: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Ordena certificaciones por prerrequisitos.

        Args:
            certifications: Lista de certificaciones.

        Returns:
            Certificaciones ordenadas.
        """
        # Implementación simple: ordenar por nivel
        level_order = {
            "foundational": 1,
            "associate": 2,
            "professional": 3,
            "expert": 4,
            "master": 5
        }

        return sorted(
            certifications,
            key=lambda x: level_order.get(x.get("level", "foundational"), 1)
        )

    def _calculate_success_probability(
        self,
        student_profile: AEPStudentProfile,
        certifications: List[Dict[str, Any]],
        timeline_months: int
    ) -> float:
        """
        Calcula probabilidad de éxito de la ruta.

        Args:
            student_profile: Perfil del estudiante.
            certifications: Certificaciones en la ruta.
            timeline_months: Meses disponibles.

        Returns:
            Probabilidad de éxito (0-1).
        """
        base_probability = 0.7  # Base 70%

        # Ajustar por experiencia previa
        experience_bonus = min(
            0.2, len(student_profile.assessments or []) * 0.05)

        # Ajustar por timeline
        weeks_needed = sum(cert.get("estimated_prep_weeks", 8)
                           for cert in certifications)
        months_needed = weeks_needed / 4.3  # Aproximadamente

        timeline_factor = min(1.0, timeline_months /
                              months_needed) if months_needed > 0 else 1.0
        timeline_adjustment = (timeline_factor - 0.5) * 0.1  # -0.1 a +0.1

        return min(0.95, max(0.3, base_probability + experience_bonus + timeline_adjustment))

    def _create_recommendation_summary(
        self,
        recommendations: List[AEPCertificationRecommendation]
    ) -> Dict[str, Any]:
        """Crea resumen de recomendaciones."""
        if not recommendations:
            return {"total": 0, "top_providers": [], "avg_prep_time": 0}

        providers = {}
        total_prep_time = 0

        for rec in recommendations:
            provider = rec.provider.value
            providers[provider] = providers.get(provider, 0) + 1
            total_prep_time += rec.estimated_prep_time

        top_providers = sorted(
            providers.items(), key=lambda x: x[1], reverse=True)[:3]

        return {
            "total": len(recommendations),
            "top_providers": top_providers,
            "avg_prep_time": total_prep_time / len(recommendations),
            "readiness_distribution": {
                level.value: len(
                    [r for r in recommendations if r.readiness_level == level])
                for level in ReadinessLevel
            }
        }

    def _calculate_overall_completion(self, progress_reports: List[Dict[str, Any]]) -> float:
        """Calcula completación general."""
        if not progress_reports:
            return 0.0
        return sum(r["completion_percentage"] for r in progress_reports) / len(progress_reports)

    def _get_upcoming_milestones(self, progress_reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Obtiene próximos milestones."""
        upcoming = []
        for report in progress_reports:
            upcoming.extend(report.get("upcoming_milestones", [])[
                            :2])  # 2 por ruta
        return upcoming[:5]  # Top 5 overall

    async def _get_career_impact_description(self, cert_data: Dict[str, Any]) -> str:
        """Obtiene descripción de impacto profesional."""
        provider = cert_data.get("provider", "")
        level = cert_data.get("level", "")

        if provider in ["aws", "azure"] and level in ["associate", "professional"]:
            return "Alto impacto: Aumento salarial significativo, mejores oportunidades laborales"
        elif level == "foundational":
            return "Impacto moderado: Base sólida para carrera, primeras oportunidades"
        else:
            return "Impacto variable: Depende de industria y experiencia específica"

    def _get_recommended_path(self, cert_data: Dict[str, Any], student_profile: AEPStudentProfile) -> List[str]:
        """Obtiene ruta recomendada."""
        path = []

        # Agregar prerrequisitos si existen
        prereqs = cert_data.get("prerequisites", [])
        path.extend(prereqs)

        # Agregar la certificación principal
        path.append(cert_data["title"])

        # Sugerir certificaciones complementarias basadas en perfil
        complementary = self._suggest_complementary_certifications(
            cert_data,
            student_profile
        )
        path.extend(complementary)

        return path

    def _suggest_complementary_certifications(
        self,
        cert_data: Dict[str, Any],
        student_profile: AEPStudentProfile
    ) -> List[str]:
        """Sugiere certificaciones complementarias."""
        suggestions = []

        provider = cert_data.get("provider")
        level = cert_data.get("level")

        # Lógica simple de sugerencias
        if provider == "aws" and level == "associate":
            suggestions.append("AWS Solutions Architect Professional")
        elif provider == "azure" and level == "foundational":
            suggestions.append("Azure Administrator Associate")

        return suggestions[:2]  # Máximo 2 sugerencias

    async def _refresh_certification_database_from_api(self) -> None:
        """
        Refresca catálogo local con datos oficiales de Microsoft Learn.

        Conserva certificaciones fallback y sobreescribe con datos del API
        cuando estén disponibles.
        """
        if self._catalog_loaded_from_api:
            return

        try:
            certs = await certifications_tool.fetch_all_certifications()
        except Exception as exc:
            self.logger.warning(
                "No se pudo consultar catálogo oficial de certificaciones: %s",
                exc,
            )
            return

        if not certs:
            self.logger.warning(
                "Catálogo oficial de certificaciones vacío. "
                "Se usa catálogo local fallback."
            )
            return

        merged_catalog = dict(self.certification_database)
        for cert in certs:
            cert_key = (cert.get("cert_id") or "").strip().lower()
            if not cert_key:
                continue

            provider = "azure"
            cert_products = [p.lower() for p in cert.get("products", [])]
            if cert_products and all("azure" not in p for p in cert_products):
                provider = "other"

            level_map = {
                "beginner": "foundational",
                "intermediate": "associate",
                "advanced": "professional",
            }
            level = level_map.get(
                cert.get("level", "intermediate"), "associate")

            merged_catalog[cert_key] = {
                "title": cert.get("name", cert_key.upper()),
                "provider": provider,
                "level": level,
                "prerequisites": [],
                "estimated_prep_weeks": 8,
                "cost_usd": 99,
                "validity_years": 3,
                "knowledge_areas": [
                    skill.lower().replace(" ", "_")
                    for skill in cert.get("skills_measured", [])[:8]
                ],
            }

        self.certification_database = merged_catalog
        self._catalog_loaded_from_api = True
        self.logger.info(
            "Catálogo oficial aplicado en CertAdvisor: %s certificaciones",
            len(certs),
        )

    async def _calculate_detailed_readiness_scores(
        self,
        student_profile: AEPStudentProfile,
        certification: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calcula puntuaciones detalladas de preparación."""
        scores = {}

        # Puntuación basada en áreas de conocimiento
        cert_areas = certification.get("knowledge_areas", [])
        knowledge_score = 0

        for cert_area in cert_areas:
            for student_area in student_profile.knowledge_areas:
                if student_area.area_name == cert_area:
                    knowledge_score += student_area.proficiency_level * 25  # 25% por área
                    break

        scores["knowledge_proficiency"] = min(100, knowledge_score)

        # Puntuación basada en evaluaciones previas
        assessment_score = min(
            100, len(student_profile.assessments or []) * 10)
        scores["assessment_experience"] = assessment_score

        # Puntuación basada en experiencia educativa
        education_score = 80 if student_profile.education_level in [
            "bachelor", "master", "phd"] else 60
        scores["education_alignment"] = education_score

        return scores

    def _get_readiness_actions(self, readiness_level: ReadinessLevel) -> List[str]:
        """Obtiene acciones recomendadas basadas en nivel de preparación."""
        actions = {
            ReadinessLevel.READY: ["Programar examen", "Repaso final"],
            ReadinessLevel.PROFICIENT: ["Práctica intensiva", "Simulacros de examen"],
            ReadinessLevel.DEVELOPING: ["Estudio estructurado", "Completar brechas de conocimiento"],
            ReadinessLevel.BEGINNING: ["Curso preparatorio", "Aprendizaje básico"],
            ReadinessLevel.NOT_READY: [
                "Desarrollo de fundamentos", "Orientación profesional"]
        }
        return actions.get(readiness_level, ["Evaluar objetivos profesionales"])

    def _get_resources_for_area(self, area: str) -> List[str]:
        """Obtiene recursos recomendados para un área."""
        resources = {
            "cloud_computing": ["AWS Free Tier", "Azure Learn", "Google Cloud Skills Boost"],
            "networking": ["Cisco Networking Academy", "Professor Messer", "Network+ Study Guide"],
            "hardware": ["CompTIA A+ Study Guide", "Professor Messer A+", "Hardware tutorials"],
            "aws_services": ["AWS Documentation", "A Cloud Guru", "Linux Academy"],
            "azure_services": ["Microsoft Learn", "Azure Documentation", "PluralSight"]
        }
        return resources.get(area, ["Documentación oficial", "Cursos en línea"])

    def _generate_basic_recommendations(self, target_cert: str, topics: List[str]) -> List[Dict[str, Any]]:
        """Genera recomendaciones básicas de certificación."""
        recommendations = []

        # Recomendaciones basadas en el target certification
        if "aws" in target_cert.lower():
            recommendations.append({
                "certification": "AWS Solutions Architect Associate",
                "provider": "AWS",
                "level": "Associate",
                "reason": "Buena base para cloud computing",
                "estimated_time": "2-3 meses"
            })
            recommendations.append({
                "certification": "AWS Developer Associate",
                "provider": "AWS",
                "level": "Associate",
                "reason": "Complementa conocimientos de desarrollo",
                "estimated_time": "2-3 meses"
            })
        elif "azure" in target_cert.lower():
            recommendations.append({
                "certification": "Azure Fundamentals",
                "provider": "Microsoft",
                "level": "Foundational",
                "reason": "Introducción a Azure",
                "estimated_time": "1 mes"
            })
            recommendations.append({
                "certification": "Azure Administrator Associate",
                "provider": "Microsoft",
                "level": "Associate",
                "reason": "Administración de Azure",
                "estimated_time": "2-3 meses"
            })
        else:
            # Recomendaciones genéricas
            recommendations.append({
                "certification": "CompTIA A+",
                "provider": "CompTIA",
                "level": "Associate",
                "reason": "Base sólida en TI",
                "estimated_time": "1-2 meses"
            })

        return recommendations

    def _calculate_basic_readiness(self, student_profile: AEPStudentProfile) -> str:
        """Calcula nivel básico de preparación."""
        assessment_count = len(student_profile.assessments or [])
        avg_proficiency = sum(area.proficiency_level for area in student_profile.knowledge_areas) / \
            len(student_profile.knowledge_areas) if student_profile.knowledge_areas else 0

        score = (assessment_count * 10) + (avg_proficiency * 50)

        if score >= 70:
            return "ready"
        elif score >= 40:
            return "developing"
        else:
            return "not_ready"

    def _get_next_steps(self, readiness_level: str) -> List[str]:
        """Obtiene próximos pasos basados en el nivel de preparación."""
        steps = {
            "ready": [
                "Programar examen de certificación",
                "Repaso final intensivo",
                "Práctica con exámenes simulados"
            ],
            "developing": [
                "Completar más evaluaciones",
                "Enfocarse en áreas de baja proficiency",
                "Estudiar conceptos específicos"
            ],
            "beginning": [
                "Completar fundamentos básicos",
                "Hacer evaluaciones iniciales",
                "Desarrollar plan de estudio estructurado"
            ]
        }

        return steps.get(readiness_level, ["Evaluar objetivos de aprendizaje"])
