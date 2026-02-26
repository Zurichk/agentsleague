"""
Modelos de datos Pydantic para AEP CertMaster.

Define las estructuras de datos principales del sistema:
perfiles de estudiante, planes de estudio, evaluaciones,
objetivos de examen y resultados de agentes.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel, Field


# ============================================================
# Enumeraciones
# ============================================================


class AEPDifficultyLevel(str, Enum):
    """Niveles de dificultad basados en Bloom's Taxonomy."""

    REMEMBER = "remember"
    UNDERSTAND = "understand"
    APPLY = "apply"
    ANALYZE = "analyze"
    EVALUATE = "evaluate"
    CREATE = "create"


class AEPStudentLevel(str, Enum):
    """Nivel de experiencia del estudiante."""

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class AEPAssessmentResult(str, Enum):
    """Resultado de una evaluación."""

    PASSED = "passed"
    FAILED = "failed"
    IN_PROGRESS = "in_progress"


class AEPWorkflowState(str, Enum):
    """Estado del flujo de trabajo del estudiante."""

    ONBOARDING = "onboarding"
    LEARNING_PATH_CURATION = "learning_path_curation"
    STUDY_PLAN_GENERATION = "study_plan_generation"
    ENGAGEMENT_SETUP = "engagement_setup"
    AWAITING_READINESS = "awaiting_readiness"
    ASSESSMENT = "assessment"
    REVIEW = "review"
    CERTIFICATION_PLANNING = "certification_planning"
    COMPLETED = "completed"


# ============================================================
# Modelos del Estudiante
# ============================================================


class AEPStudentProfile(BaseModel):
    """
    Perfil del estudiante con sus datos y preferencias.

    Attributes:
        student_id: Identificador único del estudiante.
        name: Nombre del estudiante.
        email: Correo electrónico para notificaciones.
        level: Nivel de experiencia actual.
        target_certification: Certificación objetivo.
        topics_of_interest: Temas que desea aprender.
        available_hours_per_week: Horas semanales
            disponibles para estudio.
        target_exam_date: Fecha objetivo del examen.
        preferred_language: Idioma preferido.
    """

    student_id: str = Field(
        default="", description="Identificador único"
    )
    name: str = Field(description="Nombre del estudiante")
    email: str = Field(
        default="", description="Correo electrónico"
    )
    level: AEPStudentLevel = Field(
        default=AEPStudentLevel.BEGINNER,
        description="Nivel de experiencia",
    )
    target_certification: str = Field(
        default="",
        description="Certificación objetivo (e.g. AZ-900)",
    )
    topics_of_interest: list[str] = Field(
        default_factory=list,
        description="Temas de interés del estudiante",
    )
    available_hours_per_week: float = Field(
        default=10.0,
        description="Horas semanales disponibles",
        ge=1.0,
        le=60.0,
    )
    target_exam_date: date | None = Field(
        default=None,
        description="Fecha objetivo del examen"
    )
    preferred_language: str = Field(
        default="es", description="Idioma preferido"
    )


# ============================================================
# Modelos de Rutas de Aprendizaje
# ============================================================


class AEPLearningModule(BaseModel):
    """
    Módulo individual de aprendizaje de Microsoft Learn.

    Attributes:
        title: Título del módulo.
        url: URL del módulo en Microsoft Learn.
        duration_minutes: Duración estimada en minutos.
        description: Descripción breve del módulo.
        skills_covered: Habilidades que cubre.
    """

    title: str = Field(description="Título del módulo")
    url: str = Field(default="", description="URL del módulo")
    duration_minutes: int = Field(
        default=30, description="Duración estimada en minutos"
    )
    description: str = Field(
        default="", description="Descripción breve"
    )
    skills_covered: list[str] = Field(
        default_factory=list,
        description="Habilidades cubiertas",
    )


class AEPLearningPath(BaseModel):
    """
    Ruta de aprendizaje compuesta por módulos.

    Attributes:
        path_id: Identificador de la ruta.
        title: Título de la ruta de aprendizaje.
        description: Descripción de la ruta.
        modules: Lista de módulos que componen la ruta.
        estimated_hours: Horas totales estimadas.
        relevance_score: Puntuación de relevancia (0-1).
    """

    path_id: str = Field(
        default="", description="Identificador de la ruta"
    )
    title: str = Field(description="Título de la ruta")
    description: str = Field(
        default="", description="Descripción de la ruta"
    )
    modules: list[AEPLearningModule] = Field(
        default_factory=list
    )
    estimated_hours: float = Field(
        default=0.0, description="Horas totales estimadas"
    )
    relevance_score: float = Field(
        default=0.0,
        description="Relevancia para el objetivo (0-1)",
        ge=0.0,
        le=1.0,
    )


# ============================================================
# Modelos del Plan de Estudio
# ============================================================


class AEPStudySession(BaseModel):
    """
    Sesión de estudio individual dentro del plan.

    Attributes:
        session_date: Fecha de la sesión.
        topic: Tema a estudiar.
        module_title: Título del módulo asociado.
        duration_minutes: Duración en minutos.
        objectives: Objetivos de la sesión.
        completed: Si la sesión fue completada.
    """

    session_date: date = Field(
        description="Fecha de la sesión"
    )
    topic: str = Field(description="Tema a estudiar")
    module_title: str = Field(
        default="", description="Módulo asociado"
    )
    duration_minutes: int = Field(
        default=60, description="Duración en minutos"
    )
    objectives: list[str] = Field(
        default_factory=list,
        description="Objetivos de la sesión",
    )
    completed: bool = Field(
        default=False, description="Sesión completada"
    )


class AEPMilestone(BaseModel):
    """
    Hito dentro del plan de estudio.

    Attributes:
        title: Título del hito.
        target_date: Fecha objetivo.
        description: Qué se espera lograr.
        achieved: Si el hito fue alcanzado.
    """

    title: str = Field(description="Título del hito")
    target_date: date = Field(description="Fecha objetivo")
    description: str = Field(
        default="", description="Descripción del logro"
    )
    achieved: bool = Field(
        default=False, description="Hito alcanzado"
    )


class AEPStudyPlan(BaseModel):
    """
    Plan de estudio completo generado para el estudiante.

    Attributes:
        plan_id: Identificador del plan.
        student_id: ID del estudiante asociado.
        certification: Certificación objetivo.
        start_date: Fecha de inicio.
        target_exam_date: Fecha objetivo del examen.
        sessions: Lista de sesiones de estudio.
        milestones: Hitos del plan.
        total_hours: Total de horas planificadas.
        weekly_hours: Horas semanales asignadas.
    """

    plan_id: str = Field(
        default="", description="Identificador del plan"
    )
    student_id: str = Field(
        default="", description="ID del estudiante"
    )
    certification: str = Field(
        description="Certificación objetivo"
    )
    start_date: date = Field(description="Fecha de inicio")
    target_exam_date: date = Field(
        description="Fecha objetivo del examen"
    )
    sessions: list[AEPStudySession] = Field(
        default_factory=list
    )
    milestones: list[AEPMilestone] = Field(
        default_factory=list
    )
    total_hours: float = Field(
        default=0.0, description="Horas totales"
    )
    weekly_hours: float = Field(
        default=10.0, description="Horas semanales"
    )


# ============================================================
# Modelos de Evaluación
# ============================================================


class AEPQuestionOption(BaseModel):
    """
    Opción de respuesta para una pregunta.

    Attributes:
        option_id: Identificador de la opción (A, B, C, D).
        text: Texto de la opción.
        is_correct: Si es la respuesta correcta.
    """

    option_id: str = Field(description="ID de la opción")
    text: str = Field(description="Texto de la opción")
    is_correct: bool = Field(
        default=False, description="Respuesta correcta"
    )


class AEPQuestion(BaseModel):
    """
    Pregunta de evaluación con metadatos.

    Attributes:
        question_id: Identificador de la pregunta.
        text: Enunciado de la pregunta.
        options: Opciones de respuesta.
        difficulty: Nivel de dificultad (Bloom's Taxonomy).
        topic: Tema que evalúa.
        explanation: Explicación de la respuesta correcta.
        student_answer: Respuesta seleccionada por el
            estudiante.
    """

    question_id: str = Field(
        default="", description="ID de la pregunta"
    )
    text: str = Field(description="Enunciado de la pregunta")
    options: list[AEPQuestionOption] = Field(
        default_factory=list
    )
    difficulty: AEPDifficultyLevel = Field(
        default=AEPDifficultyLevel.UNDERSTAND
    )
    topic: str = Field(
        default="", description="Tema evaluado"
    )
    explanation: str = Field(
        default="",
        description="Explicación de la respuesta correcta",
    )
    student_answer: str | None = Field(
        default=None,
        description="Respuesta del estudiante",
    )


class AEPAssessment(BaseModel):
    """
    Evaluación completa con preguntas y resultados.

    Attributes:
        assessment_id: Identificador de la evaluación.
        student_id: ID del estudiante evaluado.
        certification: Certificación evaluada.
        questions: Lista de preguntas.
        score: Puntuación obtenida (0-1).
        result: Resultado (passed/failed/in_progress).
        strengths: Áreas fuertes identificadas.
        weaknesses: Áreas débiles identificadas.
        feedback: Retroalimentación del critic agent.
        created_at: Fecha de creación.
    """

    assessment_id: str = Field(
        default="", description="ID de la evaluación"
    )
    student_id: str = Field(
        default="", description="ID del estudiante"
    )
    certification: str = Field(
        description="Certificación evaluada"
    )
    questions: list[AEPQuestion] = Field(
        default_factory=list
    )
    score: float = Field(
        default=0.0,
        description="Puntuación (0-1)",
        ge=0.0,
        le=1.0,
    )
    result: AEPAssessmentResult = Field(
        default=AEPAssessmentResult.IN_PROGRESS
    )
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    feedback: str = Field(
        default="", description="Retroalimentación"
    )
    created_at: datetime = Field(
        default_factory=datetime.now
    )


# ============================================================
# Modelos de Certificación
# ============================================================


class AEPCertification(BaseModel):
    """
    Información de una certificación Microsoft.

    Attributes:
        cert_id: Identificador (e.g. AZ-900).
        name: Nombre completo de la certificación.
        description: Descripción breve.
        level: Nivel de dificultad.
        exam_url: URL para registrar el examen.
        skills_measured: Habilidades que mide.
        estimated_study_hours: Horas de estudio estimadas.
    """

    cert_id: str = Field(
        description="ID de certificación (e.g. AZ-900)"
    )
    name: str = Field(description="Nombre completo")
    description: str = Field(default="", description="Descripción")
    level: AEPStudentLevel = Field(
        default=AEPStudentLevel.BEGINNER
    )
    exam_url: str = Field(
        default="", description="URL del examen"
    )
    skills_measured: list[str] = Field(
        default_factory=list
    )
    estimated_study_hours: float = Field(
        default=40.0,
        description="Horas de estudio estimadas",
    )


# ============================================================
# Modelos del Workflow
# ============================================================


class AEPWorkflowContext(BaseModel):
    """
    Contexto completo del flujo de trabajo del estudiante.

    Mantiene el estado actual y la historia de interacciones
    a lo largo del proceso de preparación.

    Attributes:
        student: Perfil del estudiante.
        state: Estado actual del workflow.
        learning_paths: Rutas de aprendizaje curadas.
        study_plan: Plan de estudio generado.
        assessments: Historial de evaluaciones.
        iteration_count: Número de iteraciones del ciclo.
        agent_logs: Registro de razonamiento de los agentes.
    """

    student: AEPStudentProfile = Field(
        description="Perfil del estudiante"
    )
    state: AEPWorkflowState = Field(
        default=AEPWorkflowState.ONBOARDING
    )
    learning_paths: list[AEPLearningPath] = Field(
        default_factory=list
    )
    study_plan: AEPStudyPlan | None = Field(default=None)
    assessments: list[AEPAssessment] = Field(
        default_factory=list
    )
    iteration_count: int = Field(
        default=0,
        description="Iteraciones del ciclo de mejora",
    )
    agent_logs: list[dict] = Field(
        default_factory=list,
        description="Logs de razonamiento de agentes",
    )

    def add_agent_log(
        self,
        agent_name: str,
        action: str,
        reasoning: str,
        result: str,
    ) -> None:
        """
        Registra una acción de un agente en el log.

        Args:
            agent_name: Nombre del agente.
            action: Acción realizada.
            reasoning: Razonamiento del agente.
            result: Resultado de la acción.
        """
        self.agent_logs.append(
            {
                "agent": agent_name,
                "action": action,
                "reasoning": reasoning,
                "result": result,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def get_latest_assessment(self) -> AEPAssessment | None:
        """
        Retorna la evaluación más reciente.

        Returns:
            Última evaluación o None si no hay ninguna.
        """
        if self.assessments:
            return self.assessments[-1]
        return None
