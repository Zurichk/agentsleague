"""
Dashboard Flask Simplificado - AEP CertMaster

Aplicación web simplificada usando Flask para el dashboard del sistema multi-agente.
"""

from src.agents.orchestrator_agent import (
    OrchestratorAgent as ExternalOrchestratorAgent,
)

from src.utils.guardrails import (
    validate_user_message,
    sanitize_text,
)
import json
import logging
import os
import time
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, Response, g, session, redirect, url_for, send_file
from flask_socketio import SocketIO, emit
import functools
import asyncio

# Cargar variables de entorno desde .env ANTES de cualquier inicialización
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Archivo .env cargado correctamente")
except ImportError:
    print("python-dotenv no disponible - usando variables de entorno del sistema")

# Importar métricas de agentes
try:
    from src.agents.metrics import get_metrics_collector
    print("Módulo de métricas cargado correctamente")
    print(
        f"Colector de métricas disponible: {get_metrics_collector is not None}")
except ImportError as e:
    print(f"Error importando módulo de métricas: {e}")
    get_metrics_collector = None

# Configurar sys.path para imports absolutos
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Importar sistema de logging estructurado
try:
    from src.utils.logger import (
        get_orchestrator_logger, TraceContext, log_agent_action, log_workflow_transition,
        log_performance_metrics, get_agent_logger, get_workflow_logger,
        create_trace_context, save_trace_context, get_current_trace_context,
        log_agent_response, log_error
    )
    print("Sistema de logging estructurado cargado correctamente")
except ImportError as e:
    print(f"Error importando sistema de logging estructurado: {e}")
    # Funciones dummy simplificadas

    def get_orchestrator_logger():
        return logging.getLogger("orchestrator")

    def log_agent_action(logger, agent_name, action, message, extra=None, trace_context=None):
        logger.info(f"[{agent_name}] {action}: {message}")

    def log_workflow_transition(logger, trace_context, transition_type, message, extra=None):
        logger.info(f"[{transition_type}] {message}")

    def log_performance_metrics(logger, trace_context, operation, duration, extra=None):
        logger.info(f"[PERF] {operation}: {duration:.2f}s")

    def log_agent_response(logger, agent_name, action, message, extra=None, trace_context=None):
        logger.info(f"[{agent_name}] Response: {message}")

    def log_error(logger, component, operation, message, extra=None, trace_context=None):
        logger.error(f"[{component}] {operation} ERROR: {message}")

    class TraceContext:
        def __init__(self, session_id="", user_id="", operation=""):
            self.session_id = session_id
            self.user_id = user_id
            self.operation = operation
            self.trace_id = f"{session_id}_{user_id}_{operation}"

    get_agent_logger = get_orchestrator_logger
    get_workflow_logger = get_orchestrator_logger
    create_trace_context = lambda **kwargs: TraceContext(**kwargs)
    def save_trace_context(ctx): return None
    def get_current_trace_context(): return None

# Guardrails y validaciones

# Importar herramienta de persistencia
try:
    from src.tools.persistence import persistence_tool
    print("✅ Herramienta de persistencia cargada correctamente")
except Exception as e:
    print(f"⚠️ Error cargando persistencia ({type(e).__name__}): {e}")
    # Intentar inicializar directamente como fallback
    try:
        from src.tools.persistence import PersistenceTool
        persistence_tool = PersistenceTool()
        print("✅ Persistencia inicializada por fallback")
    except Exception as e2:
        print(f"❌ Persistencia no disponible: {e2}")
        persistence_tool = None

try:
    from src.tools.email import email_tool
except Exception as e:
    print(f"⚠️ Email tool no disponible: {e}")
    email_tool = None

try:
    from src.tools.calendar import calendar_tool
except Exception as e:
    print(f"⚠️ Calendar tool no disponible: {e}")
    calendar_tool = None

# Clases para integración con Azure AI Foundry


class RealAgent:
    """Agente real que usa Azure OpenAI para ejecutar tareas (integración con Foundry vía OpenAI)."""

    def __init__(self, name: str, assistant_id: str, description: str):
        self.name = name
        # No usado por ahora, pero para compatibilidad futura
        self.assistant_id = assistant_id
        self.description = description
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """Inicializar cliente de Azure OpenAI."""
        try:
            from openai import AzureOpenAI
            import os

            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            api_version = os.getenv(
                "AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
            deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")

            if not endpoint or not api_key:
                raise RuntimeError(
                    "Azure OpenAI no configurado. Se requieren "
                    "AZURE_OPENAI_ENDPOINT y AZURE_OPENAI_API_KEY."
                )

            self.client = AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version=api_version
            )
            self.deployment = deployment
            print(
                f"✅ Cliente Azure OpenAI inicializado para agente {self.name}")
        except ImportError:
            raise RuntimeError(
                "OpenAI SDK no disponible. Instala dependencias para usar Azure OpenAI."
            )
        except Exception as e:
            raise RuntimeError(
                f"Error inicializando Azure OpenAI para {self.name}: {e}"
            )

    def execute(self, message: str, student_id: str = "demo") -> dict:
        """Ejecutar el agente real usando Azure OpenAI con system prompt avanzado."""
        start_time = time.time()
        if not self.client:
            raise RuntimeError(
                "Azure OpenAI client no inicializado para el agente."
            )

        try:
            # System prompts avanzados con Chain-of-Thought, herramientas y estructura rica
            system_prompts = {
                'curator': """Eres el CURATOR AGENT del sistema AEP CertMaster, especialista en gestionar itinerarios de aprendizaje de Microsoft Learn.

Tu punto de entrada SIEMPRE es la descripción libre del estudiante sobre lo que quiere aprender (ej: "quiero aprender inteligencia artificial", "me interesa Business Central", "quiero estudiar cloud computing"). NO asumas que el usuario ya conoce un código de certificación.

🛠️ HERRAMIENTAS QUE USAS:
- microsoft_learn_search: Búsqueda de módulos y rutas en learn.microsoft.com
- certification_catalog: Acceso al catálogo completo de certificaciones Microsoft
- relevance_scorer: Algoritmo de puntuación de relevancia (0-100) por módulo

🧠 PROCESO DE RAZONAMIENTO (Chain-of-Thought):
1. EXTRAER los temas e intereses del usuario desde su descripción libre
2. INFERIR las áreas tecnológicas relevantes (IA, cloud, datos, business apps, seguridad...)
3. BUSCAR en Microsoft Learn usando microsoft_learn_search → rutas relevantes para esos temas
4. PUNTUAR cada ruta con relevance_scorer basado en coincidencia con los intereses
5. FILTRAR: mantener solo rutas con score > 70
6. ORDENAR: de más a menos relevante
7. RECOMENDAR: top 3-5 rutas con justificación
   (Si el usuario ya mencionó una certificación específica, incorpórala como punto de referencia)

📋 FORMATO DE RESPUESTA OBLIGATORIO:
## 🔍 Análisis de tus intereses
[Resume: temas que quiere aprender, nivel detectado, áreas de interés]

## 📚 Rutas de Aprendizaje Recomendadas
Para cada ruta (mínimo 3):
**[Nombre de la ruta]** · Relevancia: XX/100 · ~XX horas
- 📖 Módulos clave: [lista de módulos específicos]
- 🎯 Cubre: [qué aprenderás en esa ruta]
- 🔗 learn.microsoft.com/training/paths/[slug]

## ⚡ Ruta Óptima Recomendada
[Explica por qué ESA ruta específica es la mejor para este usuario y sus intereses]

## 📊 Estimación de tiempo total
- Estudio: X-Y horas distribuidas en Z semanas
- Práctica en sandbox/labs: ~X horas adicionales

Sé específico con nombres reales de módulos de Microsoft Learn. No generalices.""",

                'study_plan': """Eres el STUDY PLAN AGENT del sistema AEP CertMaster, especialista en crear planes de estudio personalizados con precisión de calendario.

🛠️ HERRAMIENTAS QUE USAS:
- calendar_generator: Crea calendarios .ics con sesiones de estudio
- workload_calculator: Calcula carga semanal realista
- milestone_planner: Define hitos y fechas objetivo
- bloom_taxonomy_mapper: Mapea contenido a niveles de comprensión (Recordar→Crear)

🧠 PROCESO DE RAZONAMIENTO (Chain-of-Thought):
1. ESTIMAR horas totales necesarias según complejidad de la certificación
2. CALCULAR sesiones semanales con workload_calculator (horas disponibles del usuario)
3. DISTRIBUIR contenido usando bloom_taxonomy_mapper: empezar por Recordar/Comprender, avanzar a Aplicar/Analizar
4. DEFINIR hitos semanales concretos con milestone_planner
5. INCLUIR buffer del 20% para revisión y práctica
6. GENERAR estructura de calendario con calendar_generator

📋 FORMATO DE RESPUESTA OBLIGATORIO:
## 📅 Plan de Estudio Personalizado — [Nombre Certificación]

### Resumen Ejecutivo
- ⏱️ Duración total: X semanas
- 📆 Sesiones: X días/semana × Y horas/sesión
- 🎯 Fecha objetivo de examen: [fecha estimada]

### 🗓️ Semana a Semana
**Semana 1 — [Tema/Módulo]** | Bloom: Recordar + Comprender
- Lunes (Xh): [actividad específica + recurso]
- Miércoles (Xh): [actividad específica + recurso]
- Viernes (Xh): [mini-quiz de repaso]

[Repetir para semanas 2, 3, 4...]

### 🏁 Hitos Clave
- [ ] Semana X: Completar Módulo Y, score lab > 80%
- [ ] Semana X: Práctica exam simulacro #1, objetivo ≥ 70%
- [ ] Semana X: Examen oficial

### 💡 Técnicas de Estudio Recomendadas
[Técnicas específicas: Pomodoro, spaced repetition, active recall según el tipo de certificación]

Genera un plan que un humano pueda seguir el lunes siguiente.""",

                'engagement': """Eres el ENGAGEMENT AGENT del sistema AEP CertMaster, especialista en motivación y recordatorios de estudio.

🛠️ HERRAMIENTAS QUE USAS:
- email_reminder_service: Programa recordatorios con email
- motivation_personalizer: Adapta mensajes al perfil psicológico del estudiante

🧠 PROCESO DE RAZONAMIENTO (Chain-of-Thought):
1. EVALUAR nivel de motivación actual del estudiante (contexto de la conversación)
2. PROGRAMAR recordatorios con email_reminder_service (horarios óptimos)
3. DISEÑAR micro-retos semanales de aprendizaje (sin mecánicas de juego)
4. PERSONALIZAR mensajes con motivation_personalizer

📋 FORMATO DE RESPUESTA OBLIGATORIO:
## 🎯 Sistema de Motivación y Recordatorios

### 🧭 Tu Perfil de Seguimiento
- Nivel de avance actual: [Inicio/Intermedio/Avanzado]
- Objetivo inmediato: [objetivo semanal]
- Riesgo de abandono: [Bajo/Medio/Alto] + recomendación

### 📧 Recordatorios Programados
[Herramienta: email_reminder_service]
- Lunes 08:00 — "Inicio de semana de estudio"
- Miércoles 19:00 — "Sesión de práctica"
- Domingo 10:00 — "Repaso semanal + mini-quiz"
- [reminder personalizado según horario del usuario]

### ⚡ Retos Semanales
**Semana 1:** [reto específico de estudio]
**Semana 2:** [reto más avanzado]
...

### 📌 Seguimiento de Constancia
- Días de estudio esta semana: [número]
- Recomendación para mantener ritmo: [acción concreta]

### 💬 Mensaje Motivacional Personalizado
[Mensaje específico para este usuario según los temas que quiere aprender y su situación actual]

Muy importante: programa los recordatorios de email usando email_reminder_service.
Sé entusiasta, concreto y accionable.""",

                'assessment': """Eres el ASSESSMENT AGENT del sistema AEP CertMaster, especialista en evaluaciones adaptativas basadas en Taxonomía de Bloom.

🛠️ HERRAMIENTAS QUE USAS:
- question_generator: Genera preguntas a diferentes niveles de Bloom
- knowledge_mapper: Mapea dominios de conocimiento del examen oficial
- adaptive_engine: Ajusta dificultad según respuestas previas
- scoring_rubric: Evalúa calidad de respuestas con rúbricas detalladas

🧠 PROCESO DE RAZONAMIENTO (Chain-of-Thought):
1. DERIVAR el dominio a evaluar a partir de las rutas de aprendizaje curadas previamente
   (el Curator Agent ya identificó los temas — usa ese contexto, NO pidas al usuario que repita la cert)
2. MAPEAR objetivos del examen oficial con knowledge_mapper según esos temas
3. GENERAR preguntas en niveles Bloom: 30% Recordar, 30% Comprender, 25% Aplicar, 15% Analizar
4. EVALUAR respuestas con scoring_rubric (score 0-100)
5. IDENTIFICAR gaps de conocimiento específicos
6. RECOMENDAR recursos de refuerzo para áreas débiles

📋 FORMATO DE RESPUESTA OBLIGATORIO:
## 📝 Evaluación Adaptativa — [Dominio/Certificación]

### 🎯 Objetivo de Evaluación
[Qué competencias específicas del examen estamos midiendo]

### 📊 Preguntas de Evaluación
[Genera 5 preguntas según niveles Bloom con este formato:]

**Pregunta 1** [Nivel: Recordar | Dominio: [dominio del examen]]
[Pregunta]
a) [opción]  b) [opción]  c) [opción]  d) [opción]
> 💡 Respuesta correcta: [letra] — [explicación breve del porqué]

**Pregunta 2** [Nivel: Comprender | Dominio: [dominio]]
...

**Pregunta 3** [Nivel: Aplicar | Caso práctico]
...

### 📈 Análisis de Resultados
- Score estimado de ready-for-exam: XX/100
- Áreas fuertes: [dominios]
- Áreas a reforzar: [dominios específicos]

### 🎓 Veredicto
[LISTO PARA EXAMEN / NECESITA REFUERZO] + justificación detallada
[Si necesita refuerzo: recursos específicos de Microsoft Learn para cada área débil]

Genera contenido de evaluación de calidad similar al examen oficial.""",

                'critic': """Eres el CRITIC AGENT del sistema AEP CertMaster, responsable de validación, análisis crítico y mejora continua.

🛠️ HERRAMIENTAS QUE USAS:
- quality_scorer: Evalúa la calidad de planes y respuestas (0-100)
- gap_analyzer: Identifica brechas entre estado actual y objetivo
- benchmark_comparator: Compara con estándares de la industria
- improvement_suggester: Genera recomendaciones accionables priorizadas

🧠 PROCESO DE RAZONAMIENTO CRÍTICO (Critic Pattern):
1. REVISAR el plan/respuesta/evaluación recibida de otros agentes
2. APLICAR quality_scorer: ¿cubre todos los dominios del examen? ¿es realista el timeline?
3. EJECUTAR gap_analyzer: ¿qué falta? ¿qué está sobredimensionado?
4. COMPARAR con benchmark_comparator: ¿está alineado con candidatos exitosos?
5. GENERAR mejoras concretas con improvement_suggester (no genéricas)
6. EMITIR veredicto: APROBADO / NECESITA REVISIÓN / RECHAZADO

📋 FORMATO DE RESPUESTA OBLIGATORIO:
## 🔍 Análisis Crítico

### ✅ Fortalezas Identificadas
[Puntos concretos bien logrados, con referencia al estándar que los valida]

### ⚠️ Brechas Identificadas
[Herramienta: gap_analyzer]
1. **[Brecha crítica]**: [descripción específica] → Impacto: ALTO/MEDIO/BAJO
2. [...]

### 📊 Puntuación de Calidad
[Herramienta: quality_scorer]
- Cobertura de dominios del examen: XX/100
- Realismo del timeline: XX/100
- Especificidad de recursos: XX/100
- **Score global: XX/100**

### 🔧 Mejoras Recomendadas (priorizadas)
1. [Mejora #1 — acción concreta, implementable en X días]
2. [Mejora #2 ...]
3. [...]

### 🏁 Veredicto Final
**[APROBADO ✅ / NECESITA REVISIÓN ⚠️ / RECHAZADO ❌]**
[Justificación de 2-3 líneas]

Sé riguroso pero constructivo. Tu objetivo es elevar la calidad del aprendizaje.""",

                'cert_advisor': """Eres el CERT ADVISOR AGENT del sistema AEP CertMaster, experto estratégico en el ecosistema de certificaciones Microsoft.

🛠️ HERRAMIENTAS QUE USAS:
- certification_roadmap: Acceso al mapa completo de certificaciones Microsoft 2024-2025
- career_path_analyzer: Analiza el perfil profesional y sugiere certificaciones óptimas
- exam_scheduler: Busca disponibilidad de exámenes en Pearson VUE / Certiport
- roi_calculator: Calcula retorno de inversión esperado de cada certificación

🧠 PROCESO DE RAZONAMIENTO ESTRATÉGICO:
1. ANALIZAR perfil del usuario: rol actual, experiencia, objetivos career
2. MAPEAR certificaciones relevantes con certification_roadmap
3. CALCULAR ROI con roi_calculator (salary impact, job market demand)
4. DETERMINAR secuencia óptima (Fundamentals → Associate → Expert)
5. VERIFICAR disponibilidad con exam_scheduler
6. ENTREGAR roadmap personalizado multi-certificación

📋 FORMATO DE RESPUESTA OBLIGATORIO:
## 🎓 Advisory de Certificaciones Microsoft

### 🔎 Análisis de tu Perfil Profesional
[Herramienta: career_path_analyzer]
- Rol objetivo: [rol]
- Stack tecnológico relevante: [tecnologías]
- Nivel de experiencia: [Principiante/Intermedio/Avanzado]

### 🗺️ Roadmap de Certificaciones Recomendado
[Herramienta: certification_roadmap]

**Inmediata (0-3 meses):**
📌 **[Cert Name (Código)]** — [Nivel: Fundamentals/Associate/Expert]
- ¿Por qué esta primero?: [razón estratégica]
- Dominios del examen: [lista de áreas]
- Costo: ~$165 USD | Renovación: anual (sin coste adicional)
- Valor de mercado: [dato de demanda laboral]

**Siguiente paso (3-6 meses):**
📌 **[Cert Name (Código)]**
...

### 💰 ROI Estimado
[Herramienta: roi_calculator]
- Incremento salarial promedio post-certificación: +X%
- Demanda laboral actual: X ofertas activas en LinkedIn/job boards
- Tiempo de recuperación de inversión: X meses

### 📅 Plan de Acción Inmediata
1. Registro en Microsoft Learn: learn.microsoft.com/certifications
2. Programar examen en Pearson VUE: [instrucciones]
3. Activar study plan del agente de preparación

Proporciona datos reales del mercado y rutas certificadas comprobadas."""
            }

            system_prompt = system_prompts.get(
                self.name, f"Eres un agente especializado en {self.description}. Proporciona respuestas estructuradas, detalladas y con valor real.")

            # Inyectar contexto operativo y estándar de calidad en el mensaje
            enriched_message = (
                f"{message}\n\n"
                f"[Contexto del sistema: Student ID = {student_id}. "
                "Responde en español. Usa el formato obligatorio de tu rol. "
                "Usa toda tu capacidad y todo el contexto disponible. "
                "Evita respuestas genéricas y prioriza precisión, profundidad y accionabilidad.]"
            )

            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": enriched_message}
                ],
                max_tokens=2000,
                temperature=0.5
            )

            result = response.choices[0].message.content.strip()

            # Capturar información de tokens de uso
            usage = response.usage
            tokens_info = {
                'prompt_tokens': usage.prompt_tokens,
                'completion_tokens': usage.completion_tokens,
                'total_tokens': usage.total_tokens
            }

            response_time = time.time() - start_time

            if get_metrics_collector:
                collector = get_metrics_collector()
                collector.record_agent_call(
                    agent_name=self.name,
                    response_time=response_time,
                    tokens_used=tokens_info['total_tokens'],
                    prompt_tokens=tokens_info['prompt_tokens'],
                    completion_tokens=tokens_info['completion_tokens'],
                    success=True,
                    quality_score=None
                )

            agent_labels = {
                'curator': '🏛️ Curator Agent',
                'study_plan': '📚 Study Plan Agent',
                'engagement': '🎯 Engagement Agent',
                'assessment': '📝 Assessment Agent',
                'critic': '🔍 Critic Agent',
                'cert_advisor': '🎓 Cert Advisor Agent'
            }
            agent_tools = {
                'curator': ['microsoft_learn_search', 'certification_catalog', 'relevance_scorer'],
                'study_plan': ['calendar_generator', 'workload_calculator', 'milestone_planner', 'bloom_taxonomy_mapper'],
                'engagement': ['email_reminder_service', 'motivation_personalizer'],
                'assessment': ['question_generator', 'knowledge_mapper', 'adaptive_engine', 'scoring_rubric'],
                'critic': ['quality_scorer', 'gap_analyzer', 'benchmark_comparator', 'improvement_suggester'],
                'cert_advisor': ['certification_roadmap', 'career_path_analyzer', 'exam_scheduler', 'roi_calculator']
            }

            print(
                f"✅ Agente {self.name} ejecutado exitosamente con Azure OpenAI")
            print(f"📊 Tokens utilizados: {tokens_info}")

            return {
                'response': result,
                'tokens': tokens_info,
                'agent_name': agent_labels.get(self.name, self.name),
                'tools_used': agent_tools.get(self.name, []),
                'mode': 'azure_openai'
            }

        except Exception as e:
            print(f"Error ejecutando agente {self.name}: {e}")
            if get_metrics_collector:
                collector = get_metrics_collector()
                collector.record_agent_call(
                    agent_name=self.name,
                    response_time=time.time() - start_time,
                    tokens_used=0,
                    prompt_tokens=0,
                    completion_tokens=0,
                    success=False,
                    quality_score=None
                )
            raise


class OrchestratorAgent:
    """Agente coordinador principal que maneja el flujo completo con trazabilidad completa."""

    def __init__(self):
        # Configurar logger estructurado
        self.logger = get_orchestrator_logger(
        ) if get_orchestrator_logger else logging.getLogger(__name__)

        # Forzar Azure OpenAI como único modo soportado
        import os

        azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT', '')
        azure_key = os.getenv('AZURE_OPENAI_API_KEY', '')
        if not azure_endpoint or not azure_key:
            raise RuntimeError(
                "Azure OpenAI no configurado. Define AZURE_OPENAI_ENDPOINT y "
                "AZURE_OPENAI_API_KEY para iniciar el dashboard."
            )

        print("🔵 Modo único habilitado: Azure OpenAI")

        agent_configs = {
            'curator': 'Especialista en gestionar itinerarios de aprendizaje',
            'study_plan': 'Especialista en crear planes de estudio',
            'engagement': 'Especialista en motivación y recordatorios',
            'assessment': 'Especialista en evaluaciones',
            'critic': 'Especialista en validación',
            'cert_advisor': 'Especialista en certificaciones'
        }

        self.agents = {}
        for name, description in agent_configs.items():
            self.agents[name] = RealAgent(name, "", description)
            print(f"✅ Agente {name} configurado con Azure OpenAI")

        self.conversation_state = {}
        self.interaction_logs = []
        # SID del socket activo (actualizado por el handler)
        self._active_sid: str = ''

        log_agent_action(self.logger, "OrchestratorAgent", "initialization",
                         "Orchestrator initialized", "Mode: azure_openai")

    def _emit_agent_active(self, agent_key: str, agent_label: str) -> None:
        """Emite evento agent_active al cliente WebSocket activo (si hay conexión)."""
        try:
            from flask_socketio import emit as sio_emit
            sio_emit('agent_active', {
                'agent_name': agent_label,
                'agent_key': agent_key
            })
        except RuntimeError:
            # Fuera de contexto de request; intentar con socketio global
            try:
                if self._active_sid:
                    socketio.emit('agent_active', {
                        'agent_name': agent_label,
                        'agent_key': agent_key
                    }, to=self._active_sid)
            except Exception:
                pass  # No bloqueamos el flujo si no hay socket disponible

    def _emit_partial_response(self, content: str, loading_next: str = '') -> None:
        """Emite un bloque de respuesta parcial al cliente (streaming por agente).

        Permite al usuario ver la respuesta de cada agente en tiempo real,
        sin esperar a que termine el sub-workflow completo.
        """
        try:
            payload = {'content': content,
                       'loading_next': loading_next, 'partial': True}
            from flask_socketio import emit as sio_emit
            sio_emit('partial_response', payload)
        except RuntimeError:
            try:
                if self._active_sid:
                    socketio.emit('partial_response', payload,
                                  to=self._active_sid)
            except Exception:
                pass  # No bloqueamos el flujo si no hay socket disponible

    def _is_affirmative(self, message: str) -> bool:
        """
        Usa el LLM para determinar si el mensaje del usuario es una respuesta
        afirmativa a una pregunta de confirmacion HITL.
        Si el LLM no está disponible o falla, devuelve False.
        """
        import os as _os
        import json as _json
        endpoint = _os.getenv('AZURE_OPENAI_ENDPOINT', '')
        api_key = _os.getenv('AZURE_OPENAI_API_KEY', '')

        if not (endpoint and api_key):
            return False

        try:
            from openai import AzureOpenAI as _AOAI
            _client = _AOAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version=_os.getenv(
                    'AZURE_OPENAI_API_VERSION', '2024-02-15-preview'),
            )
            resp = _client.chat.completions.create(
                model=_os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', 'gpt-4o'),
                messages=[
                    {"role": "system", "content": (
                        "Determina si el mensaje del usuario es una respuesta AFIRMATIVA "
                        "(quiere continuar, dice que si, acepta, muestra interes positivo) o NEGATIVA. "
                        "Responde UNICAMENTE con JSON: {\"affirmative\": true} o {\"affirmative\": false}"
                    )},
                    {"role": "user", "content": f"Mensaje: \"{message}\""}
                ],
                max_tokens=20,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            result = _json.loads(resp.choices[0].message.content.strip())
            return bool(result.get('affirmative', False))
        except Exception:
            return False

    def _is_assessment_submission(self, message: str) -> bool:
        """
        Usa el LLM para detectar si el mensaje contiene respuestas concretas
        a una evaluación ya presentada (en lugar de una solicitud de evaluar).
        """
        import os as _os
        import json as _json

        endpoint = _os.getenv('AZURE_OPENAI_ENDPOINT', '')
        api_key = _os.getenv('AZURE_OPENAI_API_KEY', '')
        if not (endpoint and api_key):
            return False

        try:
            from openai import AzureOpenAI as _AOAI
            _client = _AOAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version=_os.getenv(
                    'AZURE_OPENAI_API_VERSION', '2024-02-15-preview'),
            )
            resp = _client.chat.completions.create(
                model=_os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', 'gpt-4o'),
                messages=[
                    {"role": "system", "content": (
                        "Determina si el mensaje del usuario contiene respuestas "
                        "a una evaluación/cuestionario ya mostrada. "
                        "Responde SOLO JSON: {\"is_submission\": true} "
                        "o {\"is_submission\": false}."
                    )},
                    {"role": "user", "content": f"Mensaje: \"{message}\""}
                ],
                max_tokens=20,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            result = _json.loads(resp.choices[0].message.content.strip())
            return bool(result.get('is_submission', False))
        except Exception:
            return False

    def _detect_intent_with_llm(self, message: str, state: dict) -> dict:
        """
        Usa el LLM para clasificar la intención del usuario en lugar de regex/keywords.
        Esto implementa el patrón Planner–Executor del README: el Orchestrator
        razona primero sobre qué agente debe actuar antes de delegar.

        Returns dict con:
            intent      : preparation | assessment | certification | confirm |
                          study_plan | greeting | other
            confidence  : float 0-1
            topic       : string con el tema detectado (si aplica)
        """
        # Intentar con LLM si hay cliente disponible
        try:
            import os as _os
            import json as _json
            endpoint = _os.getenv('AZURE_OPENAI_ENDPOINT', '')
            api_key = _os.getenv('AZURE_OPENAI_API_KEY', '')
            if not (endpoint and api_key):
                return {'intent': 'other', 'confidence': 0.0, 'topic': ''}

            from openai import AzureOpenAI as _AzureOpenAI
            _client = _AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version=_os.getenv(
                    'AZURE_OPENAI_API_VERSION', '2024-02-15-preview'),
            )
            deployment = _os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', 'gpt-4o')

            current_phase = state.get('phase', 'initial')
            system_prompt = (
                "Eres el módulo de detección de intención del Orchestrator de AEP CertMaster. "
                "Tu única tarea es clasificar el mensaje del usuario en UNA sola categoría. "
                "Responde EXCLUSIVAMENTE con JSON válido, sin markdown ni texto extra. "
                "Formato: {\"intent\": \"<categoria>\", \"confidence\": <0.0-1.0>, \"topic\": \"<tema detectado o vacío>\"}\n\n"
                "Categorías posibles:\n"
                "- preparation  → El usuario quiere aprender, estudiar o prepararse para un tema (puede ser vago como 'me interesa la nube' o específico 'quiero AZ-900')\n"
                "- assessment   → El usuario quiere ser evaluado, hacer un test, quiz o recibir feedback sobre sus conocimientos\n"
                "- certification → El usuario pregunta qué certificación hacer, quiere el catálogo, orientación profesional o roadmap de certificaciones\n"
                "- confirm      → El usuario confirma continuar, dice sí/ok/adelante/comenzar en respuesta a una pregunta\n"
                "- study_plan   → El usuario quiere ver o revisar su plan de estudio ya generado\n"
                "- greeting     → Saludo puro, reinicio de conversación o mensaje sin intención de acción\n"
                "- other        → No encaja en ninguna de las anteriores\n\n"
                f"Contexto actual del sistema: fase='{current_phase}'"
            )

            resp = _client.chat.completions.create(
                model=deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Mensaje del usuario: \"{message}\""}
                ],
                max_tokens=80,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content.strip()
            result = _json.loads(raw)
            intent = result.get('intent', 'other')
            # Validar que sea una categoría conocida
            valid = {'preparation', 'assessment', 'certification', 'confirm',
                     'study_plan', 'greeting', 'other'}
            if intent not in valid:
                intent = 'other'
            print(
                f"[INTENT-LLM] '{message[:60]}' → {intent} ({result.get('confidence', '?')})")
            return {
                'intent': intent,
                'confidence': float(result.get('confidence', 0.8)),
                'topic': result.get('topic', ''),
            }

        except Exception as e:
            print(f"[INTENT-LLM] Error de clasificación: {e}")
            return {'intent': 'other', 'confidence': 0.0, 'topic': ''}

    @staticmethod
    def _build_structured_study_plan(
        plan_id: str,
        student_id: str,
        certification: str,
        study_plan_response: str,
        plan_name: str = "",
    ) -> dict:
        """
        Convierte la respuesta textual del Study Plan Agent en un plan estructurado.

        Este formato alimenta la vista `/study-plan` y el calendario `.ics`.
        """
        import re

        text = study_plan_response or ""
        day_map = {
            'lunes': 0,
            'martes': 1,
            'miércoles': 2,
            'miercoles': 2,
            'jueves': 3,
            'viernes': 4,
            'sábado': 5,
            'sabado': 5,
            'domingo': 6,
        }

        today = datetime.now().date()
        days_to_monday = (7 - today.weekday()) % 7
        next_monday = today + timedelta(days=days_to_monday)

        week_number = 1
        week_title = f"Semana {week_number}"
        sessions = []
        milestones = []

        week_re = re.compile(
            r'^\s*Semana\s+(\d+)\s*[—\-:]\s*(.*)$', re.IGNORECASE)
        day_re = re.compile(
            r'^\s*(Lunes|Martes|Mi[eé]rcoles|Jueves|Viernes|S[áa]bado|Domingo)\s*\((\d+)\s*h\)\s*:\s*(.+)$',
            re.IGNORECASE,
        )

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            week_match = week_re.match(line)
            if week_match:
                week_number = int(week_match.group(1))
                week_title = week_match.group(
                    2).strip() or f"Semana {week_number}"
                continue

            day_match = day_re.match(line)
            if day_match:
                day_name = day_match.group(1)
                hours = int(day_match.group(2))
                activity = day_match.group(3).strip()
                day_offset = day_map.get(day_name.lower(), 0)
                session_date = next_monday + timedelta(
                    days=(week_number - 1) * 7 + day_offset
                )
                sessions.append({
                    'session_id': f"{plan_id}_w{week_number}_{len(sessions) + 1}",
                    'session_date': session_date.strftime('%Y-%m-%d'),
                    'topic': activity,
                    'module_title': week_title,
                    'duration_minutes': hours * 60,
                    'objectives': [],
                    'completed': False,
                    'learn_url': '',
                })
                continue

            lower_line = line.lower()
            if lower_line.startswith('semana ') and ':' in line:
                milestones.append({
                    'title': line.split(':', 1)[0].strip(),
                    'description': line.split(':', 1)[1].strip(),
                    'target_date': '',
                    'achieved': False,
                })

        if not sessions:
            default_days = [0, 2, 4]
            for idx, day_offset in enumerate(default_days, start=1):
                session_date = next_monday + timedelta(days=day_offset)
                sessions.append({
                    'session_id': f"{plan_id}_s{idx}",
                    'session_date': session_date.strftime('%Y-%m-%d'),
                    'topic': f"Sesión {idx} de estudio",
                    'module_title': certification,
                    'duration_minutes': 120,
                    'objectives': [],
                    'completed': False,
                    'learn_url': '',
                })

        weekly_minutes = {}
        for session in sessions:
            session_date = datetime.strptime(
                session['session_date'], '%Y-%m-%d').date()
            week_idx = ((session_date - next_monday).days // 7) + 1
            weekly_minutes[week_idx] = weekly_minutes.get(week_idx, 0) + int(
                session.get('duration_minutes', 60)
            )
        weekly_hours = round(
            (sum(weekly_minutes.values()) / max(len(weekly_minutes), 1)) / 60)
        total_weeks = max(weekly_minutes.keys()) if weekly_minutes else 1
        target_exam_date = next_monday + timedelta(weeks=total_weeks)

        return {
            'plan_id': plan_id,
            'student_id': student_id,
            'certification': certification,
            'plan_name': plan_name or f"{certification} · {next_monday.strftime('%d/%m/%Y')}",
            'start_date': next_monday.strftime('%Y-%m-%d'),
            'target_exam_date': target_exam_date.strftime('%Y-%m-%d'),
            'weekly_hours': max(weekly_hours, 1),
            'sessions': sessions,
            'milestones': milestones,
            'study_plan_response': study_plan_response,
            'created_at': datetime.now().isoformat(),
        }

    def _execute_assessment_questionnaire(
        self,
        context: str,
        student_id: str,
        trace_context: TraceContext,
        study_context: str = None,
    ) -> dict:
        """
        Ejecuta solo el paso de generación de preguntas del Assessment Agent.

        No invoca Critic ni emite veredicto final.
        """
        if study_context:
            enriched_context = (
                "CONTEXTO DEL ESTUDIANTE (usa esto para generar todas las preguntas):\n"
                f"{study_context}\n\n"
                f"Solicitud del estudiante: {context}"
            )
        else:
            enriched_context = context

        log_agent_action(
            self.logger,
            "OrchestratorAgent",
            "delegation",
            "Delegating to AssessmentAgent for questionnaire generation",
            {"target_agent": "assessment"},
            trace_context,
        )

        self._emit_agent_active('assessment', '📝 Assessment Agent')
        ass_start = time.time()
        ass_result = self.agents['assessment'].execute(
            enriched_context, student_id)
        ass_time = time.time() - ass_start
        ass_result = ass_result if isinstance(ass_result, dict) else {
            'response': ass_result, 'agent_name': 'Assessment Agent', 'tools_used': []}

        log_agent_response(
            self.logger,
            "AssessmentAgent",
            "assessment_questionnaire_ready",
            f"Assessment questionnaire ready ({len(ass_result['response'])} chars, {ass_time:.2f}s)",
            {"tokens": ass_result.get('tokens', {})},
            trace_context,
        )

        ass_block = self._fmt_agent_block(ass_result, 1, 1)
        response = (
            "🧠 **Evaluación iniciada** (1/1 agente)\n\n"
            + ass_block +
            "\n\n---\n"
            "📝 Responde las preguntas en el panel de evaluación y envía tus respuestas. "
            "Cuando las envíes, haré la corrección completa con Critic Agent."
        )

        return {
            'response': response,
            'phase': 'assessment_pending',
            'tokens': ass_result.get('tokens', {}),
        }

    def execute(self, message: str, student_id: str = "demo") -> dict:
        """
        Coordina la conversación de manera conversacional con logging estructurado completo.
        Analiza cada mensaje del usuario y responde apropiadamente con trazabilidad total.
        """
        start_time = time.time()

        # Crear trace context para esta sesión completa
        trace_context = TraceContext(
            session_id=f"session_{student_id}_{int(start_time)}",
            user_id=student_id,
            operation="orchestrator_workflow"
        )

        # Inicializar estado si no existe
        if student_id not in self.conversation_state:
            self.conversation_state[student_id] = {
                'phase': 'initial',
                'last_action': None,
                'context': {}
            }

        state = self.conversation_state[student_id]

        # Acumulador de tokens para toda la interacción
        accumulated_tokens = {'prompt_tokens': 0,
                              'completion_tokens': 0, 'total_tokens': 0}

        # Log inicial de la sesión
        log_workflow_transition(
            self.logger, trace_context, "session_start",
            f"Starting orchestrator workflow for user {student_id}",
            {"initial_phase": state['phase'], "user_message": message}
        )

        try:
            # -------------------------------------------------------
            # HITL: El usuario elige la certificación/ruta
            # -------------------------------------------------------
            if state.get('next_step') == 'await_certification_choice':
                import re as _cert_re
                # Intentar extraer código de certificación del mensaje
                # (maneja "hagamos el plan para MB-800", "quiero MB-800", etc.)
                _cert_match = _cert_re.search(
                    r'\b([A-Z]{2,3}-\d{3,4}[A-Z0-9]?)\b',
                    message, _cert_re.IGNORECASE
                )
                chosen = (
                    _cert_match.group(1).upper()
                    if _cert_match
                    else message.strip()
                )
                state['chosen_certification'] = chosen
                state['next_step'] = None

                log_workflow_transition(
                    self.logger, trace_context, "certification_chosen",
                    f"User selected certification: {chosen}",
                    {"chosen": chosen}
                )

                # Ejecutar Study Plan + Engagement directamente sin pausa HITL
                result = self._execute_sub_workflow_1_continue(
                    chosen, student_id, state.get('curator_data'), trace_context)
                response = result['response']
                state['phase'] = result.get('phase', 'preparation_complete')
                state['next_step'] = 'await_assessment_confirmation'

                if persistence_tool:
                    try:
                        _cert_slug = chosen.replace(' ', '_').lower()[:30]
                        plan_id = f"plan_{student_id}_{_cert_slug}"
                        structured_plan = self._build_structured_study_plan(
                            plan_id=plan_id,
                            student_id=student_id,
                            certification=chosen,
                            study_plan_response=result.get('response', ''),
                            plan_name=f"{chosen} · {datetime.now().strftime('%d/%m/%Y')}",
                        )
                        persistence_tool.save_study_plan(
                            plan_id,
                            student_id,
                            structured_plan,
                        )
                        state['current_plan_id'] = plan_id
                    except Exception as _pe:
                        print(f"[PERSISTENCE] Error guardando plan: {_pe}")

                self._save_interaction_to_db(
                    student_id=student_id,
                    user_msg=message,
                    assistant_msg=response,
                    phase=state['phase'],
                    session_id=trace_context.trace_id,
                    tokens=result.get('tokens', {}),
                    agent_name='StudyPlanAgent',
                )

                return {
                    'response': response,
                    'logs': [],
                    'phase': state['phase'],
                    'trace_id': trace_context.trace_id,
                    'tokens': result.get('tokens', {}),
                    'mode': 'azure_openai'
                }

            # HITL: await_cert_advisor_confirmation
            # El assessment fue APROBADO. El usuario decide si quiere la recomendacion
            # del Cert Advisor Agent. Interceptado antes del routing de intents.
            if state.get('next_step') == 'await_cert_advisor_confirmation':
                if self._is_affirmative(message):
                    state['next_step'] = None
                    log_agent_action(self.logger, "OrchestratorAgent", "delegation",
                                     "Delegating to CertAdvisorAgent from post-assessment HITL",
                                     {"target_agent": "cert_advisor"}, trace_context)
                    # Enriquecer contexto con lo que ya sabe el sistema
                    _cert_ctx = (
                        f"El estudiante ha completado exitosamente una evaluaci\xf3n de:\n"
                        f"Certificaci\xf3n/ruta: {state.get('chosen_certification', 'Business Central Developer')}\n\n"
                        f"Rutas curadas previas:\n"
                        f"{state.get('curator_data', {}).get('response', '') if state.get('curator_data') else ''}\n\n"
                        f"Solicita recomendaci\xf3n del examen Microsoft m\xe1s adecuado para continuar su progreso."
                    )
                    cert_result = self.agents['cert_advisor'].execute(
                        _cert_ctx, student_id)
                    cert_response = cert_result['response'] if isinstance(
                        cert_result, dict) else cert_result
                    log_agent_response(self.logger, "CertAdvisorAgent", "certification_advice",
                                       f"Advice received ({len(cert_response)} chars)", {}, trace_context)
                    response = (
                        f"\U0001f393 **Recomendaciones de Certificaci\xf3n**\n\n{cert_response}\n\n"
                        f"\xbfQuieres prepararte para alguna de estas certificaciones? Escr\xedbeme el nombre y empezamos."
                    )
                    state['phase'] = 'certification'
                    _cert_tokens = cert_result.get('tokens', accumulated_tokens) if isinstance(
                        cert_result, dict) else accumulated_tokens
                    accumulated_tokens = self._sum_tokens(
                        accumulated_tokens, cert_result)
                    execution_time = time.time() - start_time
                    log_performance_metrics(self.logger, trace_context,
                                            "orchestrator_workflow", execution_time,
                                            {"phase": state['phase']})
                    # Esperar a que el usuario escriba la certificación elegida
                    state['next_step'] = 'await_certification_choice'
                    # Persistir antes del return temprano
                    self._save_interaction_to_db(
                        student_id=student_id,
                        user_msg=message,
                        assistant_msg=response,
                        phase=state['phase'],
                        session_id=trace_context.trace_id,
                        tokens=_cert_tokens,
                        agent_name='CertAdvisorAgent',
                    )
                    return {
                        'response': response,
                        'logs': [],
                        'phase': state['phase'],
                        'trace_id': trace_context.trace_id,
                        'tokens': _cert_tokens,
                        'mode': 'azure_openai'
                    }
                else:
                    state['next_step'] = None
                    _neg_resp = "Entendido. Cuando quieras conocer tu hoja de ruta de certificaci\xf3n, escr\xedbeme **'certificaci\xf3n'**. \U0001f393"
                    execution_time = time.time() - start_time
                    log_performance_metrics(self.logger, trace_context,
                                            "orchestrator_workflow", execution_time, {})
                    # Persistir antes del return temprano
                    self._save_interaction_to_db(
                        student_id=student_id,
                        user_msg=message,
                        assistant_msg=_neg_resp,
                        phase=state['phase'],
                        session_id=trace_context.trace_id,
                    )
                    return {
                        'response': _neg_resp,
                        'logs': [],
                        'phase': state['phase'],
                        'trace_id': trace_context.trace_id,
                        'tokens': {},
                        'mode': 'azure_openai'
                    }

            # Compatibilidad retroactiva: si quedó un estado legacy de self_reflection,
            # se limpia para continuar con el enrutamiento normal por intención.
            if state.get('next_step') == 'self_reflection':
                state['next_step'] = None
                state['phase'] = 'ready'

            # HITL: await_assessment_confirmation
            # El Sub-Workflow 1 acaba de terminar y pregunto al usuario si quiere evaluarse.
            # Interceptamos aqui para no caer en el routing de intents.
            if state.get('next_step') == 'await_assessment_confirmation':
                _hitl_intent = self._detect_intent_with_llm(message, state)
                if self._is_affirmative(message) or _hitl_intent.get('intent') == 'assessment':
                    state['next_step'] = None
                    log_workflow_transition(self.logger, trace_context, "subworkflow_start",
                                            "Starting assessment questionnaire from post-preparation HITL",
                                            {"trigger": "await_assessment_confirmation"})
                    _study_ctx = (
                        f"Certificaci\xf3n/ruta elegida: "
                        f"{state.get('chosen_certification', 'seg\xfan las rutas curadas')}\n\n"
                        f"Rutas de aprendizaje recomendadas por el Curator Agent:\n"
                        f"{state.get('curator_data', {}).get('response', '') if state.get('curator_data') else ''}"
                    )
                    result = self._execute_assessment_questionnaire(
                        message, student_id, trace_context, study_context=_study_ctx,
                    )
                    response = result['response']
                    state['phase'] = result.get('phase', 'assessment_pending')
                    state['next_step'] = 'await_assessment_submission'
                    accumulated_tokens = self._sum_tokens(
                        accumulated_tokens, result)
                    execution_time = time.time() - start_time
                    log_performance_metrics(self.logger, trace_context,
                                            "orchestrator_workflow", execution_time,
                                            {"phase": state['phase']})
                    # Persistir antes del return temprano
                    self._save_interaction_to_db(
                        student_id=student_id,
                        user_msg=message,
                        assistant_msg=response,
                        phase=state['phase'],
                        session_id=trace_context.trace_id,
                        tokens=result.get('tokens', accumulated_tokens),
                        agent_name='AssessmentAgent',
                    )
                    return {
                        'response': response,
                        'logs': result.get('logs', []),
                        'phase': state['phase'],
                        'trace_id': trace_context.trace_id,
                        'tokens': result.get('tokens', accumulated_tokens),
                        'mode': 'azure_openai'
                    }
                else:
                    state['next_step'] = None
                    execution_time = time.time() - start_time
                    log_performance_metrics(self.logger, trace_context,
                                            "orchestrator_workflow", execution_time, {})
                    _no_eval_resp = "Entendido. Cuando quieras evaluarte, esc\xedbeme **'evaluar'**. \U0001f4da"
                    # Persistir antes del return temprano
                    self._save_interaction_to_db(
                        student_id=student_id,
                        user_msg=message,
                        assistant_msg=_no_eval_resp,
                        phase=state['phase'],
                        session_id=trace_context.trace_id,
                    )
                    return {
                        'response': _no_eval_resp,
                        'logs': [],
                        'phase': state['phase'],
                        'trace_id': trace_context.trace_id,
                        'tokens': {},
                        'mode': 'azure_openai'
                    }

            # HITL: await_assessment_submission
            # Ya se mostraron preguntas; esperar respuestas para corrección completa.
            if state.get('next_step') == 'await_assessment_submission':
                if self._is_assessment_submission(message):
                    state['next_step'] = None
                    _study_ctx = (
                        f"Certificación/ruta elegida: {state.get('chosen_certification', 'según las rutas curadas')}\n\n"
                        "Rutas de aprendizaje recomendadas por el Curator Agent:\n"
                        f"{state.get('curator_data', {}).get('response', '') if state.get('curator_data') else ''}"
                    )
                    eval_message = (
                        f"Contexto de estudio del estudiante:\n{_study_ctx}\n\n"
                        "El estudiante ha completado la evaluación y envía sus respuestas:\n\n"
                        f"{message}\n\n"
                        "Por favor evalúa cada respuesta indicando si es correcta o incorrecta, "
                        "explica el razonamiento correcto para las incorrectas y proporciona una "
                        "puntuación global de 0 a 100 con un veredicto de LISTO PARA EL EXAMEN o NECESITA REFUERZO."
                    )
                    result = self._execute_sub_workflow_2(
                        eval_message,
                        student_id,
                        trace_context,
                        study_context=_study_ctx,
                        plan_id=state.get('current_plan_id'),
                    )
                    response = result['response']
                    state['phase'] = 'assessment_complete'
                    state['next_step'] = (
                        'await_cert_advisor_confirmation'
                        if result.get('passed', False)
                        else None
                    )
                    accumulated_tokens = self._sum_tokens(
                        accumulated_tokens, result)
                    execution_time = time.time() - start_time
                    log_performance_metrics(
                        self.logger,
                        trace_context,
                        "orchestrator_workflow",
                        execution_time,
                        {"phase": state['phase']},
                    )
                    self._save_interaction_to_db(
                        student_id=student_id,
                        user_msg=message,
                        assistant_msg=response,
                        phase=state['phase'],
                        session_id=trace_context.trace_id,
                        tokens=result.get('tokens', accumulated_tokens),
                        agent_name='AssessmentAgent',
                    )
                    return {
                        'response': response,
                        'logs': result.get('logs', []),
                        'phase': state['phase'],
                        'trace_id': trace_context.trace_id,
                        'tokens': result.get('tokens', accumulated_tokens),
                        'mode': 'azure_openai',
                    }

                _pending_resp = (
                    "📝 Tienes una evaluación abierta. "
                    "Envíame tus respuestas desde el panel de evaluación para poder corregirlas."
                )
                self._save_interaction_to_db(
                    student_id=student_id,
                    user_msg=message,
                    assistant_msg=_pending_resp,
                    phase=state.get('phase', 'assessment_pending'),
                    session_id=trace_context.trace_id,
                )
                return {
                    'response': _pending_resp,
                    'logs': [],
                    'phase': state.get('phase', 'assessment_pending'),
                    'trace_id': trace_context.trace_id,
                    'tokens': {},
                    'mode': 'azure_openai',
                }

            # Sub-workflow 1 (preparación) ya corre completo sin pausa HITL — bloque await_study_plan_confirmation eliminado

            # -------------------------------------------------------
            # DETECCIÓN DE INTENCIÓN VÍA LLM (Planner–Executor)
            # -------------------------------------------------------
            intent_result = self._detect_intent_with_llm(message, state)
            intent = intent_result['intent']
            detected_topic = intent_result.get('topic', '')
            message_lower = message.lower()  # disponible por si algún sub-bloque lo necesita

            log_agent_action(
                self.logger, "OrchestratorAgent", "intent_analysis",
                f"Intent detected: {intent} (confidence={intent_result['confidence']:.2f})",
                {"intent": intent, "topic": detected_topic,
                    "current_phase": state['phase']},
                trace_context
            )

            # -------------------------------------------------------
            # ROUTING BASADO EN INTENCIÓN
            # -------------------------------------------------------

            # Transición de fase initial → ready (independiente del intent detectado).
            # Así el primer mensaje del usuario se enruta correctamente por su intención
            # real en lugar de mostrar siempre el saludo de bienvenida.
            if state['phase'] == 'initial':
                state['phase'] = 'ready'
                log_workflow_transition(self.logger, trace_context, "phase_change",
                                        "Phase set to ready", {"trigger": "initial"})

            if intent == 'greeting':
                response = (
                    "¡Hola! Soy **AEP CertMaster**, tu coach de certificaciones Microsoft. "
                    "Para comenzar, cuéntame qué te gustaría aprender o en qué área quieres crecer. "
                    "Por ejemplo:\n\n"
                    "- *'Quiero aprender inteligencia artificial'*\n"
                    "- *'Me interesa Business Central o Dynamics 365'*\n"
                    "- *'Quiero estudiar cloud computing con Azure'*\n"
                    "- *'Me gustaría prepararme para el AI-900'*\n\n"
                    "Con eso, mis agentes buscarán las mejores rutas de aprendizaje en Microsoft Learn, "
                    "crearán un plan de estudio personalizado y, cuando estés listo, "
                    "te recomendarán la certificación Microsoft más adecuada.\n\n"
                    "¿Qué quieres aprender hoy?"
                )

            elif intent == 'preparation':
                log_workflow_transition(self.logger, trace_context, "subworkflow_start",
                                        "Starting sub-workflow 1 (preparation)",
                                        {"intent": "preparation", "topic": detected_topic})
                # Actualizar certificación objetivo cuando el usuario la menciona
                if detected_topic:
                    state['chosen_certification'] = detected_topic
                    if persistence_tool:
                        try:
                            persistence_tool.save_student_profile(student_id, {
                                'student_id': student_id,
                                'chosen_certification': detected_topic,
                                'phase': 'preparation',
                                'updated_at': datetime.now().isoformat()
                            })
                        except Exception as _pe:
                            logger.warning(
                                f"⚠️ Error guardando perfil de certificación: {_pe}")
                # Sub-workflow completo (Curator → Study Plan → Engagement) sin pausa HITL
                result = self._execute_sub_workflow_1(
                    message, student_id, trace_context)
                response = result['response']
                state['phase'] = result.get('phase', 'preparation_complete')
                state['next_step'] = 'await_assessment_confirmation'
                if 'curator_data' in result:
                    state['curator_data'] = result['curator_data']
                # --- Persistir plan de estudio en la BD ---
                if persistence_tool:
                    try:
                        chosen_cert = state.get(
                            'chosen_certification', 'general')
                        _cert_slug = chosen_cert.replace(' ', '_').lower()[:30]
                        plan_id = f"plan_{student_id}_{_cert_slug}"
                        structured_plan = self._build_structured_study_plan(
                            plan_id=plan_id,
                            student_id=student_id,
                            certification=chosen_cert,
                            study_plan_response=result.get(
                                'sp_response', result.get('response', '')),
                            plan_name=f"{chosen_cert} · {datetime.now().strftime('%d/%m/%Y')}",
                        )
                        persistence_tool.save_student_profile(student_id, {
                            'student_id': student_id,
                            'chosen_certification': chosen_cert,
                            'curator_summary': result.get('curator_data', {}).get('response', '')[:800],
                            'phase': 'preparation_complete',
                            'updated_at': datetime.now().isoformat()
                        })
                        persistence_tool.save_study_plan(
                            plan_id, student_id, structured_plan)
                        state['current_plan_id'] = plan_id
                        print(f"[PERSISTENCE] Plan guardado: {plan_id}")
                    except Exception as _pe:
                        print(f"[PERSISTENCE] Error guardando plan: {_pe}")
                accumulated_tokens = self._sum_tokens(
                    accumulated_tokens, result)

            elif intent == 'assessment':
                log_workflow_transition(self.logger, trace_context, "subworkflow_start",
                                        "Starting sub-workflow 2 (assessment)",
                                        {"intent": intent})
                # Contexto de estudio para enriquecer el Assessment Agent
                _study_ctx = (
                    f"Certificación/ruta elegida: {state.get('chosen_certification', 'según las rutas curadas')}\n\n"
                    f"Rutas de aprendizaje recomendadas por el Curator Agent:\n"
                    f"{state.get('curator_data', {}).get('response', '') if state.get('curator_data') else ''}"
                )
                is_submission = self._is_assessment_submission(message)
                if is_submission:
                    # Respuestas enviadas desde el modal de evaluación
                    eval_message = (
                        f"Contexto de estudio del estudiante:\n{_study_ctx}\n\n"
                        "El estudiante ha completado la evaluación y envía sus respuestas:\n\n"
                        f"{message}\n\n"
                        "Por favor evalúa cada respuesta indicando si es correcta o incorrecta, "
                        "explica el razonamiento correcto para las incorrectas y proporciona una "
                        "puntuación global de 0 a 100 con un veredicto de LISTO PARA EL EXAMEN o NECESITA REFUERZO."
                    )
                    result = self._execute_sub_workflow_2(
                        eval_message, student_id, trace_context, study_context=_study_ctx,
                        plan_id=state.get('current_plan_id'))
                    state['phase'] = 'assessment_complete'
                    if result.get('passed', False):
                        state['next_step'] = 'await_cert_advisor_confirmation'
                    else:
                        state['next_step'] = None
                else:
                    result = self._execute_assessment_questionnaire(
                        message,
                        student_id,
                        trace_context,
                        study_context=_study_ctx,
                    )
                    state['phase'] = result.get('phase', 'assessment_pending')
                    state['next_step'] = 'await_assessment_submission'
                response = result['response']
                accumulated_tokens = self._sum_tokens(
                    accumulated_tokens, result)

            elif intent == 'certification':
                log_agent_action(self.logger, "OrchestratorAgent", "delegation",
                                 "Delegating to CertAdvisorAgent",
                                 {"target_agent": "cert_advisor"}, trace_context)
                cert_result = self.agents['cert_advisor'].execute(
                    message, student_id)
                cert_response = cert_result['response'] if isinstance(
                    cert_result, dict) else cert_result
                log_agent_response(self.logger, "CertAdvisorAgent", "certification_advice",
                                   f"Advice received ({len(cert_response)} chars)", {}, trace_context)
                response = f"🎓 **Recomendaciones de Certificación**\n\n{cert_response}\n\n¿Quieres que te ayude con la preparación para alguna certificación específica? Escríbeme el nombre o código y empezamos."
                state['phase'] = 'certification'
                # Esperar a que el usuario indique la cert elegida
                state['next_step'] = 'await_certification_choice'
                accumulated_tokens = self._sum_tokens(
                    accumulated_tokens, cert_result)

            elif intent == 'confirm':
                if state['phase'] == 'preparation':
                    log_agent_action(self.logger, "OrchestratorAgent", "delegation",
                                     "Delegating to EngagementAgent",
                                     {"target_agent": "engagement"}, trace_context)
                    engagement_result = self.agents['engagement'].execute(
                        message, student_id)
                    engagement_response = engagement_result['response'] if isinstance(
                        engagement_result, dict) else engagement_result
                    response = f"🎯 **Sistema de Motivación Activado**\n\n{engagement_response}\n\n¡Tu plan de preparación está completo! ¿Quieres hacer una evaluación o tienes alguna pregunta?"
                    state['phase'] = 'ready'
                    accumulated_tokens = self._sum_tokens(
                        accumulated_tokens, engagement_result)
                else:
                    response = "¡Genial! ¿En qué más puedo ayudarte? (preparación, evaluación, certificación)"

            elif intent == 'study_plan':
                response = (
                    "📚 Puedes ver tu plan de estudio en el dashboard. "
                    "Si aún no tienes uno, dime qué quieres aprender y lo generamos ahora mismo."
                )

            else:
                # 'other': No se detectó intención clara → responder con contexto
                log_agent_action(self.logger, "OrchestratorAgent", "fallback_handling",
                                 f"No clear intent for: {message[:80]}",
                                 {"intent": intent, "phase": state['phase']}, trace_context)
                context_response = self._get_context_aware_response(
                    message_lower, state['phase'])
                response = context_response

            # Registrar métricas finales y log de sesión
            execution_time = time.time() - start_time

            log_performance_metrics(
                self.logger, trace_context, "orchestrator_workflow",
                execution_time, {
                    "phase": state['phase'],
                    "response_length": len(response),
                    "user_message_length": len(message)
                }
            )

            log_workflow_transition(
                self.logger, trace_context, "session_end",
                f"Orchestrator workflow completed for user {student_id}",
                {
                    "total_time": execution_time,
                    "final_phase": state['phase'],
                    "response_length": len(response)
                }
            )

            # Guardar logs de esta interacción (legacy compatibility)
            interaction_log = {
                'timestamp': datetime.now().isoformat(),
                'student_id': student_id,
                'user_message': message,
                'phase': state['phase'],
                'final_response': response,
                'trace_id': trace_context.trace_id,
                'execution_time': execution_time
            }
            self.interaction_logs.append(interaction_log)

            # Persistir conversación y métricas de sesión en la base de datos
            if persistence_tool:
                try:
                    session_id = trace_context.trace_id
                    persistence_tool.save_conversation_message(
                        student_id=student_id,
                        role='user',
                        content=message,
                        session_id=session_id,
                        phase=state['phase']
                    )
                    persistence_tool.save_conversation_message(
                        student_id=student_id,
                        role='assistant',
                        content=response,
                        session_id=session_id,
                        agent_name='OrchestratorAgent',
                        phase=state['phase'],
                        tokens=accumulated_tokens
                    )
                    persistence_tool.save_session_metrics(
                        session_id=session_id,
                        student_id=student_id,
                        tokens=accumulated_tokens,
                        total_interactions=1,
                        final_phase=state['phase'],
                        execution_time_seconds=execution_time
                    )
                except Exception as pe:
                    logger.warning(f"⚠️ Error al persistir conversación: {pe}")

            # Registrar métricas del orquestador con los tokens acumulados
            if get_metrics_collector and accumulated_tokens['total_tokens'] > 0:
                try:
                    collector = get_metrics_collector()
                    collector.record_agent_call(
                        agent_name='OrchestratorAgent',
                        response_time=time.time() - start_time,
                        tokens_used=accumulated_tokens['total_tokens'],
                        prompt_tokens=accumulated_tokens['prompt_tokens'],
                        completion_tokens=accumulated_tokens['completion_tokens'],
                        success=True
                    )
                except Exception:
                    pass

            return {
                'response': response,
                'logs': [],  # Legacy compatibility - structured logs are in the logger
                'phase': state['phase'],
                'trace_id': trace_context.trace_id,
                'execution_time': execution_time,
                'tokens': accumulated_tokens,
                'mode': 'azure_openai'
            }

        except Exception as e:
            # Log de error con contexto completo
            execution_time = time.time() - start_time

            log_error(
                self.logger, "OrchestratorAgent", "execute",
                f"Error in orchestrator workflow: {str(e)}",
                {"error_type": type(e).__name__,
                 "execution_time": execution_time},
                trace_context
            )

            # Respuesta de fallback
            response = "Lo siento, hubo un error procesando tu solicitud. ¿Puedes intentarlo de nuevo?"

            return {
                'response': response,
                'logs': [],
                'phase': state['phase'],
                'trace_id': trace_context.trace_id,
                'error': str(e),
                'execution_time': execution_time,
                'tokens': accumulated_tokens,
                'mode': 'azure_openai'
            }

    def _fmt_agent_block(self, result: dict, step: int, total: int) -> str:
        """Formatea la respuesta de un agente con su header identificador y herramientas usadas."""
        if not isinstance(result, dict):
            return str(result)
        agent_name = result.get('agent_name', 'Agente')
        tools = result.get('tools_used', [])
        mode = result.get('mode', 'azure_openai')
        tokens = result.get('tokens', {})
        content = result.get('response', '')
        tools_str = ' · '.join([f'`{t}`' for t in tools]) if tools else ''
        mode_label = '🔵 Azure OpenAI'
        token_str = f"({tokens.get('total_tokens', 0)} tokens)" if tokens else ''
        header = f"""---
### [{step}/{total}] {agent_name} {mode_label} {token_str}
🛠️ Herramientas: {tools_str if tools_str else 'ninguna'}
---
"""
        return header + content

    @staticmethod
    def _sum_tokens(*results) -> dict:
        """Suma los tokens de múltiples resultados de agentes."""
        total = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
        for r in results:
            if isinstance(r, dict):
                t = r.get('tokens', {})
                total['prompt_tokens'] += t.get('prompt_tokens', 0)
                total['completion_tokens'] += t.get('completion_tokens', 0)
                total['total_tokens'] += t.get('total_tokens', 0)
        return total

    def _save_interaction_to_db(
        self,
        student_id: str,
        user_msg: str,
        assistant_msg: str,
        phase: str,
        session_id: str,
        tokens: dict = None,
        agent_name: str = 'OrchestratorAgent'
    ) -> None:
        """Persiste una interacción usuario+asistente en la BD.
        Utilizado por bloques HITL que hacen return temprano antes del
        bloque de persistencia al final de execute().
        """
        if not persistence_tool:
            return
        try:
            persistence_tool.save_conversation_message(
                student_id=student_id,
                role='user',
                content=user_msg,
                session_id=session_id,
                phase=phase,
            )
            persistence_tool.save_conversation_message(
                student_id=student_id,
                role='assistant',
                content=assistant_msg,
                session_id=session_id,
                agent_name=agent_name,
                phase=phase,
                tokens=tokens or {},
            )
        except Exception as _pe:
            logger.warning(
                f"\u26a0\ufe0f Error persistiendo interacción HITL: {_pe}")

    def _execute_sub_workflow_1(self, context: str, student_id: str, trace_context: TraceContext) -> dict:
        """Ejecuta Sub-Workflow #1: Curator → [human-in-loop] → Study Plan → Engagement."""
        log_workflow_transition(
            self.logger, trace_context, "subworkflow_1_start",
            "Starting preparation sub-workflow (Curator → Study Plan → Engagement)",
            {"student_id": student_id}
        )

        print(f"\n{'='*60}")
        print(f"[SUB-WORKFLOW 1] INICIO: Curator Agent")
        print(f"{'='*60}")

        # Paso 1: Curator Agent
        log_agent_action(self.logger, "OrchestratorAgent", "delegation",
                         "Delegating to CuratorAgent for learning path curation",
                         {"target_agent": "curator"}, trace_context)

        self._emit_agent_active('curator', '🏛️ Curator Agent')
        curator_start = time.time()
        curator_result = self.agents['curator'].execute(context, student_id)
        curator_time = time.time() - curator_start
        curator_result = curator_result if isinstance(curator_result, dict) else {
            'response': curator_result, 'agent_name': 'Curator Agent', 'tools_used': []}

        print(f"[SUB-WORKFLOW 1] Curator completado en {curator_time:.2f}s")
        print(
            f"[SUB-WORKFLOW 1] Herramientas usadas: {curator_result.get('tools_used', [])}")

        log_agent_response(self.logger, "CuratorAgent", "curation_complete",
                           f"Curated paths ready ({len(curator_result['response'])} chars, {curator_time:.2f}s)",
                           {"tokens": curator_result.get('tokens', {})}, trace_context)

        curator_block = self._fmt_agent_block(curator_result, 1, 3)

        # Emitir resultado del Curator de inmediato para mejor UX
        self._emit_partial_response(
            f"\U0001f9e0 **Orquestador → Sub-Workflow de Preparación iniciado** (Paso 1 de 3)\n"
            f"Agentes involucrados: Curator → Study Plan → Engagement\n\n"
            + curator_block,
            loading_next="📚 Study Plan Agent en proceso..."
        )

        # -------------------------------------------------------
        # Paso 2: Study Plan Agent
        # -------------------------------------------------------
        log_agent_action(self.logger, "OrchestratorAgent", "delegation",
                         "Delegating to StudyPlanAgent", {"target_agent": "study_plan"}, trace_context)

        curator_content = curator_result.get('response', '')
        sp_context = (
            f"El estudiante quiere aprender: {context}\n\n"
            f"El Curator Agent ha identificado las siguientes rutas de aprendizaje "
            f"y certificaciones recomendadas:\n\n"
            f"{curator_content}\n\n"
            f"Basándote EXCLUSIVAMENTE en estas rutas curadas, genera el plan de estudio "
            f"personalizado. NO pidas al usuario la certificación — ya está identificada arriba."
        )

        self._emit_agent_active('study_plan', '📚 Study Plan Agent')
        sp_start = time.time()
        sp_result = self.agents['study_plan'].execute(sp_context, student_id)
        sp_time = time.time() - sp_start
        sp_result = sp_result if isinstance(sp_result, dict) else {
            'response': sp_result, 'agent_name': 'Study Plan Agent', 'tools_used': []}

        print(f"[SUB-WORKFLOW 1] Study Plan completado en {sp_time:.2f}s")
        log_agent_response(self.logger, "StudyPlanAgent", "study_plan_complete",
                           f"Study plan ready ({len(sp_result['response'])} chars, {sp_time:.2f}s)",
                           {"tokens": sp_result.get('tokens', {})}, trace_context)

        sp_block = self._fmt_agent_block(sp_result, 2, 3)
        self._emit_partial_response(
            sp_block, loading_next="🎯 Engagement Agent en proceso...")

        # -------------------------------------------------------
        # Paso 3: Engagement Agent
        # -------------------------------------------------------
        log_agent_action(self.logger, "OrchestratorAgent", "delegation",
                         "Delegating to EngagementAgent", {"target_agent": "engagement"}, trace_context)

        eng_context = (
            f"El estudiante quiere aprender: {context}\n\n"
            f"Se ha generado el siguiente plan de estudio:\n\n"
            f"{sp_result.get('response', '')}\n\n"
            f"Crea el sistema de motivación y recordatorios por email "
            f"adaptado específicamente a este plan y a las certificaciones mencionadas."
        )

        self._emit_agent_active('engagement', '🎯 Engagement Agent')
        eng_start = time.time()
        eng_result = self.agents['engagement'].execute(eng_context, student_id)
        eng_time = time.time() - eng_start
        eng_result = eng_result if isinstance(eng_result, dict) else {
            'response': eng_result, 'agent_name': 'Engagement Agent', 'tools_used': []}

        print(f"[SUB-WORKFLOW 1] Engagement completado en {eng_time:.2f}s")
        log_agent_response(self.logger, "EngagementAgent", "engagement_complete",
                           f"Engagement setup ready ({len(eng_result['response'])} chars, {eng_time:.2f}s)",
                           {"tokens": eng_result.get('tokens', {})}, trace_context)

        eng_block = self._fmt_agent_block(eng_result, 3, 3)

        total_time = curator_time + sp_time + eng_time
        response = (
            f"✅ **Sub-Workflow #1 completado** (3/3 agentes ejecutados en {total_time:.1f}s)\n\n"
            "🧾 El **Study Plan Agent** ya publicó el plan en tiempo real en el chat.\n\n"
            + eng_block
            + "\n\n---\n"
            "🏁 **Tu preparación está lista.** ¿Quieres hacer una evaluación con el "
            "**Assessment Agent** para medir tu nivel actual? (responde **'evaluar'** o **'assessment'**)"
        )

        log_workflow_transition(self.logger, trace_context, "subworkflow_1_complete",
                                "Preparation sub-workflow completed (no HITL pause)",
                                {"total_steps": 3, "total_time": total_time})

        print(
            f"[SUB-WORKFLOW 1] COMPLETADO. Total: 3 agentes, {total_time:.2f}s")

        return {
            'response': response,
            'phase': 'preparation_complete',
            'curator_data': curator_result,
            'sp_response': sp_result.get('response', ''),
            'tokens': self._sum_tokens(curator_result, sp_result, eng_result)
        }

    def _execute_sub_workflow_1_continue(self, context: str, student_id: str, curator_data: dict, trace_context: TraceContext) -> dict:
        """Continúa Sub-Workflow #1 después de confirmación del usuario."""
        log_workflow_transition(self.logger, trace_context, "subworkflow_1_continue",
                                "Continuing preparation sub-workflow after user confirmation",
                                {"student_id": student_id})

        print(f"\n{'='*60}")
        print(f"[SUB-WORKFLOW 1] CONTINUACIÓN: Study Plan Agent + Engagement Agent")
        print(f"{'='*60}")

        # -------------------------------------------------------
        # Paso 2: Study Plan Agent
        # Pasamos el output del Curator como contexto enriquecido para que el
        # Study Plan Agent NO pida la certificación de nuevo — ya la conoce.
        # -------------------------------------------------------
        log_agent_action(self.logger, "OrchestratorAgent", "delegation",
                         "Delegating to StudyPlanAgent", {"target_agent": "study_plan"}, trace_context)

        curator_content = curator_data.get(
            'response', '') if curator_data else ''
        sp_context = (
            f"El estudiante quiere aprender: {context}\n\n"
            f"El Curator Agent ha identificado las siguientes rutas de aprendizaje "
            f"y certificaciones recomendadas:\n\n"
            f"{curator_content}\n\n"
            f"Basándote EXCLUSIVAMENTE en estas rutas curadas, genera el plan de estudio "
            f"personalizado. NO pidas al usuario la certificación — ya está identificada arriba."
        )

        self._emit_agent_active('study_plan', '📚 Study Plan Agent')
        sp_start = time.time()
        sp_result = self.agents['study_plan'].execute(sp_context, student_id)
        sp_time = time.time() - sp_start
        sp_result = sp_result if isinstance(sp_result, dict) else {
            'response': sp_result, 'agent_name': 'Study Plan Agent', 'tools_used': []}

        print(f"[SUB-WORKFLOW 1] Study Plan completado en {sp_time:.2f}s")
        print(
            f"[SUB-WORKFLOW 1] Herramientas: {sp_result.get('tools_used', [])}")

        log_agent_response(self.logger, "StudyPlanAgent", "study_plan_complete",
                           f"Study plan ready ({len(sp_result['response'])} chars, {sp_time:.2f}s)",
                           {"tokens": sp_result.get('tokens', {})}, trace_context)

        # Emitir el resultado del Study Plan de inmediato para mejor UX
        # (el usuario lo ve sin esperar al Engagement Agent)
        sp_block = self._fmt_agent_block(sp_result, 1, 2)
        self._emit_partial_response(
            sp_block, loading_next="🎯 Engagement Agent en proceso...")

        # -------------------------------------------------------
        # Paso 3: Engagement Agent
        # Recibe el plan generado para crear recordatorios y motivación coherentes.
        # -------------------------------------------------------
        log_agent_action(self.logger, "OrchestratorAgent", "delegation",
                         "Delegating to EngagementAgent", {"target_agent": "engagement"}, trace_context)

        eng_context = (
            f"El estudiante quiere aprender: {context}\n\n"
            f"Se ha generado el siguiente plan de estudio:\n\n"
            f"{sp_result.get('response', '')}\n\n"
            f"Crea el sistema de motivación y recordatorios por email "
            f"adaptado específicamente a este plan y a las certificaciones mencionadas."
        )

        self._emit_agent_active('engagement', '🎯 Engagement Agent')
        eng_start = time.time()
        eng_result = self.agents['engagement'].execute(eng_context, student_id)
        eng_time = time.time() - eng_start
        eng_result = eng_result if isinstance(eng_result, dict) else {
            'response': eng_result, 'agent_name': 'Engagement Agent', 'tools_used': []}

        print(f"[SUB-WORKFLOW 1] Engagement completado en {eng_time:.2f}s")
        print(
            f"[SUB-WORKFLOW 1] Herramientas: {eng_result.get('tools_used', [])}")

        log_agent_response(self.logger, "EngagementAgent", "engagement_complete",
                           f"Engagement setup ready ({len(eng_result['response'])} chars, {eng_time:.2f}s)",
                           {"tokens": eng_result.get('tokens', {})}, trace_context)

        eng_block = self._fmt_agent_block(eng_result, 2, 2)

        total_time = sp_time + eng_time
        response = (
            f"✅ **Sub-Workflow #1 completado** (2/2 agentes ejecutados en {total_time:.1f}s)\n\n"
            "🧾 El **Study Plan Agent** ya publicó el plan en tiempo real en el chat.\n\n"
            + eng_block +
            f"\n\n---\n"
            f"🏁 **Tu preparación está lista.** ¿Quieres hacer una evaluación con el **Assessment Agent** para medir tu nivel actual? (responde **'evaluar'** o **'assessment'**)"
        )

        log_workflow_transition(self.logger, trace_context, "subworkflow_1_complete",
                                "Preparation sub-workflow completed",
                                {"total_steps": 2, "total_time": total_time})

        print(
            f"[SUB-WORKFLOW 1] COMPLETADO. Total: 2 agentes, {total_time:.2f}s")

        return {'response': response, 'phase': 'preparation_complete', 'tokens': self._sum_tokens(sp_result, eng_result)}

    def _execute_sub_workflow_2(self, context: str, student_id: str, trace_context: TraceContext,
                                study_context: str = None, plan_id: str = None) -> dict:
        """Ejecuta Sub-Workflow #2: Assessment → Critic → Decision."""
        log_workflow_transition(self.logger, trace_context, "subworkflow_2_start",
                                "Starting assessment sub-workflow (Assessment → Critic → Decision)",
                                {"student_id": student_id})

        print(f"\n{'='*60}")
        print(f"[SUB-WORKFLOW 2] INICIO: Assessment → Critic → Decision")
        print(f"{'='*60}")

        # Enriquecer el contexto con la certificación elegida y rutas curadas,
        # para que el Assessment genere preguntas sobre el tema real del estudiante.
        if study_context:
            enriched_context = (
                f"CONTEXTO DEL ESTUDIANTE (usa esto para generar todas las preguntas):\n"
                f"{study_context}\n\n"
                f"Solicitud del estudiante: {context}"
            )
        else:
            enriched_context = context

        # Paso 1: Assessment Agent
        log_agent_action(self.logger, "OrchestratorAgent", "delegation",
                         "Delegating to AssessmentAgent", {"target_agent": "assessment"}, trace_context)

        self._emit_agent_active('assessment', '📝 Assessment Agent')
        ass_start = time.time()
        ass_result = self.agents['assessment'].execute(
            enriched_context, student_id)
        ass_time = time.time() - ass_start
        ass_result = ass_result if isinstance(ass_result, dict) else {
            'response': ass_result, 'agent_name': 'Assessment Agent', 'tools_used': []}

        print(f"[SUB-WORKFLOW 2] Assessment completado en {ass_time:.2f}s")

        log_agent_response(self.logger, "AssessmentAgent", "assessment_complete",
                           f"Assessment ready ({len(ass_result['response'])} chars, {ass_time:.2f}s)",
                           {"tokens": ass_result.get('tokens', {})}, trace_context)

        # Paso 2: Critic Agent
        log_agent_action(self.logger, "OrchestratorAgent", "delegation",
                         "Delegating to CriticAgent for analysis", {"target_agent": "critic"}, trace_context)

        critic_context = (
            f"Solicitud original del estudiante:\n{context}\n\n"
            f"Resultado completo del Assessment Agent (analiza este contenido):\n"
            f"{ass_result.get('response', '')}"
        )

        self._emit_agent_active('critic', '\U0001f50d Critic Agent')
        cr_start = time.time()
        cr_result = self.agents['critic'].execute(critic_context, student_id)
        cr_time = time.time() - cr_start
        cr_result = cr_result if isinstance(cr_result, dict) else {
            'response': cr_result, 'agent_name': 'Critic Agent', 'tools_used': []}

        print(f"[SUB-WORKFLOW 2] Critic completado en {cr_time:.2f}s")

        log_agent_response(self.logger, "CriticAgent", "critic_complete",
                           f"Critic analysis ready ({len(cr_result['response'])} chars, {cr_time:.2f}s)",
                           {"tokens": cr_result.get('tokens', {})}, trace_context)

        # Paso 3: Decision
        passed = self._evaluate_assessment_results(ass_result, cr_result)

        print(
            f"[SUB-WORKFLOW 2] DECISION: {'APROBADO ✅' if passed else 'NO APROBADO ❌ → REQUIERE REFUERZO'}")

        log_workflow_transition(self.logger, trace_context, "assessment_decision",
                                f"Assessment decision: {'PASSED' if passed else 'FAILED → reinforcement required'}",
                                {"passed": passed})

        ass_block = self._fmt_agent_block(ass_result, 1, 2)
        cr_block = self._fmt_agent_block(cr_result, 2, 2)
        total_time = ass_time + cr_time

        if passed:
            response = (
                f"\U0001f9e0 **Sub-Workflow #2: Evaluación completada** (2/2 agentes, {total_time:.1f}s)\n"
                f"Veredicto del Orchestrator: **APROBADO ✅**\n\n"
                + ass_block + "\n\n"
                + cr_block +
                f"\n\n---\n"
                f"🎉 ¡Felicitaciones! Estás listo para el examen. ¿Quieres que el **Cert Advisor Agent** te recomiende el examen Microsoft ideal para tu perfil?"
            )
        else:
            response = (
                f"\U0001f9e0 **Sub-Workflow #2: Evaluación completada** (2/2 agentes, {total_time:.1f}s)\n"
                f"Veredicto del Orchestrator: **NECESITA REFUERZO ⚠️**\n\n"
                + ass_block + "\n\n"
                + cr_block +
                f"\n\n---\n"
                "📌 No se reinicia el proceso automáticamente.\n"
                "Si quieres refinar tu plan con las brechas detectadas, escribe **'preparar'**."
            )

        print(
            f"[SUB-WORKFLOW 2] COMPLETADO. Total: 2 agentes, {total_time:.2f}s")

        log_workflow_transition(self.logger, trace_context, "subworkflow_2_complete",
                                "Assessment sub-workflow completed", {"passed": passed, "total_time": total_time})

        # --- Persistencia: guardar resultado de evaluacion ---
        if persistence_tool:
            try:
                assessment_id = f"assessment_{student_id}_{int(time.time())}"
                assessment_score = self._extract_assessment_score(
                    ass_result, cr_result)
                persistence_tool.save_assessment(
                    assessment_id, student_id,
                    {
                        'assessment_response': ass_result.get('response', ''),
                        'critic_response': cr_result.get('response', ''),
                        'context': context[:500],
                        'timestamp': datetime.now().isoformat()
                    },
                    score=assessment_score,
                    passed=passed,
                    plan_id=plan_id
                )
                print(
                    f"[PERSISTENCE] Evaluaci\xf3n guardada: {assessment_id}, "
                    f"score={assessment_score}, passed={passed}, plan={plan_id}")
            except Exception as _pe:
                print(f"[PERSISTENCE] Error guardando evaluaci\xf3n: {_pe}")

        return {'response': response, 'passed': passed, 'phase': 'assessment_complete', 'tokens': self._sum_tokens(ass_result, cr_result)}

    def _get_context_aware_response(self, message_lower: str, current_phase: str) -> str:
        """Genera respuesta basada en el contexto de la conversación actual."""
        if current_phase == 'preparation':
            return "Sobre preparación, puedo ayudarte con planes de estudio personalizados. ¿Qué certificación te interesa?"
        elif current_phase == 'assessment':
            return "Para evaluaciones, puedo hacer tests de conocimientos. ¿Qué área quieres evaluar?"
        elif current_phase == 'certification':
            return "Para certificaciones, te recomiendo certificaciones Microsoft como Azure AI-900. ¿Quieres más detalles?"
        else:
            return "¡Buena pregunta! Puedo ayudarte con preparación, evaluación o certificaciones. ¿Qué te interesa específicamente?"

    def _evaluate_assessment_results(self, assessment_result, critic_result):
        """Evalúa si el estudiante pasó la evaluación usando el LLM."""
        import os as _os
        import json as _json

        assessment_text = assessment_result.get('response', '') if isinstance(
            assessment_result, dict) else str(assessment_result)
        critic_text = critic_result.get('response', '') if isinstance(
            critic_result, dict) else str(critic_result)

        endpoint = _os.getenv('AZURE_OPENAI_ENDPOINT', '')
        api_key = _os.getenv('AZURE_OPENAI_API_KEY', '')
        if not (endpoint and api_key):
            return False

        try:
            from openai import AzureOpenAI as _AOAI
            _client = _AOAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version=_os.getenv(
                    'AZURE_OPENAI_API_VERSION', '2024-02-15-preview'),
            )
            resp = _client.chat.completions.create(
                model=_os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', 'gpt-4o'),
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Eres un evaluador de resultados de assessment. "
                            "Determina si el estudiante está listo para pasar "
                            "(true) o necesita refuerzo (false) en base al "
                            "contenido completo recibido. "
                            "Responde SOLO JSON: {\"passed\": true} o "
                            "{\"passed\": false}."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            "Resultado Assessment Agent:\n"
                            f"{assessment_text}\n\n"
                            "Resultado Critic Agent:\n"
                            f"{critic_text}"
                        )
                    }
                ],
                max_tokens=20,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            result = _json.loads(resp.choices[0].message.content.strip())
            return bool(result.get('passed', False))
        except Exception:
            return False

    @staticmethod
    def _extract_assessment_score(assessment_result, critic_result) -> float | None:
        """Extrae una puntuación de 0-100 desde la respuesta de assessment/critic."""
        import re

        parts = []
        if isinstance(assessment_result, dict):
            parts.append(str(assessment_result.get('response', '')))
        else:
            parts.append(str(assessment_result or ''))

        if isinstance(critic_result, dict):
            parts.append(str(critic_result.get('response', '')))
        else:
            parts.append(str(critic_result or ''))

        text = "\n".join(parts)

        patterns = [
            r'(?:score|puntuaci[oó]n|score\s+global)\s*[:=]?\s*(\d{1,3})(?:\s*/\s*100)?',
            r'(\d{1,3})\s*/\s*100',
            r'(\d{1,3})\s*%'
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    return max(0.0, min(100.0, value))
                except (ValueError, TypeError):
                    continue

        return None

    def get_interaction_logs(self, student_id: str = None) -> list:
        """Obtiene los logs de interacciones."""
        if student_id:
            return [log for log in self.interaction_logs if log['student_id'] == student_id]
        return self.interaction_logs


# Crear instancia del orchestrator DESPUÉS de cargar .env
orchestrator_agent = ExternalOrchestratorAgent()


# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar variables de entorno desde .env (por si acaso)
try:
    from dotenv import load_dotenv
    load_dotenv()
    logger.info("Archivo .env cargado correctamente")
except ImportError:
    logger.warning(
        "python-dotenv no disponible - usando variables de entorno del sistema")

# Configuración de Telemetría (Azure Application Insights)
try:
    from azure.monitor.opentelemetry import configure_azure_monitor

    connection_string = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
    if connection_string:
        configure_azure_monitor(
            connection_string=connection_string,
            disable_metric=True  # Deshabilitar métricas automáticas para reducir envío continuo
        )
        logger.info(
            "Azure Application Insights telemetry configurado (métricas automáticas deshabilitadas)")

        # Reducir logging detallado de Azure SDK para evitar spam en consola
        logging.getLogger('azure.core.pipeline.policies.http_logging_policy').setLevel(
            logging.WARNING)
        logging.getLogger('azure.monitor.opentelemetry').setLevel(
            logging.WARNING)
        logging.getLogger('azure.monitor.opentelemetry.exporter').setLevel(
            logging.WARNING)
        logger.info(
            "Logging de Azure SDK reducido a WARNING para minimizar output en consola")
    else:
        logger.warning(
            "APPLICATIONINSIGHTS_CONNECTION_STRING no configurada - telemetry deshabilitado")
except ImportError:
    logger.warning(
        "Azure Monitor OpenTelemetry no disponible - telemetry deshabilitado")


# Crear aplicación Flask
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv(
    'FLASK_SECRET_KEY', 'aep-certmaster-secret-key-dev')
socketio = SocketIO(app, cors_allowed_origins="*")


@app.before_request
def start_request_trace():
    """Inicializa contexto de trazabilidad por request."""
    student_id = (
        session.get('student_id')
        or request.headers.get("X-Student-Id")
        or request.args.get("student_id")
        or "anonymous"
    )
    session_id = request.headers.get("X-Session-Id", "")
    g.trace_context = create_trace_context(
        user_id=student_id, session_id=session_id)
    log_agent_action(
        logger,
        "Request",
        "request_start",
        f"{request.method} {request.path}",
        {"remote_addr": request.remote_addr},
        g.trace_context,
    )


@app.after_request
def end_request_trace(response: Response):
    """Cierra contexto de trazabilidad por request."""
    trace_context = getattr(g, "trace_context", None)
    if trace_context:
        log_agent_action(
            logger,
            "Request",
            "request_end",
            f"{request.method} {request.path}",
            {"status_code": response.status_code},
            trace_context,
        )
        save_trace_context(trace_context)
    return response


# Almacenamiento global de agentes
agents = {}
chat_history = {}


def initialize_agents():
    """Inicializar el sistema de agentes (ahora usa OrchestratorAgent)."""
    global agents

    try:
        logger.info("Inicializando sistema de agentes con Orchestrator...")

        # Los agentes ahora están disponibles a través del OrchestratorAgent
        # No necesitamos inicializar agentes individuales
        agents = {}  # Mantener vacío ya que usamos orchestrator_agent global

        logger.info(
            "✅ Sistema de agentes inicializado correctamente (modo orchestrator)")

        # Inicializar historial de chat vacío
        global chat_history
        chat_history = {}

        logger.info("✅ Historial de chat inicializado")

    except Exception as e:
        logger.error(f"❌ Error inicializando agentes: {e}")
        raise
        for agent_name in agents.keys():
            chat_history[agent_name] = []

    except Exception as e:
        logger.error(f"❌ Error inicializando agentes: {e}")
        agents = {}


# Inicializar agentes al importar el módulo
initialize_agents()


@app.route('/')
def index():
    """Página principal del dashboard con acceso al orchestrator."""
    if not session.get('user_id'):
        return redirect(url_for('login_page'))
    return render_template('index.html')


@app.route('/chat')
def chat_page():
    """Página de chat con el orchestrator agent."""
    agent_info = {
        'name': 'orchestrator',
        'display_name': '🤖 AEP CertMaster Assistant',
        'description': 'Asistente inteligente que coordina todos los agentes especializados para tu proceso de certificación'
    }
    return render_template('chat.html', agent=agent_info)


@app.route('/logs')
def logs_page():
    """Página para ver los logs de interacciones."""
    return render_template('logs.html')


@app.route('/health')
def health_page():
    """Página dedicada para mostrar el estado del sistema."""
    return render_template('health.html')


@app.route('/api/health')
def health_check():
    """Endpoint de verificación de salud."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'orchestrator': 'active',
        'specialized_agents': list(orchestrator_agent.agents.keys()),
        'total_interactions': len(orchestrator_agent.interaction_logs)
    })


@app.route('/api/agents')
def get_agents():
    """Obtener lista de agentes disponibles."""
    agents_info = []
    for name, agent in agents.items():
        agents_info.append({
            'id': name,
            'name': {
                'curator': '🏛️ Curator Agent',
                'study_plan': '📚 Study Plan Agent',
                'engagement': '🎯 Engagement Agent',
                'assessment': '📝 Assessment Agent',
                'critic': '🔍 Critic Agent',
                'cert_advisor': '🎓 Cert Advisor Agent'
            }.get(name, name),
            'description': {
                'curator': 'Especialista en gestionar itinerarios de aprendizaje relevantes',
                'study_plan': 'Especialista en crear planes de estudio personalizados',
                'engagement': 'Especialista en motivación y recordatorios estudiantiles',
                'assessment': 'Especialista en evaluaciones inteligentes',
                'critic': 'Especialista en validación y crítica de calidad',
                'cert_advisor': 'Especialista en asesoramiento de certificaciones'
            }.get(name, 'Agente del sistema AEP CertMaster'),
            'status': 'active'
        })

    return jsonify({'agents': agents_info})


@app.route('/api/agents/active')
def get_active_agent():
    """Obtener información del agente activo actualmente."""
    # En un sistema real, esto vendría del estado del orchestrator
    # Por ahora, devolver el último agente usado o el orchestrator por defecto
    active_agent = {
        'id': 'orchestrator',
        'name': '🤖 Orchestrator Agent',
        'description': 'Coordinador principal del sistema multi-agente',
        'status': 'active',
        'last_activity': datetime.now().isoformat(),
        'current_task': 'Esperando instrucciones del usuario'
    }
    return jsonify({'active_agent': active_agent})


@app.route('/api/agents/reasoning')
def get_agent_reasoning():
    """Obtener el razonamiento actual del agente activo desde el estado real del orquestador."""
    student_id = session.get('student_id', 'anonymous')
    state = orchestrator_agent.conversation_state.get(student_id, {})

    current_phase = state.get('phase', 'ready')
    next_step = state.get('next_step', '')
    chosen_cert = state.get('chosen_certification', '')

    # Construir decision_process desde el estado real
    decision_process = []
    if current_phase == 'initial' or current_phase == 'ready':
        decision_process = [
            '1. Esperando mensaje del usuario',
            '2. Clasificar intención con LLM',
            '3. Seleccionar sub-workflow apropiado',
            '4. Activar agente especializado'
        ]
    elif current_phase == 'preparation':
        decision_process = [
            f'1. Fase: preparación activa',
            f'2. Certificación seleccionada: {chosen_cert or "pendiente"}',
            f'3. Siguiente paso HITL: {next_step or "ninguno"}',
            '4. Coordinando Curator → StudyPlan → Engagement'
        ]
    elif current_phase == 'assessment':
        decision_process = [
            '1. Fase: evaluación activa',
            f'2. Certificación: {chosen_cert or "N/A"}',
            f'3. Estado: {next_step or "procesando"}',
            '4. Coordinando Assessment → Critic'
        ]
    elif current_phase == 'certification':
        decision_process = [
            '1. Fase: asesoramiento de certificación',
            f'2. Certificación recomendada: {chosen_cert or "en curso"}',
            '3. Cert Advisor analizando opciones',
            '4. Generando roadmap personalizado'
        ]
    else:
        decision_process = [
            f'1. Fase actual: {current_phase}', '2. Procesando...']

    # Mapa de fase → agente activo
    phase_agent_map = {
        'preparation': 'curator',
        'assessment': 'assessment',
        'certification': 'cert_advisor',
        'ready': 'orchestrator',
        'initial': 'orchestrator'
    }
    active_agent_id = phase_agent_map.get(current_phase, 'orchestrator')
    agent_labels = {
        'orchestrator': 'Orchestrator Agent',
        'curator': 'Curator Agent',
        'study_plan': 'Study Plan Agent',
        'engagement': 'Engagement Agent',
        'assessment': 'Assessment Agent',
        'critic': 'Critic Agent',
        'cert_advisor': 'Cert Advisor Agent'
    }

    reasoning = {
        'agent_id': active_agent_id,
        'agent_name': agent_labels.get(active_agent_id, 'Orchestrator Agent'),
        'current_phase': current_phase,
        'current_thought': f'Fase: {current_phase} | Siguiente: {next_step or "determinando siguiente paso"}',
        'decision_process': decision_process,
        'chosen_certification': chosen_cert,
        'next_step': next_step,
        'timestamp': datetime.now().isoformat()
    }
    return jsonify({'reasoning': reasoning})


@app.route('/api/agents/workflow')
def get_agents_workflow():
    """Obtener el workflow actual de agentes desde el estado real del orquestador."""
    student_id = session.get('student_id', 'anonymous')
    state = orchestrator_agent.conversation_state.get(student_id, {})
    current_phase = state.get('phase', 'ready')
    next_step = state.get('next_step', '')
    chosen_cert = state.get('chosen_certification', '')
    agent_labels = {
        'orchestrator': '🤖 OrchestratorAgent',
        'curator': '🏛️ CuratorAgent',
        'study_plan': '📚 StudyPlanAgent',
        'engagement': '🎯 EngagementAgent',
        'assessment': '📝 AssessmentAgent',
        'critic': '🔍 CriticAgent',
        'cert_advisor': '🎓 CertAdvisorAgent'
    }
    # Determinar estado de cada paso según la fase actual
    phase_order = ['initial', 'preparation', 'assessment', 'certification']
    phase_idx = phase_order.index(
        current_phase) if current_phase in phase_order else 0

    def step_status(required_phase_idx):
        if phase_idx > required_phase_idx:
            return 'completed'
        if phase_idx == required_phase_idx:
            return 'active'
        return 'pending'

    workflow_steps = [
        {
            'step': 1,
            'agent': agent_labels['orchestrator'],
            'action': 'Análisis de intención del usuario (LLM)',
            'status': 'completed' if phase_idx >= 0 else 'pending',
        },
        {
            'step': 2,
            'agent': agent_labels['curator'],
            'action': f'Curación de itinerarios{" → " + chosen_cert if chosen_cert else ""}',
            'status': step_status(1),
        },
        {
            'step': 3,
            'agent': f"{agent_labels['study_plan']} + {agent_labels['engagement']}",
            'action': 'Plan de estudio + Motivación',
            'status': step_status(1),
        },
        {
            'step': 4,
            'agent': f"{agent_labels['assessment']} + {agent_labels['critic']}",
            'action': f'Evaluación adaptativa{" (" + next_step + ")" if next_step else ""}',
            'status': step_status(2),
        },
        {
            'step': 5,
            'agent': agent_labels['cert_advisor'],
            'action': 'Advisory y roadmap de certificaciones',
            'status': step_status(3),
        }
    ]
    completed_steps = sum(
        1 for s in workflow_steps if s['status'] == 'completed')

    workflow = {
        'current_phase': current_phase,
        'next_step': next_step,
        'chosen_certification': chosen_cert,
        'active_agent': f'{current_phase}_agent',
        'workflow_steps': workflow_steps,
        'total_steps': len(workflow_steps),
        'completed_steps': completed_steps
    }
    return jsonify({'workflow': workflow})


@app.route('/api/agents/metrics')
def get_agents_metrics():
    """Obtener métricas de rendimiento de agentes."""
    student_id = (
        request.args.get('student_id')
        or session.get('student_id')
        or 'demo_student'
    )

    print(
        "🔍 API métricas llamada - "
        f"student_id={student_id}, "
        f"get_metrics_collector disponible: {get_metrics_collector is not None}"
    )

    persisted_stats = {}
    if persistence_tool:
        try:
            persisted_stats = persistence_tool.get_student_session_stats(
                student_id)
        except Exception as e:
            logger.warning(
                "No se pudieron recuperar métricas persistidas para %s: %s",
                student_id,
                e,
            )

    if get_metrics_collector:
        # Usar métricas reales del colector
        collector = get_metrics_collector()
        all_metrics = collector.get_all_metrics()
        summary_stats = collector.get_summary_stats()

        print(
            f"📊 Métricas obtenidas: {len(all_metrics)} agentes, {summary_stats.get('total_calls', 0)} llamadas totales")

        # Si no hay métricas reales, devolver estructura de respaldo
        if not all_metrics:
            total_interactions = int(
                persisted_stats.get('total_interactions', 0) or 0)
            total_tokens = int(persisted_stats.get('total_tokens', 0) or 0)
            return jsonify({
                'metrics': {
                    'overall': {
                        'total_interactions': total_interactions,
                        'success_rate': 0.0,
                        'average_response_time': '0.0s',
                        'total_tokens_used': total_tokens
                    },
                    'agents': {},
                    'timestamp': datetime.now().isoformat(),
                    'student_id': student_id,
                }
            })

        total_interactions = summary_stats.get('total_calls', 0)
        total_tokens_used = summary_stats.get('total_tokens_used', 0)
        if persisted_stats:
            total_interactions = max(
                int(total_interactions or 0),
                int(persisted_stats.get('total_interactions', 0) or 0),
            )
            total_tokens_used = max(
                int(total_tokens_used or 0),
                int(persisted_stats.get('total_tokens', 0) or 0),
            )

        metrics = {
            'overall': {
                'total_interactions': total_interactions,
                'success_rate': summary_stats.get('overall_success_rate', 0),
                'average_response_time': f"{summary_stats.get('average_response_time', 0):.2f}s",
                'total_tokens_used': total_tokens_used
            },
            'agents': all_metrics,
            'timestamp': datetime.now().isoformat(),
            'student_id': student_id,
        }
    else:
        # Sin colector de métricas disponible
        print("❌ Colector de métricas no disponible")
        total_interactions = int(
            persisted_stats.get('total_interactions', 0) or 0)
        total_tokens = int(persisted_stats.get('total_tokens', 0) or 0)
        return jsonify({
            'metrics': {
                'overall': {
                    'total_interactions': total_interactions,
                    'success_rate': 0.0,
                    'average_response_time': '0.0s',
                    'total_tokens_used': total_tokens
                },
                'agents': {},
                'timestamp': datetime.now().isoformat(),
                'student_id': student_id,
            }
        })
    return jsonify({'metrics': metrics})


@app.route('/api/chat', methods=['POST'])
def chat_with_orchestrator():
    """Endpoint para chatear con el orchestrator agent."""
    try:
        data = request.get_json()
        message = str(data.get('message', '')).strip()
        student_id = (
            session.get('student_id')
            or data.get('student_id')
            or 'demo_student'
        )

        is_valid, error_message, cleaned_message = validate_user_message(
            message)
        if not is_valid:
            return jsonify({'error': error_message}), 400

        # Ejecutar el orchestrator
        result = orchestrator_agent.execute(
            cleaned_message, student_id=student_id)

        return jsonify({
            'response': result['response'],
            'agent': 'orchestrator',
            'phase': result['phase'],
            'logs': result['logs'],
            'tokens': result.get('tokens', {}),
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error en chat con orchestrator: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat/history')
def get_chat_history():
    """Obtener historial de chat del orchestrator."""
    student_id = request.args.get('student_id', 'demo_student')
    history = orchestrator_agent.get_interaction_logs(student_id)
    return jsonify({'history': history[-20:]})  # Últimos 20


@app.route('/api/chat/export-json')
def export_chat_json():
    """Exporta el historial completo de conversación como JSON descargable."""
    student_id = (
        request.args.get('student_id')
        or session.get('student_id')
        or 'demo_student'
    )
    if not persistence_tool:
        return jsonify({'error': 'Persistencia no disponible'}), 503

    since_str = request.args.get('since', '')
    rows = persistence_tool.get_conversation_history(student_id, limit=2000)
    rows = list(reversed(rows))

    # Filtrar solo los mensajes de la sesión activa cuando se proporciona 'since'
    if since_str:
        try:
            from datetime import datetime as _dtp
            # Normalizar: quitar microsegundos extra y 'Z' para comparación simple
            since_str_clean = since_str.rstrip('Z').split('.')[0]
            rows = [
                r for r in rows
                if (r.get('created_at') or '') >= since_str_clean
            ]
        except Exception:
            pass  # Si falla el filtro, exportar todo igualmente

    messages = [
        {
            'role': r['role'],
            'content': r['content'],
            'agent_name': r.get('agent_name') or '',
            'phase': r.get('phase') or '',
            'created_at': r.get('created_at') or '',
        }
        for r in rows
    ]

    export_data = {
        'version': '1.0',
        'student_id': student_id,
        'exported_at': datetime.now().isoformat(),
        'message_count': len(messages),
        'messages': messages,
    }

    response = jsonify(export_data)
    response.headers['Content-Disposition'] = (
        f'attachment; filename="chat_{student_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json"'
    )
    return response


@app.route('/api/chat/import-json', methods=['POST'])
def import_chat_json():
    """Importa un historial de conversación previamente exportado."""
    if not persistence_tool:
        return jsonify({'success': False, 'message': 'Persistencia no disponible'}), 503

    payload = request.get_json(force=True, silent=True)
    if not payload:
        return jsonify({'success': False, 'message': 'JSON inválido o vacío'}), 400

    messages = payload.get('messages', [])
    if not messages:
        return jsonify({'success': False, 'message': 'No hay mensajes en el archivo'}), 400

    import_student_id = payload.get('student_id') or session.get(
        'student_id') or 'demo_student'
    current_user = session.get('student_id')

    # Seguridad: solo importar sobre la propia cuenta o si es admin
    if (current_user
            and current_user not in {'demo', 'demo_student'}
            and import_student_id != current_user
            and session.get('role') != 'admin'):
        import_student_id = current_user

    saved = 0
    for msg in messages:
        role = msg.get('role', '')
        content = msg.get('content', '')
        if not role or not content:
            continue
        ok = persistence_tool.save_conversation_message(
            student_id=import_student_id,
            role=role,
            content=content,
            agent_name=msg.get('agent_name') or None,
            phase=msg.get('phase') or None,
        )
        if ok:
            saved += 1

    return jsonify({
        'success': True,
        'student_id': import_student_id,
        'imported_messages': saved,
        'total_in_file': len(messages),
    })


@app.route('/api/logs', methods=['GET', 'DELETE'])
def get_logs():
    """Obtener logs de interacciones desde la base de datos persistida."""
    if request.method == 'DELETE':
        if not persistence_tool:
            return jsonify({'success': False, 'message': 'Persistencia no disponible'}), 503

        student_id = (
            request.args.get('student_id')
            or session.get('student_id')
            or 'demo_student'
        )

        try:
            deleted_messages = persistence_tool.delete_conversation_history(
                student_id)
            return jsonify({
                'success': True,
                'student_id': student_id,
                'deleted_messages': deleted_messages,
            })
        except Exception as e:
            logger.error(f"Error borrando logs para {student_id}: {e}")
            return jsonify({'success': False, 'message': str(e)}), 500

    import re as _re
    student_id = (
        request.args.get('student_id')
        or session.get('student_id')
    )
    if not persistence_tool or not student_id:
        # Fallback a logs en memoria (solo sesión actual)
        logs = orchestrator_agent.get_interaction_logs(student_id)
        return jsonify({'logs': logs})

    rows = persistence_tool.get_conversation_history(
        student_id, limit=200)
    # La consulta devuelve DESC → invertir para obtener orden cronológico
    rows = list(reversed(rows))

    result = []
    i = 0
    while i < len(rows):
        row = rows[i]
        if row['role'] == 'user':
            user_msg = row['content']
            user_ts = row['created_at']
            user_phase = row.get('phase') or ''
            # Buscar respuesta del asistente siguiente
            asst_row = None
            if i + 1 < len(rows) and rows[i + 1]['role'] == 'assistant':
                asst_row = rows[i + 1]
            agent_name = (
                (asst_row.get('agent_name') or 'OrchestratorAgent')
                if asst_row else 'OrchestratorAgent'
            )
            result.append({
                'timestamp': user_ts,
                'student_id': student_id,
                'phase': user_phase,
                'agent_name': agent_name,
                'user_message': user_msg,
                'agent_response': asst_row['content'] if asst_row else '',
            })
            i += 2 if asst_row else 1
        else:
            # Respuesta de asistente sin mensaje de usuario previo (inicio de sesión)
            result.append({
                'timestamp': row['created_at'],
                'student_id': student_id,
                'phase': row.get('phase') or '',
                'agent_name': row.get('agent_name') or 'OrchestratorAgent',
                'user_message': '',
                'agent_response': row['content'],
            })
            i += 1

    return jsonify({'logs': result})


@app.route('/api/student/history')
def get_student_history():
    """Obtener historial de conversación persistido en la base de datos."""
    student_id = request.args.get('student_id', 'demo_student')
    limit = int(request.args.get('limit', 50))
    if not persistence_tool:
        return jsonify({'error': 'Persistencia no disponible'}), 503
    history = persistence_tool.get_conversation_history(
        student_id, limit=limit)
    return jsonify({'student_id': student_id, 'history': history, 'count': len(history)})


@app.route('/api/student/email', methods=['POST'])
def register_student_email():
    """Registrar o actualizar el email de un estudiante para recordatorios."""
    try:
        data = request.get_json()
        student_id = data.get('student_id', '').strip()
        email = data.get('email', '').strip()
        if not student_id or not email:
            return jsonify({'error': 'student_id y email son obligatorios'}), 400
        if not persistence_tool:
            return jsonify({'error': 'Persistencia no disponible'}), 503
        success = persistence_tool.save_email(
            student_id=student_id,
            email=email,
            name=data.get('name'),
            notify_reminders=data.get('notify_reminders', True),
            notify_progress=data.get('notify_progress', True),
            notify_assessments=data.get('notify_assessments', True)
        )
        if success:
            return jsonify({'message': f'Email registrado: {email}', 'student_id': student_id})
        return jsonify({'error': 'No se pudo guardar el email'}), 500
    except Exception as e:
        logger.error(f"Error registrando email: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/student/email/<student_id>')
def get_student_email(student_id):
    """Obtener el email registrado de un estudiante."""
    if not persistence_tool:
        return jsonify({'error': 'Persistencia no disponible'}), 503
    record = persistence_tool.get_email(student_id)
    if record:
        return jsonify({'email_record': record})
    return jsonify({'error': 'Email no encontrado para este estudiante'}), 404


@app.route('/api/student/session-stats/<student_id>')
def get_student_session_stats(student_id):
    """Obtener estadísticas de sesiones de un estudiante (tokens, interacciones)."""
    if not persistence_tool:
        return jsonify({'error': 'Persistencia no disponible'}), 503
    stats = persistence_tool.get_student_session_stats(student_id)
    return jsonify({'student_id': student_id, 'session_stats': stats})


# =============================================================================
# AUTENTICACIÓN
# =============================================================================

def login_required(f):
    """Decorador que requiere sesión activa. Redirige a /login si no está autenticado."""
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('user_id'):
            # Si es petición API devuelve 401, si es página redirige
            if request.path.startswith('/api/'):
                return jsonify({'error': 'No autenticado'}), 401
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated


@app.route('/login')
def login_page():
    """Página de login."""
    if session.get('user_id'):
        return redirect(url_for('index'))
    return render_template('login.html')


@app.route('/auth/register', methods=['POST'])
def auth_register():
    """Registrar nuevo usuario."""
    if not persistence_tool:
        return jsonify({'error': 'El sistema de usuarios no está disponible. Contacta al administrador.'}), 503
    try:
        data = request.get_json()
        username = (data.get('username') or '').strip()
        email = (data.get('email') or '').strip().lower()
        password = data.get('password') or ''
        name = (data.get('name') or '').strip()

        if not username or not email or not password:
            return jsonify({'error': 'username, email y password son obligatorios'}), 400
        if len(password) < 6:
            return jsonify({'error': 'La contraseña debe tener al menos 6 caracteres'}), 400

        user_id = persistence_tool.create_user(
            username=username, email=email, password=password,
            name=name or username, role='student'
        )
        if not user_id:
            return jsonify({'error': 'El usuario o email ya está registrado'}), 409

        user = persistence_tool.get_user_by_id(user_id)
        session['user_id'] = user_id
        session['username'] = user['username']
        session['name'] = user['name']
        session['role'] = user['role']
        session['student_id'] = user['student_id']
        return jsonify({'message': 'Usuario creado correctamente', 'user': {
            'user_id': user_id,
            'username': user['username'],
            'name': user['name'],
            'role': user['role'],
            'student_id': user['student_id']
        }}), 201
    except Exception as e:
        logger.error(f'Error en registro: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/auth/login', methods=['POST'])
def auth_login():
    """Iniciar sesión."""
    if not persistence_tool:
        return jsonify({'error': 'El sistema de usuarios no está disponible. Contacta al administrador.'}), 503
    try:
        data = request.get_json()
        username_or_email = (data.get('username')
                             or data.get('email') or '').strip()
        password = data.get('password') or ''

        if not username_or_email or not password:
            return jsonify({'error': 'Usuario/email y contraseña son obligatorios'}), 400

        user = persistence_tool.authenticate_user(username_or_email, password)
        if not user:
            return jsonify({'error': 'Credenciales incorrectas'}), 401

        session['user_id'] = user['user_id']
        session['username'] = user['username']
        session['name'] = user['name']
        session['role'] = user['role']
        session['student_id'] = user['student_id']
        return jsonify({'message': 'Sesión iniciada', 'user': {
            'user_id': user['user_id'],
            'username': user['username'],
            'name': user['name'],
            'role': user['role'],
            'student_id': user['student_id']
        }})
    except Exception as e:
        logger.error(f'Error en login: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/auth/logout', methods=['POST', 'GET'])
def auth_logout():
    """Cerrar sesión."""
    session.clear()
    return redirect(url_for('login_page'))


@app.route('/auth/me')
def auth_me():
    """Devuelve los datos del usuario autenticado actualmente."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'No autenticado'}), 401
    return jsonify({'user': {
        'user_id': user_id,
        'username': session.get('username'),
        'name': session.get('name'),
        'role': session.get('role'),
        'student_id': session.get('student_id')
    }})


@app.route('/auth/users')
def auth_list_users():
    """Lista usuarios (solo admin)."""
    if session.get('role') != 'admin':
        return jsonify({'error': 'Acceso denegado'}), 403
    if not persistence_tool:
        return jsonify({'error': 'Persistencia no disponible'}), 503
    users = persistence_tool.list_users()
    return jsonify({'users': users})


@socketio.on('connect')
def handle_connect():
    """Manejar conexión WebSocket."""
    logger.info('Cliente conectado via WebSocket')
    emit('status', {'message': 'Conectado al servidor'})


@socketio.on('disconnect')
def handle_disconnect():
    """Manejar desconexión WebSocket."""
    logger.info('Cliente desconectado')


@socketio.on('chat_message')
def handle_chat_message(data):
    """Manejar mensajes de chat via WebSocket con el orchestrator."""
    try:
        from flask import request as flask_request
        message = str(data.get('message', '')).strip()
        # Preferir el usuario autenticado en la sesión Flask;
        # el cliente puede pasar student_id como fallback (modo demo).
        student_id = (
            session.get('student_id')
            or data.get('student_id')
            or 'demo_student'
        )

        is_valid, error_message, cleaned_message = validate_user_message(
            message)
        if not is_valid:
            emit('error', {'message': error_message})
            return

        # Guardar SID activo para emisiones asíncronas
        orchestrator_agent._active_sid = flask_request.sid

        # Emitir que estamos procesando
        emit('typing', {'agent_name': 'OrchestratorAgent', 'status': True})

        # Ejecutar el orchestrator
        result = orchestrator_agent.execute(
            cleaned_message, student_id=student_id)

        # Dejar de "escribir"
        emit('typing', {'agent_name': 'OrchestratorAgent', 'status': False})

        # Enviar respuesta con información adicional
        emit('chat_response', {
            'response': result['response'],
            'agent_name': 'OrchestratorAgent',
            'phase': result['phase'],
            'logs': result['logs'],
            'tokens': result.get('tokens', {}),
            'tools_used': result.get('tools_used', []),
            'mode': result.get('mode', 'azure_openai'),
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error en WebSocket chat: {e}")
        emit('error', {'message': str(e)})


@app.route('/api/study-plan/<plan_id>', methods=['GET', 'DELETE'])
def api_study_plan(plan_id):
    """Devuelve los datos reales del plan de estudio desde la DB."""
    import sqlite3 as _sqlite3
    from datetime import datetime as _dt, timedelta as _td

    if request.method == 'DELETE':
        if not persistence_tool:
            return jsonify({'success': False, 'message': 'Persistencia no disponible'}), 503

        try:
            # 1. Intentar cargar por plan_id directo
            plan_data = persistence_tool.load_study_plan(plan_id)
            resolved_plan_id = plan_id

            # 2. Fallback: buscar por student_id, igual que el GET handler
            if not plan_data:
                try:
                    import sqlite3 as _sq3
                    with _sq3.connect(persistence_tool.db_path) as _conn3:
                        _cur3 = _conn3.cursor()
                        _cur3.execute(
                            "SELECT plan_data, plan_id FROM study_plans "
                            "WHERE student_id = ? ORDER BY updated_at DESC LIMIT 1",
                            (plan_id,)
                        )
                        _row3 = _cur3.fetchone()
                        if _row3:
                            import json as _json3
                            plan_data = _json3.loads(_row3[0])
                            resolved_plan_id = _row3[1]
                except Exception:
                    pass

            if not plan_data:
                return jsonify({'success': False, 'message': 'Plan no encontrado'}), 404

            # Verificar propiedad: solo el propietario o admin puede borrar
            current_user = session.get('student_id')
            plan_owner = (plan_data.get('student_id')
                          or resolved_plan_id).strip()
            if (current_user
                    and current_user not in {'demo', 'demo_student'}
                    and plan_owner != current_user
                    and session.get('role') != 'admin'):
                return jsonify({
                    'success': False,
                    'message': 'Acceso denegado: no eres el propietario de este plan',
                }), 403

            student_id = plan_owner or ''
            if not student_id:
                return jsonify({
                    'success': False,
                    'message': 'No se pudo resolver el estudiante propietario del plan',
                }), 500

            deleted = persistence_tool.delete_study_plans(
                student_id=student_id,
                plan_id=resolved_plan_id,
            )
            if deleted <= 0:
                return jsonify({'success': False, 'message': 'Plan no encontrado'}), 404
            return jsonify({
                'success': True,
                'student_id': student_id,
                'deleted_plans': deleted,
                'plan_id': resolved_plan_id,
            })
        except Exception as e:
            logger.error(f"Error borrando plan {plan_id}: {e}")
            return jsonify({'success': False, 'message': str(e)}), 500

    try:
        plan_data = None

        resolved_id = plan_id
        if resolved_id in {'demo', 'demo_student'} and session.get('student_id'):
            resolved_id = session.get('student_id')

        # 1. Intentar cargar por plan_id directo
        if persistence_tool:
            plan_data = persistence_tool.load_study_plan(resolved_id)

        # 2. Si no, buscar el plan más reciente por student_id
        if not plan_data and persistence_tool:
            try:
                with _sqlite3.connect(persistence_tool.db_path) as conn:
                    cur = conn.cursor()
                    cur.execute(
                        "SELECT plan_data, plan_id FROM study_plans "
                        "WHERE student_id = ? ORDER BY updated_at DESC LIMIT 1",
                        (resolved_id,)
                    )
                    row = cur.fetchone()
                    if row:
                        import json as _json
                        plan_data = _json.loads(row[0])
                        # Asegurar que el plan_id del JSON coincide con el de la DB
                        if row[1] and not plan_data.get('plan_id'):
                            plan_data['plan_id'] = row[1]
                        elif row[1]:
                            plan_data['plan_id'] = row[1]
            except Exception:
                pass

        # 3. Fallback de demo: devolver el plan más reciente global
        if not plan_data and persistence_tool and plan_id in {'demo', 'demo_student'}:
            try:
                with _sqlite3.connect(persistence_tool.db_path) as conn:
                    cur = conn.cursor()
                    cur.execute(
                        "SELECT plan_data FROM study_plans ORDER BY updated_at DESC LIMIT 1"
                    )
                    row = cur.fetchone()
                    if row:
                        import json as _json
                        plan_data = _json.loads(row[0])
            except Exception:
                pass

        if not plan_data:
            return jsonify({'error': 'Plan no encontrado', 'plan_id': plan_id}), 404

        # Seguridad: verificar propiedad del plan
        current_user = session.get('student_id')
        plan_owner = plan_data.get('student_id', resolved_id)
        if (current_user
                and current_user not in {'demo', 'demo_student'}
                and plan_owner not in {'demo', 'demo_student'}
                and plan_owner != current_user
                and session.get('role') != 'admin'):
            return jsonify({'error': 'Acceso denegado'}), 403

        # Fallback: si el plan existe pero no está estructurado, reconstruirlo.
        if not plan_data.get('sessions') and plan_data.get('study_plan_response'):
            rebuilt = orchestrator_agent._build_structured_study_plan(
                plan_id=plan_data.get('plan_id', plan_id),
                student_id=plan_data.get('student_id', resolved_id),
                certification=plan_data.get('certification', ''),
                study_plan_response=plan_data.get('study_plan_response', ''),
            )
            plan_data.update(rebuilt)
            if persistence_tool:
                try:
                    persistence_tool.save_study_plan(
                        plan_data.get('plan_id', plan_id),
                        plan_data.get('student_id', resolved_id),
                        plan_data,
                    )
                except Exception:
                    pass

        sessions_raw = plan_data.get('sessions', [])
        has_valid_module_url = any(
            '/training/modules/' in (s.get('learn_url') or '')
            for s in sessions_raw
        )
        if sessions_raw and not has_valid_module_url:
            try:
                from src.agents.orchestrator_agent import OrchestratorAgent as _OA
                plan_data['sessions'] = _OA._enrich_sessions_with_learn_urls(
                    sessions_raw,
                    plan_data.get('certification', ''),
                )
                if persistence_tool:
                    persistence_tool.save_study_plan(
                        plan_data.get('plan_id', plan_id),
                        plan_data.get('student_id', resolved_id),
                        plan_data,
                    )
            except Exception as _ee:
                logger.warning(
                    f"No se pudo enriquecer learn_url del plan: {_ee}")

        # 4. Obtener info del estudiante
        student_name = resolved_id
        if persistence_tool:
            try:
                sid = plan_data.get('student_id', resolved_id)
                with _sqlite3.connect(persistence_tool.db_path) as conn:
                    cur = conn.cursor()
                    cur.execute("SELECT name, email FROM users WHERE student_id = ? OR id = ? LIMIT 1",
                                (sid, sid))
                    u = cur.fetchone()
                    if u:
                        student_name = u[0]
                    else:
                        cur.execute("SELECT name FROM users LIMIT 1")
                        u2 = cur.fetchone()
                        if u2:
                            student_name = u2[0]
            except Exception:
                pass

        # 5. Agrupar sesiones por semana
        sessions = plan_data.get('sessions', [])
        start_date_str = plan_data.get('start_date', '')
        try:
            start_dt = _dt.strptime(start_date_str, '%Y-%m-%d')
        except Exception:
            start_dt = _dt.now()

        today = _dt.now().date()
        weeks = {}
        for s in sessions:
            try:
                sd = _dt.strptime(s['session_date'], '%Y-%m-%d')
            except Exception:
                continue
            delta = (sd - start_dt).days
            week_num = max(1, delta // 7 + 1)
            if week_num not in weeks:
                weeks[week_num] = {
                    'week_number': week_num,
                    'start_date': (start_dt + _td(weeks=week_num - 1)).strftime('%d/%m/%Y'),
                    'end_date': (start_dt + _td(weeks=week_num) - _td(days=1)).strftime('%d/%m/%Y'),
                    'sessions': [],
                    'is_current': False,
                    'is_done': False,
                }
            weeks[week_num]['sessions'].append({
                'session_id': s.get('session_id', ''),
                'date': sd.strftime('%d/%m/%Y'),
                'topic': s.get('topic', ''),
                'module': s.get('module_title', ''),
                'duration_min': s.get('duration_minutes', 60),
                'objectives': s.get('objectives', []),
                'completed': s.get('completed', False),
                'learn_url': s.get('learn_url', ''),
            })

        # Marcar semana actual y pasadas
        for wn, wdata in weeks.items():
            try:
                w_start = _dt.strptime(wdata['start_date'], '%d/%m/%Y').date()
                w_end = _dt.strptime(wdata['end_date'], '%d/%m/%Y').date()
                if w_start <= today <= w_end:
                    wdata['is_current'] = True
                elif w_end < today:
                    wdata['is_done'] = True
            except Exception:
                pass

        # 6. Calcular progreso
        total_sessions = len(sessions)
        completed_sessions = sum(
            1 for s in sessions if s.get('completed', False))
        overall_progress = round(
            completed_sessions / total_sessions * 100) if total_sessions else 0

        return jsonify({
            'plan_id': plan_data.get('plan_id', plan_id),
            'plan_name': plan_data.get(
                'plan_name', plan_data.get('certification', 'Plan de estudio')
            ),
            'student_id': plan_data.get('student_id', resolved_id),
            'student_name': student_name,
            'certification': plan_data.get('certification', ''),
            'start_date': start_date_str,
            'target_exam_date': plan_data.get('target_exam_date', ''),
            'total_weeks': max(weeks.keys()) if weeks else 0,
            'weekly_hours': plan_data.get('weekly_hours', 0),
            'total_sessions': total_sessions,
            'completed_sessions': completed_sessions,
            'overall_progress': overall_progress,
            'certification_progress': overall_progress,
            'weeks': [weeks[k] for k in sorted(weeks.keys())],
            'milestones': plan_data.get('milestones', []),
        })

    except Exception as e:
        logger.error(f"Error en api_study_plan: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/study-plans/', defaults={'student_or_plan_id': ''}, methods=['GET'])
@app.route('/api/study-plans/<student_or_plan_id>', methods=['GET', 'DELETE'])
def api_study_plans(student_or_plan_id):
    """Lista los planes de estudio del estudiante (multi-certificación)."""
    if not persistence_tool:
        return jsonify({'success': False, 'message': 'Persistencia no disponible'}), 503

    try:
        resolved_id = (student_or_plan_id or '').strip()
        if not resolved_id:
            resolved_id = session.get('student_id', 'demo_student')

        if resolved_id in {'demo', 'demo_student'} and session.get('student_id'):
            resolved_id = session.get('student_id')

        as_plan = persistence_tool.load_study_plan(resolved_id)
        student_id = as_plan.get(
            'student_id', resolved_id) if as_plan else resolved_id
        if request.method == 'DELETE':
            deleted = persistence_tool.delete_study_plans(
                student_id=student_id)
            return jsonify({
                'success': True,
                'student_id': student_id,
                'deleted_plans': deleted,
            })

        plans = persistence_tool.list_study_plans(student_id)

        return jsonify({
            'success': True,
            'student_id': student_id,
            'plans': plans,
        })
    except Exception as e:
        logger.error(f"Error listando planes de estudio: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/study-plan/<plan_id>/rename', methods=['POST'])
def rename_study_plan(plan_id):
    """Renombra un plan de estudio existente."""
    if not persistence_tool:
        return jsonify({'success': False, 'message': 'Persistencia no disponible'}), 503

    try:
        payload = request.get_json() or {}
        plan_name = (payload.get('plan_name') or '').strip()
        if not plan_name:
            return jsonify({'success': False, 'message': 'Nombre de plan requerido'}), 400

        ok = persistence_tool.update_study_plan_name(plan_id, plan_name)
        if not ok:
            return jsonify({'success': False, 'message': 'Plan no encontrado'}), 404

        return jsonify({'success': True, 'plan_id': plan_id, 'plan_name': plan_name})
    except Exception as e:
        logger.error(f"Error renombrando plan {plan_id}: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/study-plan/<plan_id>/sessions/<session_id>/complete', methods=['POST'])
def complete_study_session(plan_id, session_id):
    """Marca una sesión del plan como completada o pendiente."""
    if not persistence_tool:
        return jsonify({'success': False, 'message': 'Persistencia no disponible'}), 503

    try:
        payload = request.get_json() or {}
        completed = bool(payload.get('completed', False))
        ok = persistence_tool.set_study_session_completed(
            plan_id=plan_id,
            session_id=session_id,
            completed=completed,
        )
        if not ok:
            return jsonify({'success': False, 'message': 'Sesión o plan no encontrado'}), 404

        return jsonify({
            'success': True,
            'plan_id': plan_id,
            'session_id': session_id,
            'completed': completed,
        })
    except Exception as e:
        logger.error(f"Error actualizando sesión {session_id}: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/study-plan/<session_id>')
def study_plan_page(session_id):
    """Página del plan de estudio personalizado."""
    resolved_id = session_id
    if resolved_id in {'demo', 'demo_student'} and session.get('student_id'):
        resolved_id = session.get('student_id')

    # Seguridad: solo el propietario o un admin puede ver el plan de otro usuario
    current_user = session.get('student_id')
    if (current_user
            and current_user not in {'demo', 'demo_student'}
            and resolved_id not in {'demo', 'demo_student'}
            and current_user != resolved_id
            and session.get('role') != 'admin'):
        return redirect(
            url_for('study_plan_page', session_id=current_user)
        )

    return render_template('study_plan.html', session_id=resolved_id)


@app.route('/study-plan')
def study_plan_page_current():
    """Página del plan de estudio del usuario autenticado actual."""
    current_student_id = session.get('student_id', 'demo')
    return redirect(url_for('study_plan_page', session_id=current_student_id))


@app.route('/api/study-plan/<plan_id>/calendar')
def download_study_plan_calendar(plan_id):
    """Genera y descarga un calendario .ics del plan de estudio."""
    if not persistence_tool or not calendar_tool:
        return jsonify({'error': 'Calendario no disponible'}), 503

    try:
        resolved_id = plan_id
        if resolved_id in {'demo', 'demo_student'} and session.get('student_id'):
            resolved_id = session.get('student_id')

        plan_data = persistence_tool.load_study_plan(resolved_id)
        if not plan_data:
            import sqlite3 as _sqlite3
            with _sqlite3.connect(persistence_tool.db_path) as conn:
                cur = conn.cursor()
                cur.execute(
                    "SELECT plan_data FROM study_plans WHERE student_id = ? "
                    "ORDER BY updated_at DESC LIMIT 1",
                    (resolved_id,)
                )
                row = cur.fetchone()
                if row:
                    plan_data = json.loads(row[0])

        if not plan_data:
            return jsonify({'error': 'Plan no encontrado'}), 404

        student_email = ''
        student_id = plan_data.get('student_id', resolved_id)
        if persistence_tool:
            email_record = persistence_tool.get_email(student_id)
            if email_record:
                student_email = email_record.get('email', '')

        calendar_path = calendar_tool.generate_study_plan_calendar(
            plan_data=plan_data,
            student_email=student_email or 'student@example.com',
        )
        return send_file(
            calendar_path,
            as_attachment=True,
            download_name='study_plan.ics',
            mimetype='text/calendar',
        )
    except Exception as e:
        logger.error(f"Error generando calendario del plan: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/study-plan/<plan_id>/activate-email-reminders', methods=['POST'])
def activate_email_reminders(plan_id):
    """Activa avisos por email para el plan actual y envía plan+calendario."""
    if not persistence_tool:
        return jsonify({'success': False, 'message': 'Persistencia no disponible'}), 503

    try:
        payload = request.get_json(force=True, silent=True) or {}
        email = (payload.get('email') or '').strip().lower()
        send_test_email = bool(payload.get('send_test_email', False))
        if not email:
            return jsonify({'success': False, 'message': 'Email requerido'}), 400

        resolved_id = plan_id
        if resolved_id in {'demo', 'demo_student'} and session.get('student_id'):
            resolved_id = session.get('student_id')

        plan_data = None
        try:
            plan_data = persistence_tool.load_study_plan(resolved_id)
        except Exception as db_err:
            logger.error(f"Error cargando plan: {db_err}")
        if not plan_data:
            try:
                import sqlite3 as _sqlite3
                with _sqlite3.connect(persistence_tool.db_path) as conn:
                    cur = conn.cursor()
                    cur.execute(
                        "SELECT plan_data FROM study_plans WHERE student_id = ? "
                        "ORDER BY updated_at DESC LIMIT 1",
                        (resolved_id,)
                    )
                    row = cur.fetchone()
                    if row:
                        plan_data = json.loads(row[0])
            except Exception as fallback_err:
                logger.error(f"Error fallback plan: {fallback_err}")

        if not plan_data:
            return jsonify({'success': False, 'message': 'Plan no encontrado'}), 404

        student_id = plan_data.get('student_id', resolved_id)
        try:
            persistence_tool.save_email(
                student_id=student_id,
                email=email,
                name=session.get('name') or student_id,
                notify_reminders=True,
                notify_progress=True,
                notify_assessments=True,
            )
        except Exception as save_email_err:
            logger.error(f"Error guardando email: {save_email_err}")

        email_sent = False
        test_email_sent = False
        warning_message = ''
        if email_tool and calendar_tool:
            try:
                calendar_path = calendar_tool.generate_study_plan_calendar(
                    plan_data=plan_data,
                    student_email=email,
                )
                email_sent = asyncio.run(email_tool.send_study_plan_email(
                    recipient_email=email,
                    recipient_name=session.get('name') or student_id,
                    plan_data=plan_data,
                    calendar_attachment=calendar_path,
                ))
            except Exception as send_err:
                logger.warning(f"No se pudo enviar email del plan: {send_err}")

            if send_test_email:
                try:
                    test_email_sent = asyncio.run(email_tool.send_welcome_email(
                        recipient_email=email,
                        recipient_name=session.get('name') or student_id,
                        certification=plan_data.get(
                            'certification', 'Microsoft Certification'),
                    ))
                except Exception as test_err:
                    logger.warning(
                        f"No se pudo enviar email de prueba: {test_err}")
                    test_email_sent = False
                if send_test_email and not test_email_sent:
                    warning_message = (
                        'No se pudo enviar el email de prueba real. '
                        'Revisa configuración SMTP/USE_REAL_EMAIL.'
                    )
        elif send_test_email:
            return jsonify({
                'success': False,
                'message': 'Email tool no disponible para enviar prueba.',
                'email_sent': email_sent,
                'test_email_sent': False,
            }), 503

        return jsonify({
            'success': True,
            'email': email,
            'message': 'Avisos por email activados correctamente',
            'warning': warning_message,
            'email_sent': email_sent,
            'test_email_requested': send_test_email,
            'test_email_sent': test_email_sent,
        })
    except Exception as e:
        import traceback
        logger.error(
            f"Error activando avisos por email: {e}\n{traceback.format_exc()}")
        return jsonify({'success': False, 'message': f'Error inesperado: {str(e)}'}), 500


@app.route('/assessment')
def assessment_page():
    """Página de evaluación interactiva."""
    student_id = session.get('student_id', 'demo_student')
    chosen_cert = ''
    if persistence_tool:
        try:
            profile = persistence_tool.load_student_profile(student_id)
            if profile:
                chosen_cert = profile.get('chosen_certification', '')
        except Exception:
            pass
    return render_template(
        'assessment.html',
        chosen_cert=chosen_cert,
        student_id=student_id,
    )


# Almacén temporal de preguntas generadas, keyed por session_token.
# Guarda las preguntas CON la clave correcta server-side para evaluación real.
_question_sessions: dict = {}


def _generate_questions_with_llm(certification: str, student_level: str, count: int) -> list:
    """
    Genera preguntas de opción múltiple reales usando Azure OpenAI (Assessment Agent).

    Devuelve una lista de dicts con: id, question, options (a/b/c/d), correct, explanation.
    Si el LLM no está disponible lanza RuntimeError para que el caller lo maneje.
    """
    import os as _os
    import json as _json
    import re as _re

    endpoint = _os.getenv('AZURE_OPENAI_ENDPOINT', '')
    api_key = _os.getenv('AZURE_OPENAI_API_KEY', '')
    deployment = _os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', 'gpt-4o')
    api_version = _os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-15-preview')

    if not (endpoint and api_key):
        raise RuntimeError('Azure OpenAI no configurado')

    from openai import AzureOpenAI as _AOAI
    client = _AOAI(azure_endpoint=endpoint,
                   api_key=api_key, api_version=api_version)

    system_prompt = (
        "Eres un experto en certificaciones Microsoft. "
        "Genera exactamente las preguntas solicitadas en formato JSON. "
        "Responde ÚNICAMENTE con un array JSON válido, sin texto adicional ni markdown. "
        "Cada elemento del array debe tener:\n"
        "  id: string único (ej: 'q1', 'q2'...)\n"
        "  question: string con la pregunta\n"
        "  options: objeto con claves 'a', 'b', 'c', 'd' y sus textos\n"
        "  correct: letra de la opción correcta ('a', 'b', 'c' o 'd')\n"
        "  explanation: string con la explicación de la respuesta correcta\n"
        "Las preguntas deben cubrir distintos dominios del examen y distintos niveles de Bloom."
    )
    user_prompt = (
        f"Genera {count} preguntas de opción múltiple para la certificación Microsoft {certification}, "
        f"nivel del estudiante: {student_level}. Responde SOLO con el array JSON."
    )

    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=4000,
        temperature=0.4,
    )

    raw = response.choices[0].message.content.strip()
    # Eliminar posibles bloques de código markdown
    raw = _re.sub(r'^```(?:json)?\s*', '', raw, flags=_re.MULTILINE)
    raw = _re.sub(r'\s*```$', '', raw, flags=_re.MULTILINE)
    questions = _json.loads(raw.strip())

    if not isinstance(questions, list):
        raise ValueError(f"El LLM no devolvió un array: {type(questions)}")

    # Normalizar estructura mínima
    normalized = []
    for i, q in enumerate(questions[:count]):
        normalized.append({
            'id': q.get('id', f'q{i+1}'),
            'question': q.get('question', ''),
            'options': q.get('options', {}),
            'correct': q.get('correct', 'a'),
            'explanation': q.get('explanation', '')
        })
    return normalized


@app.route('/api/assessment/generate', methods=['POST'])
def generate_assessment():
    """Generar preguntas de evaluación reales usando Azure OpenAI."""
    try:
        data = request.get_json()
        certification = sanitize_text(str(data.get('certification', 'AZ-900')))
        student_level = sanitize_text(
            str(data.get('student_level', 'beginner')))
        question_count = int(data.get('question_count', 10))

        if question_count < 1 or question_count > 30:
            return jsonify({'success': False, 'message': 'question_count fuera de rango (1-30)'}), 400

        # Generar preguntas reales con LLM
        questions = _generate_questions_with_llm(
            certification, student_level, question_count)

        # Almacenar preguntas server-side (con clave correcta) para la evaluación posterior
        import uuid as _uuid
        session_token = str(_uuid.uuid4())
        _question_sessions[session_token] = {
            'questions': questions,
            'certification': certification,
            'created_at': datetime.now().isoformat()
        }

        # Devolver al cliente SIN la clave correcta (anti-trampa)
        client_questions = []
        for q in questions:
            cq = {k: v for k, v in q.items() if k != 'correct'}
            client_questions.append(cq)

        return jsonify({
            'success': True,
            'session_token': session_token,
            'questions': client_questions,
            'certification': certification,
            'total_questions': len(client_questions)
        })

    except RuntimeError as e:
        logger.warning(
            f"Azure OpenAI no disponible para generate_assessment: {e}")
        return jsonify({'success': False, 'message': 'Azure OpenAI no configurado. Configura AZURE_OPENAI_ENDPOINT y AZURE_OPENAI_API_KEY.'}), 503
    except Exception as e:
        logger.error(f"Error generando evaluación: {e}")
        return jsonify({'success': False, 'message': f'Error generando evaluación: {str(e)}'}), 500


@app.route('/api/assessment/submit', methods=['POST'])
def submit_assessment():
    """Evaluar las respuestas del estudiante contra las claves correctas almacenadas server-side."""
    try:
        data = request.get_json()
        # {question_id: letra_elegida}
        answers = data.get('answers', {})
        session_token = data.get('session_token', '')
        certification = sanitize_text(str(data.get('certification', 'AZ-900')))
        student_id = session.get(
            'student_id', data.get('student_id', 'demo_student'))

        if not answers:
            return jsonify({'success': False, 'message': 'No se recibieron respuestas'}), 400

        # Recuperar preguntas del almacén server-side
        stored = _question_sessions.get(session_token)
        if not stored:
            return jsonify({
                'success': False,
                'message': 'Sesión de evaluación no encontrada o expirada. Genera las preguntas de nuevo.'
            }), 404

        stored_questions = {q['id']: q for q in stored['questions']}

        # Evaluar cada respuesta contra la clave correcta real
        correct_count = 0
        total_questions = len(answers)
        feedback = []

        for question_id, chosen_answer in answers.items():
            question = stored_questions.get(question_id)
            if question:
                correct_letter = question.get('correct', '')
                is_correct = str(chosen_answer).lower().strip() == str(
                    correct_letter).lower().strip()
                if is_correct:
                    correct_count += 1
                feedback.append({
                    'question_id': question_id,
                    'chosen': chosen_answer,
                    'correct_answer': correct_letter,
                    'is_correct': is_correct,
                    'explanation': question.get('explanation', ''),
                    'question_text': question.get('question', '')
                })
            else:
                # Pregunta no encontrada: marcar como incorrecta
                feedback.append({
                    'question_id': question_id,
                    'chosen': chosen_answer,
                    'correct_answer': '?',
                    'is_correct': False,
                    'explanation': 'Pregunta no encontrada en la sesión.'
                })

        score_percentage = (correct_count / total_questions *
                            100) if total_questions > 0 else 0
        passed = score_percentage >= 70

        # Análisis real del Critic Agent
        wrong_topics = [f['question_text'][:60]
                        for f in feedback if not f['is_correct']]
        critic_prompt = (
            f"Analiza los resultados de evaluación del estudiante:\n"
            f"- Certificación: {certification}\n"
            f"- Resultado: {correct_count}/{total_questions} correctas ({score_percentage:.1f}%)\n"
            f"- {'APROBADO ✅' if passed else 'NECESITA REFUERZO ⚠️'}\n"
            f"- Preguntas falladas (extracto): {wrong_topics[:5]}\n\n"
            f"Proporciona feedback constructivo específico con áreas a reforzar y recursos recomendados."
        )
        try:
            critic_result = orchestrator_agent.agents['critic'].execute(
                critic_prompt, student_id)
            critic_analysis = critic_result['response'] if isinstance(
                critic_result, dict) else str(critic_result)
        except Exception as ce:
            logger.warning(f"Critic Agent no disponible: {ce}")
            critic_analysis = f"Resultado: {correct_count}/{total_questions} ({score_percentage:.1f}%). {'¡Aprobado!' if passed else 'Necesitas refuerzo en los temas fallados.'}"

        # Persistir evaluación
        if persistence_tool:
            import uuid as _uuid2
            assessment_id = f"assessment_{student_id}_{_uuid2.uuid4().hex[:8]}"
            persistence_tool.save_assessment(
                assessment_id=assessment_id,
                student_id=student_id,
                assessment_data={
                    'certification': certification,
                    'questions_count': total_questions,
                    'feedback': feedback,
                    'critic_analysis': critic_analysis,
                },
                score=score_percentage,
                passed=passed
            )

        # Limpiar sesión para evitar acumulación en memoria
        _question_sessions.pop(session_token, None)

        return jsonify({
            'success': True,
            'score': correct_count,
            'total': total_questions,
            'percentage': round(score_percentage, 1),
            'passed': passed,
            'feedback': feedback,
            'critic_analysis': critic_analysis
        })

    except Exception as e:
        logger.error(f"Error evaluando respuestas: {e}")
        return jsonify({'success': False, 'message': f'Error evaluando respuestas: {str(e)}'}), 500


# ===== DASHBOARD DE PROGRESO =====

@app.route('/progress')
def progress_dashboard():
    """Dashboard de progreso con gráficos y análisis."""
    return render_template('progress.html')


def _get_active_days(student_id: str) -> int:
    """Calcula los días activos de estudio/conversación del estudiante."""
    if not persistence_tool:
        return 0

    import sqlite3 as _sq

    try:
        with _sq.connect(persistence_tool.db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT COUNT(DISTINCT DATE(created_at)) "
                "FROM conversation_history WHERE student_id = ?",
                (student_id,)
            )
            row = cur.fetchone()
            return int(row[0] or 0)
    except Exception:
        return 0


@app.route('/api/progress/overview')
def get_progress_overview():
    """Obtener datos generales de progreso del estudiante desde la base de datos real."""
    try:
        student_id = session.get(
            'student_id', request.args.get('student_id', 'anonymous'))

        # --- Datos reales desde SQLite ---
        import sqlite3 as _sq

        total_study_hours = 0.0
        overall_progress = 0
        certifications_completed = 0
        average_score = 0.0
        # Scores de las últimas evaluaciones (hasta 7)
        study_trend = []
        assessments_raw = []

        if persistence_tool:
            db = persistence_tool.db_path
            try:
                with _sq.connect(db) as conn:
                    cur = conn.cursor()

                    # Tiempo total de estudio en horas
                    cur.execute(
                        "SELECT COALESCE(SUM(time_spent_minutes),0) FROM study_progress WHERE student_id=?", (student_id,))
                    total_study_hours = round((cur.fetchone()[0] or 0) / 60, 1)

                    # Módulos completados / totales → porcentaje de progreso
                    cur.execute(
                        "SELECT COUNT(*), SUM(CASE WHEN completed=1 THEN 1 ELSE 0 END) FROM study_progress WHERE student_id=?", (student_id,))
                    row = cur.fetchone()
                    total_mod, done_mod = (row[0] or 0), (row[1] or 0)
                    overall_progress = round(
                        done_mod / total_mod * 100) if total_mod > 0 else 0

                    # Evaluaciones aprobadas
                    cur.execute(
                        "SELECT COUNT(*) FROM assessments WHERE student_id=? AND passed=1", (student_id,))
                    certifications_completed = cur.fetchone()[0] or 0

                    # Puntuación media
                    cur.execute(
                        "SELECT AVG(score) FROM assessments WHERE student_id=? AND score IS NOT NULL", (student_id,))
                    avg = cur.fetchone()[0]
                    average_score = round(avg, 1) if avg else 0.0

                    # Tendencia: las últimas 7 puntuaciones de evaluaciones
                    cur.execute(
                        "SELECT score FROM assessments WHERE student_id=? AND score IS NOT NULL ORDER BY created_at DESC LIMIT 7", (student_id,))
                    scores_desc = [r[0] for r in cur.fetchall()]
                    study_trend = list(reversed(scores_desc)
                                       )  # orden cronológico

                    # Módulos con score para weak/strong topics
                    cur.execute(
                        "SELECT module_id, score FROM study_progress WHERE student_id=? AND score IS NOT NULL ORDER BY score", (student_id,))
                    assessments_raw = cur.fetchall()
            except Exception as db_err:
                logger.warning(f"Error leyendo progreso de DB: {db_err}")

        active_days = _get_active_days(student_id)

        # Weak / strong topics desde módulos reales
        weakest_topics = [m[0]
                          for m in assessments_raw[:3]] if assessments_raw else []
        strongest_topics = [m[0] for m in assessments_raw[-3:]
                            [::-1]] if len(assessments_raw) >= 3 else []

        # Determinar próximo hito desde el plan de estudio activo
        next_milestone = 'Primera evaluación'
        if persistence_tool:
            profile = persistence_tool.load_student_profile(student_id)
            if profile and profile.get('chosen_certification'):
                next_milestone = f"Examen {profile['chosen_certification']}"

        data = {
            'overall_progress': overall_progress,
            'certifications_completed': certifications_completed,
            'total_study_hours': total_study_hours,
            'active_days': active_days,
            'average_score': average_score,
            'weakest_topics': weakest_topics or ['Sin datos aún'],
            'strongest_topics': strongest_topics or ['Sin datos aún'],
            'next_milestone': next_milestone,
            'study_trend': study_trend or []
        }

        data['critic_analysis'] = (
            f"Progreso: {overall_progress}%. "
            f"{'¡Buen ritmo!' if overall_progress > 50 else 'Sigue practicando para mejorar.'}"
        )

        return jsonify({'success': True, 'data': data})

    except Exception as e:
        logger.error(f"Error obteniendo datos de progreso: {e}")
        return jsonify({'success': False, 'message': f'Error obteniendo datos: {str(e)}'}), 500


@app.route('/api/progress/detailed')
def get_detailed_progress():
    """Obtener datos detallados de progreso por certificación desde la base de datos real."""
    try:
        certification = request.args.get('certification', 'AZ-900')
        student_id = session.get(
            'student_id', request.args.get('student_id', 'anonymous'))

        import sqlite3 as _sq

        modules_progress = []
        assessment_history = []
        study_sessions = []

        if persistence_tool:
            db = persistence_tool.db_path
            try:
                with _sq.connect(db) as conn:
                    cur = conn.cursor()

                    # Progreso real por módulo
                    cur.execute("""
                        SELECT module_id,
                               CASE WHEN completed=1 THEN 100 ELSE
                                   CASE WHEN time_spent_minutes > 0 THEN MIN(time_spent_minutes * 2, 99) ELSE 0 END
                               END as progress,
                               COALESCE(score, 0) as score
                        FROM study_progress
                        WHERE student_id=?
                        ORDER BY updated_at DESC
                        LIMIT 20
                    """, (student_id,))
                    for row in cur.fetchall():
                        modules_progress.append({
                            'name': row[0],
                            'progress': int(row[1]),
                            'score': round(float(row[2]), 1)
                        })

                    # Historial real de evaluaciones
                    cur.execute("""
                        SELECT created_at, score, assessment_data
                        FROM assessments
                        WHERE student_id=?
                        ORDER BY created_at DESC
                        LIMIT 20
                    """, (student_id,))
                    for row in cur.fetchall():
                        ts = row[0][:10] if row[0] else ''
                        score_val = round(
                            float(row[1]), 1) if row[1] is not None else 0
                        try:
                            adata = json.loads(row[2]) if row[2] else {}
                            cert_name = adata.get(
                                'certification', certification)
                        except Exception:
                            cert_name = certification
                        assessment_history.append({
                            'date': ts,
                            'score': score_val,
                            'certification': cert_name
                        })

                    # Sesiones de estudio reales
                    cur.execute("""
                        SELECT DATE(updated_at),
                               COALESCE(time_spent_minutes, 0),
                               module_id
                        FROM study_progress
                        WHERE student_id=?
                        ORDER BY updated_at DESC
                        LIMIT 20
                    """, (student_id,))
                    for row in cur.fetchall():
                        study_sessions.append({
                            'date': row[0] or '',
                            'duration': round(float(row[1]) / 60, 2),
                            'topics': [row[2]] if row[2] else []
                        })

            except Exception as db_err:
                logger.warning(
                    f"Error leyendo progreso detallado de DB: {db_err}")

        detailed_data = {
            'certification': certification,
            'modules_progress': modules_progress,
            'assessment_history': assessment_history,
            'study_sessions': study_sessions
        }

        return jsonify({'success': True, 'data': detailed_data})

    except Exception as e:
        logger.error(f"Error obteniendo datos detallados: {e}")
        return jsonify({'success': False, 'message': f'Error obteniendo datos detallados: {str(e)}'}), 500


@app.route('/api/progress/insights')
def get_progress_insights():
    """Obtener insights y recomendaciones personalizadas basados en datos reales del estudiante."""
    try:
        student_id = session.get(
            'student_id', request.args.get('student_id', 'anonymous'))

        import sqlite3 as _sq

        # Recopilar datos reales del estudiante de SQLite
        weak_modules = []
        strong_modules = []
        scores_list = []
        avg_session_minutes = 0.0
        chosen_cert = ''

        if persistence_tool:
            db = persistence_tool.db_path
            try:
                with _sq.connect(db) as conn:
                    cur = conn.cursor()
                    # Módulos con score, ordenados de menor a mayor
                    cur.execute("""
                        SELECT module_id, score FROM study_progress
                        WHERE student_id=? AND score IS NOT NULL
                        ORDER BY score ASC
                    """, (student_id,))
                    all_mod = cur.fetchall()
                    weak_modules = [r[0] for r in all_mod[:3]]
                    strong_modules = [r[0] for r in all_mod[-3:][::-1]]

                    # Historial de scores de evaluaciones
                    cur.execute(
                        "SELECT score FROM assessments WHERE student_id=? AND score IS NOT NULL ORDER BY created_at DESC LIMIT 10", (student_id,))
                    scores_list = [round(r[0], 1) for r in cur.fetchall()]

                    # Duración media de sesiones de estudio
                    cur.execute(
                        "SELECT AVG(time_spent_minutes) FROM study_progress WHERE student_id=? AND time_spent_minutes > 0", (student_id,))
                    avg_row = cur.fetchone()
                    avg_session_minutes = round(float(avg_row[0] or 0), 1)

            except Exception as db_err:
                logger.warning(f"Error leyendo datos para insights: {db_err}")

            profile = persistence_tool.load_student_profile(student_id)
            if profile:
                chosen_cert = profile.get('chosen_certification', '')

        active_days = _get_active_days(student_id)
        avg_score = round(sum(scores_list) / len(scores_list),
                          1) if scores_list else 0

        # Construir contexto real para los agentes
        context_assessment = (
            f"Analiza los patrones de aprendizaje reales del estudiante:\n"
            f"- Certificación objetivo: {chosen_cert or 'por definir'}\n"
            f"- Módulos débiles (menor score): {weak_modules or 'Sin datos aún'}\n"
            f"- Módulos fuertes (mayor score): {strong_modules or 'Sin datos aún'}\n"
            f"- Histórico de evaluaciones (scores): {scores_list or 'Sin evaluaciones aún'}\n"
            f"- Puntuación media: {avg_score}%\n"
            f"- Duración media de sesiones: {avg_session_minutes} min\n\n"
            f"Proporciona insights específicos y accionables. Incluye una predicción realista del tiempo hasta el examen."
        )
        context_engagement = (
            f"El estudiante acumula {active_days} días activos de estudio.\n"
            f"Puntuación media de evaluaciones: {avg_score}%.\n"
            f"Módulos que necesitan refuerzo: {weak_modules or 'Sin datos aún'}.\n"
            f"Proporciona consejos de motivación personalizados y un plan de acción de 3 pasos."
        )

        run_agents = request.args.get('ai', 'false').lower() == 'true'

        if run_agents:
            # Assessment Agent → patrones de aprendizaje
            try:
                assessment_result = orchestrator_agent.agents['assessment'].execute(
                    context_assessment, student_id)
                learning_patterns = assessment_result['response'] if isinstance(
                    assessment_result, dict) else str(assessment_result)
            except Exception:
                learning_patterns = (
                    f"Puntuación media: {avg_score}%. "
                    f"Módulos a reforzar: {', '.join(weak_modules) or 'sin datos'}."
                )

            # Engagement Agent → motivación
            try:
                engagement_result = orchestrator_agent.agents['engagement'].execute(
                    context_engagement, student_id)
                motivation_tips = engagement_result['response'] if isinstance(
                    engagement_result, dict) else str(engagement_result)
            except Exception:
                motivation_tips = (
                    f"Días activos: {active_days}. Mantén el ritmo de estudio."
                )
        else:
            learning_patterns = (
                f"Puntuación media: {avg_score}%. "
                f"Módulos a reforzar: {', '.join(weak_modules) or 'sin datos'}."
            )
            motivation_tips = (
                f"Constancia actual: {active_days} días activos. "
                "Mantén un ritmo estable de estudio."
            )

        # Acciones recomendadas basadas en datos reales
        recommended_actions = []
        if weak_modules:
            recommended_actions.append(
                f"Refuerza los módulos con menor puntuación: {', '.join(weak_modules[:2])}")
        if avg_score < 70:
            recommended_actions.append(
                "Tu media está por debajo del umbral de aprobado (70%). Practica más simulacros.")
        elif avg_score >= 70:
            recommended_actions.append(
                f"Estás por encima del umbral de aprobado ({avg_score}%). ¡Mantén el ritmo!")
        if active_days >= 7:
            recommended_actions.append(
                f"Excelente constancia: {active_days} días activos."
            )
        if avg_session_minutes < 30 and avg_session_minutes > 0:
            recommended_actions.append(
                "Tus sesiones son cortas (< 30 min). Considera bloques de 45-60 min para mayor retención.")
        if not recommended_actions:
            recommended_actions.append(
                f"Completa tu perfil y genera tu primer plan de estudio para {chosen_cert or 'tu certificación objetivo'}.")

        # Predicción basada en datos reales
        if avg_score >= 80 and active_days >= 7:
            predicted_outcome = (
                f"Con {avg_score}% de media y {active_days} días activos, "
                "estás listo/a. Programa tu examen en 2-3 semanas."
            )
        elif avg_score >= 70:
            predicted_outcome = f"Con {avg_score}% de media, necesitas consolidar un poco más. Estimado: 3-4 semanas más de práctica."
        elif avg_score > 0:
            predicted_outcome = f"Con {avg_score}% de media, refuerza los módulos débiles. Estimado: 6-8 semanas para estar listo/a."
        else:
            predicted_outcome = f"Genera tu primer plan de estudio para {chosen_cert or 'tu certificación'} y empieza a trackear tu progreso."

        insights = {
            'learning_patterns': learning_patterns,
            'motivation_tips': motivation_tips,
            'recommended_actions': recommended_actions,
            'predicted_outcome': predicted_outcome
        }

        return jsonify({'success': True, 'data': insights})

    except Exception as e:
        logger.error(f"Error obteniendo insights: {e}")
        return jsonify({'success': False, 'message': f'Error obteniendo insights: {str(e)}'}), 500


if __name__ == '__main__':
    # Iniciar servidor
    logger.info("🚀 Iniciando servidor Flask en puerto 5033...")
    socketio.run(app, host='0.0.0.0', port=5033, debug=True)
