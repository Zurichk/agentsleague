"""Orchestrator agents extracted from dashboard app for modular reuse."""

import logging
import re
import time
from datetime import datetime, timedelta
from urllib.parse import quote_plus

try:
    import httpx as _httpx
except ImportError:  # pragma: no cover
    _httpx = None  # type: ignore[assignment]

try:
    from src.agents.metrics import get_metrics_collector
except ImportError:
    try:
        from metrics import get_metrics_collector
    except ImportError:
        get_metrics_collector = None

try:
    from src.utils.logger import (
        get_orchestrator_logger,
        TraceContext,
        log_agent_action,
        log_workflow_transition,
        log_performance_metrics,
        log_agent_response,
        log_error,
    )
except ImportError:
    def get_orchestrator_logger():
        return logging.getLogger("orchestrator")

    def log_agent_action(logger, agent_name, action, message, extra=None, trace_context=None):
        logger.info("[%s] %s: %s", agent_name, action, message)

    def log_workflow_transition(logger, trace_context, transition_type, message, extra=None):
        logger.info("[%s] %s", transition_type, message)

    def log_performance_metrics(logger, trace_context, operation, duration, extra=None):
        logger.info("[PERF] %s: %.2fs", operation, duration)

    def log_agent_response(logger, agent_name, action, message, extra=None, trace_context=None):
        logger.info("[%s] Response: %s", agent_name, message)

    def log_error(logger, component, operation, message, extra=None, trace_context=None):
        logger.error("[%s] %s ERROR: %s", component, operation, message)

    class TraceContext:
        def __init__(self, session_id="", user_id="", operation=""):
            self.session_id = session_id
            self.user_id = user_id
            self.operation = operation
            self.trace_id = f"{session_id}_{user_id}_{operation}"

try:
    from src.tools.persistence import persistence_tool
except Exception:
    try:
        from src.tools.persistence import PersistenceTool
        persistence_tool = PersistenceTool()
    except Exception:
        persistence_tool = None

logger = logging.getLogger(__name__)


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
            logging.getLogger(__name__).debug(
                "Azure OpenAI client inicializado para agente %s",
                self.name,
            )
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

    def __init__(self, socketio_client=None):
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

        self.logger.info("Modo runtime habilitado: Azure OpenAI")

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
            self.logger.debug("Agente %s configurado", name)

        self.conversation_state = {}
        self.interaction_logs = []
        # SID del socket activo (actualizado por el handler)
        self._active_sid: str = ''
        self.socketio_client = socketio_client

        log_agent_action(self.logger, "OrchestratorAgent", "initialization",
                         "Orchestrator initialized", "Mode: azure_openai")

    def set_socketio(self, socketio_client) -> None:
        """Actualiza el cliente SocketIO para emisiones fuera de contexto de request."""
        self.socketio_client = socketio_client

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
                if self._active_sid and self.socketio_client:
                    self.socketio_client.emit('agent_active', {
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
                if self._active_sid and self.socketio_client:
                    self.socketio_client.emit('partial_response', payload,
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

    @staticmethod
    def _extract_certification_code(text: str) -> str | None:
        """Extrae código de certificación tipo MB-800, AZ-900, AI-102."""
        match = re.search(
            r'\b([A-Z]{2,3}-\d{3,4}[A-Z0-9]?)\b', text or '', re.IGNORECASE)
        if not match:
            return None
        return match.group(1).upper()

    def _should_update_certification_goal(
        self,
        message: str,
        detected_topic: str,
        state: dict,
    ) -> bool:
        """Determina si se debe actualizar la certificación objetivo."""
        message_code = self._extract_certification_code(message)
        topic_code = self._extract_certification_code(detected_topic)
        if message_code or topic_code:
            return True

        current_goal = (state.get('chosen_certification') or '').strip()
        if current_goal:
            return False

        return bool((detected_topic or '').strip())

    @staticmethod
    def _parse_replan_choice(message: str) -> str:
        """Clasifica decisión post-evaluación: new_plan | continue_plan | unknown."""
        normalized = (message or '').strip().lower()

        continue_terms = [
            'continuar', 'seguir', 'plan actual', 'itinerario actual',
            'mantener plan', 'seguir igual', 'no cambiar',
        ]
        new_plan_terms = [
            'nuevo plan', 'refinar', 'ajustar', 'preparar', 'refuerzo',
            'replan', 'sí', 'si', 'ok', 'adelante', 'basado en evaluacion',
        ]

        if any(term in normalized for term in continue_terms):
            return 'continue_plan'
        if any(term in normalized for term in new_plan_terms):
            return 'new_plan'
        return 'unknown'

    def _build_reinforcement_context(self, state: dict, user_message: str) -> str:
        """Construye contexto de preparación manteniendo certificación y brechas."""
        chosen_cert = state.get('chosen_certification',
                                '') or 'la certificación objetivo actual'
        last_assessment = state.get('last_assessment_feedback', '')
        last_curator = state.get('curator_data', {}).get(
            'response', '') if state.get('curator_data') else ''

        return (
            f"El estudiante quiere continuar preparación para: {chosen_cert}.\n"
            f"Mensaje actual del estudiante: {user_message}.\n\n"
            f"Mantén la certificación objetivo salvo que el estudiante pida explícitamente cambiarla.\n"
            f"Último análisis de evaluación (brechas):\n{last_assessment}\n\n"
            f"Contexto previo de curator:\n{last_curator}"
        )

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
            chosen_certification = state.get('chosen_certification', '')
            next_step = state.get('next_step', '')
            system_prompt = (
                "Eres un clasificador de intención para el Orchestrator de AEP CertMaster. "
                "Debes seleccionar exactamente UNA categoría y devolver SOLO JSON válido. "
                "No devuelvas markdown ni texto adicional.\n\n"
                "Salida obligatoria:\n"
                "{\"intent\":\"preparation|assessment|certification|confirm|study_plan|greeting|other\","
                "\"confidence\":0.0-1.0,"
                "\"topic\":\"texto breve o vacío\"}\n\n"
                "Reglas de decisión:\n"
                "1) Si el usuario responde una confirmación contextual (sí/ok/adelante) usa 'confirm'.\n"
                "2) Si pide test, quiz, evaluar, corregir respuestas usa 'assessment'.\n"
                "3) Si pide roadmap/certificación sugerida usa 'certification'.\n"
                "4) Si pide crear/refinar/continuar preparación usa 'preparation'.\n"
                "5) Si solo pide ver su plan usa 'study_plan'.\n"
                "6) Si es saludo simple usa 'greeting'.\n"
                "7) Si no aplica, usa 'other'.\n"
                "8) Conserva el contexto: evita inferir nuevo tema si el mensaje es corto y ya existe una certificación activa.\n\n"
                f"Contexto: phase={current_phase}; next_step={next_step}; chosen_certification={chosen_certification or 'none'}"
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
            self.logger.debug(
                "Intent LLM: %s -> %s (confidence=%s)",
                message[:60],
                intent,
                result.get('confidence', '?'),
            )
            return {
                'intent': intent,
                'confidence': float(result.get('confidence', 0.8)),
                'topic': result.get('topic', ''),
            }

        except Exception as e:
            self.logger.warning("Intent LLM fallback por error: %s", e)
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
            r'^\s*(?:#+\s*)?Semana\s+(\d+)\s*(?:[—\-:|]\s*)?(.*)$',
            re.IGNORECASE,
        )
        day_re = re.compile(
            r'^\s*(Lunes|Martes|Mi[eé]rcoles|Jueves|Viernes|S[áa]bado|Domingo)'
            r'\s*(?:\((\d+)\s*h\))?\s*[:\-]\s*(.+)$',
            re.IGNORECASE,
        )
        bullet_re = re.compile(r'^\s*(?:[-*•]|\d+\.)\s+(.+)$')
        duration_re = re.compile(
            r'(\d+)\s*(?:h|hora|horas|min|mins|minutos)\b',
            re.IGNORECASE,
        )

        week_bullet_index: dict[int, int] = {}
        default_days = [0, 2, 4]

        def _clean_line(raw_line: str) -> str:
            value = raw_line.strip()
            value = re.sub(r'\*\*(.*?)\*\*', r'\1', value)
            value = re.sub(r'`([^`]*)`', r'\1', value)
            value = re.sub(r'\[(.*?)\]\((.*?)\)', r'\1', value)
            return value.strip()

        def _infer_minutes(activity: str, default_minutes: int = 90) -> int:
            match = duration_re.search(activity)
            if not match:
                return default_minutes
            value = int(match.group(1))
            token = activity[match.start():match.end()].lower()
            if 'h' in token or 'hora' in token:
                return min(max(value * 60, 45), 180)
            return min(max(value, 30), 180)

        def _append_session(
            week_no: int,
            module_title: str,
            activity: str,
            weekday: int | None,
            duration_minutes: int,
        ) -> None:
            nonlocal sessions
            if not activity:
                return

            if weekday is None:
                idx = week_bullet_index.get(week_no, 0)
                weekday = default_days[idx % len(default_days)]
                week_bullet_index[week_no] = idx + 1

            session_date = next_monday + timedelta(
                days=(week_no - 1) * 7 + weekday
            )

            clean_activity = _clean_line(activity)
            sessions.append({
                'session_id': f"{plan_id}_w{week_no}_{len(sessions) + 1}",
                'session_date': session_date.strftime('%Y-%m-%d'),
                'topic': clean_activity,
                'module_title': module_title or f"Semana {week_no}",
                'duration_minutes': duration_minutes,
                'objectives': [
                    f"Completar: {clean_activity}",
                    "Tomar notas clave",
                    "Validar comprensión con autoevaluación breve",
                ],
                'completed': False,
                'learn_url': '',
            })

        for raw_line in text.splitlines():
            line = _clean_line(raw_line)
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
                hours_raw = day_match.group(2)
                activity = day_match.group(3).strip()
                day_offset = day_map.get(day_name.lower(), 0)
                duration_minutes = (
                    int(hours_raw) * 60
                    if hours_raw else _infer_minutes(activity, 90)
                )
                _append_session(
                    week_no=week_number,
                    module_title=week_title,
                    activity=activity,
                    weekday=day_offset,
                    duration_minutes=duration_minutes,
                )
                continue

            bullet_match = bullet_re.match(line)
            if bullet_match:
                activity = bullet_match.group(1).strip()
                _append_session(
                    week_no=week_number,
                    module_title=week_title,
                    activity=activity,
                    weekday=None,
                    duration_minutes=_infer_minutes(activity, 90),
                )
                if 'hito' in activity.lower() or 'simulacro' in activity.lower():
                    milestones.append({
                        'title': f"Semana {week_number}",
                        'description': activity,
                        'target_date': '',
                        'achieved': False,
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
            fallback_topics = [
                "Repaso de dominios oficiales de la certificación",
                "Práctica guiada con laboratorio/sandbox",
                "Simulacro corto + análisis de errores",
                "Refuerzo de temas débiles detectados",
                "Simulacro completo de preparación",
                "Ajuste final y checklist pre-examen",
            ]
            for idx, topic in enumerate(fallback_topics, start=1):
                week_no = (idx - 1) // 3 + 1
                day_offset = default_days[(idx - 1) % len(default_days)]
                _append_session(
                    week_no=week_no,
                    module_title=f"Semana {week_no}",
                    activity=topic,
                    weekday=day_offset,
                    duration_minutes=120,
                )

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

    @staticmethod
    def _enrich_sessions_with_learn_urls(
        sessions: list,
        certification: str,
    ) -> list:
        """
        Enriquece las sesiones del plan con URLs reales de módulos de Microsoft Learn.

        Consulta la API del catálogo de Learn de forma síncrona.
        Para cada sesión busca el módulo con mayor coincidencia en el título.
        Si no se encuentra coincidencia de alta confianza (>= 3 tokens), devuelve
        una URL de búsqueda oficial con filtro de producto cuando sea posible.

        Args:
            sessions: Lista de sesiones del plan (dicts).
            certification: Código de certificación objetivo (p.ej. "MB-800").

        Returns:
            Lista de sesiones con el campo ``learn_url`` relleno.
        """
        # Mapa de certificaciones a identificadores de producto en Microsoft Learn
        AEP_CERT_PRODUCT_MAP: dict[str, str] = {
            "MB-800": "dynamics-business-central",
            "MB-300": "dynamics-business-central",
            "MB-335": "dynamics-supply-chain-management",
            "MB-700": "dynamics-finance",
            "AZ-900": "azure",
            "AZ-104": "azure",
            "AZ-204": "azure",
            "AZ-305": "azure",
            "AI-102": "azure-ai-services",
            "DP-900": "azure",
            "DP-203": "azure",
            "MS-700": "microsoft-teams",
            "PL-100": "power-platform",
            "PL-400": "power-platform",
            "PL-900": "power-platform",
            "SC-900": "azure",
        }

        cert_upper = (certification or "").upper().strip()
        product_filter = AEP_CERT_PRODUCT_MAP.get(cert_upper, "")

        def _search_url(topic: str) -> str:
            """Construye URL de búsqueda con filtro de producto si está disponible."""
            terms = quote_plus(topic[:80])
            base = (
                f"https://learn.microsoft.com/en-us/training/browse/"
                f"?terms={terms}"
            )
            return f"{base}&products={product_filter}" if product_filter else base

        if not sessions:
            return sessions

        if _httpx is None:
            for s in sessions:
                topic = s.get("topic", "")
                s["learn_url"] = _search_url(topic) if topic else ""
            return sessions

        # --- Obtener catálogo de módulos vía API real ---
        # Se pide en español (es-es) porque los tópicos del plan también están
        # en español. Así los tokens coinciden mejor (p.ej. "introducción" ↔
        # "Introducción a Dynamics 365 Business Central").
        catalog_url = (
            "https://learn.microsoft.com/api/catalog/"
            "?type=modules&locale=es-es"
        )
        modules: list = []
        try:
            with _httpx.Client(timeout=15.0, follow_redirects=True) as client:
                resp = client.get(catalog_url)
                resp.raise_for_status()
                modules = resp.json().get("modules", [])
        except Exception:
            pass

        # Pre-filtrar por certificación cuando sea posible para acelerar búsqueda
        cert_modules = (
            [
                m for m in modules
                if cert_upper in m.get("uid", "").upper()
                or cert_upper in " ".join(m.get("products", [])).upper()
                or cert_upper in m.get("title", "").upper()
            ]
            if cert_upper
            else []
        )
        # Si hay poco resultado de cert, ampliar con filtro de producto
        if len(cert_modules) < 5 and product_filter:
            cert_modules = [
                m for m in modules
                if product_filter in " ".join(m.get("products", [])).lower()
            ]
        search_pool = cert_modules if len(cert_modules) >= 3 else modules

        # Stop-words de contexto: palabras que aparecen en casi todos los
        # módulos de la certificación y no ayudan a discriminar.
        _AEP_CTX_STOPWORDS: set[str] = {
            'dynamics', 'business', 'central', 'microsoft', 'azure',
            '365', 'learn', 'your', 'this', 'that', 'with', 'from',
            'into', 'and', 'the', 'are', 'can', 'will', 'use', 'using',
            'how', 'work', 'course', 'training', 'power', 'platform',
            # Español
            'dynamics', 'para', 'con', 'del', 'los', 'las', 'una',
            'este', 'esta', 'como', 'sus', 'cada', 'semana', 'modulo',
            'recursos', 'practicas', 'horas', 'dias', 'minutos', 'dias',
        }
        # Stop-words adicionales por certificación (palabras que están en el
        # 100% de los módulos de esa cert y no discriminan nada)
        _CERT_EXTRA_STOPWORDS: dict[str, set[str]] = {
            'MB-800': {'business', 'central', 'dynamics'},
            'MB-300': {'business', 'central', 'dynamics'},
            'MB-335': {'supply', 'chain', 'dynamics'},
            'MB-700': {'finance', 'dynamics'},
            'AZ-900': {'azure', 'cloud'},
            'AZ-104': {'azure'},
            'AZ-204': {'azure'},
            'AZ-305': {'azure'},
            'AI-102': {'azure'},
            'PL-100': {'power', 'platform'},
            'PL-400': {'power', 'platform'},
            'PL-900': {'power', 'platform'},
            'MS-700': {'teams', 'microsoft'},
        }
        ctx_stop = _AEP_CTX_STOPWORDS | _CERT_EXTRA_STOPWORDS.get(
            cert_upper, set())

        import unicodedata as _ud

        def _norm(text: str) -> str:
            """Normaliza texto: minúsculas, quita tildes/diéresis."""
            nfkd = _ud.normalize('NFKD', text.lower())
            return ''.join(c for c in nfkd if not _ud.combining(c))

        def _score(module: dict, tokens: list[str]) -> int:
            """
            Puntúa un módulo usando coincidencias ponderadas y normalizadas.
            Título = 2 pts/token, resumen = 1 pt/token.
            También prueba sin la 's' final (plural→singular) y los primeros
            6 caracteres como raíz verbal española (migracion↔migrar, etc.).
            """
            title = _norm(module.get("title", ""))
            summary = _norm(module.get("summary", ""))
            score = 0
            for t in tokens:
                tn = _norm(t)
                # ── forma exacta ──
                in_title = tn in title
                in_summary = tn in summary
                # ── singular (quitar 's' final) ──
                tn_sing = tn[:-1] if tn.endswith('s') and len(tn) > 4 else tn
                if not in_title:
                    in_title = tn_sing in title
                if not in_summary:
                    in_summary = tn_sing in summary
                # ── raíz verbal española (primeros 5 chars para palabras ≥7) ──
                #    cubre: migra(cion)↔migrar, intro(duccion)↔introducir,
                #           confi(guracion)↔configurar, factu(racion)↔factura
                stem_match = False
                if not in_title and not in_summary and len(tn) >= 7:
                    stem = tn[:5]
                    stem_match = stem in title or stem in summary
                if in_title:
                    score += 2
                elif in_summary:
                    score += 1
                elif stem_match:
                    score += 1   # coincidencia por raíz verbal
            return score

        def _normalize_url(url: str) -> str:
            if not url:
                return ""
            if url.startswith("/"):
                full = f"https://learn.microsoft.com{url}"
            else:
                full = url
            # Eliminar el parámetro de seguimiento que añade la API
            return full.split("?")[0].rstrip("/")

        # Con el scoring ponderado (título=2, resumen=1, stem=1), el umbral mínimo
        # de 3 garantiza al menos un token específico en el título o combinación
        # de varios en el resumen. Para el pool general se exige 4.
        score_threshold = 3 if len(cert_modules) >= 3 else 4

        for session in sessions:
            topic: str = session.get("topic", "")
            if not topic:
                continue

            # Tokens discriminantes: normalizar (quitar tildes) y excluir
            # stop-words del contexto de la certificación.
            tokens = [
                _norm(t) for t in topic.split()
                if len(t) > 3 and _norm(t) not in ctx_stop
            ]
            if not tokens:
                tokens = [_norm(t) for t in topic.split() if len(t) > 3]

            best_module = None
            best_score = 0
            for module in search_pool:
                score = _score(module, tokens)
                if score > best_score:
                    best_score = score
                    best_module = module

            # Usar URL del módulo solo si hay suficiente coincidencia;
            # si no, la URL de búsqueda oficial es más segura que un módulo incorrecto.
            if best_module and best_score >= score_threshold:
                raw_url = _normalize_url(best_module.get("url", ""))
                session["learn_url"] = raw_url or _search_url(topic)
            else:
                session["learn_url"] = _search_url(topic)

        return sessions

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

        self._persist_agent_response(
            student_id=student_id,
            session_id=trace_context.trace_id,
            agent_name='AssessmentAgent',
            phase='assessment_pending',
            result=ass_result,
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
                        structured_plan['sessions'] = (
                            OrchestratorAgent._enrich_sessions_with_learn_urls(
                                structured_plan.get('sessions', []),
                                chosen,
                            )
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
                    self._persist_agent_response(
                        student_id=student_id,
                        session_id=trace_context.trace_id,
                        agent_name='CertAdvisorAgent',
                        phase='certification',
                        result=cert_result if isinstance(cert_result, dict) else {
                            'response': cert_response,
                            'tokens': {},
                        },
                    )
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

            # HITL: await_replan_choice
            # Tras evaluación fallida, el usuario decide entre nuevo plan por brechas
            # o continuar el itinerario actual.
            if state.get('next_step') == 'await_replan_choice':
                decision = self._parse_replan_choice(message)
                if decision == 'continue_plan':
                    state['next_step'] = 'await_assessment_confirmation'
                    keep_response = (
                        "Perfecto. Continuamos con tu itinerario actual "
                        f"({state.get('chosen_certification', 'objetivo actual')}). "
                        "Cuando quieras una nueva evaluación escribe **'evaluar'**."
                    )
                    self._save_interaction_to_db(
                        student_id=student_id,
                        user_msg=message,
                        assistant_msg=keep_response,
                        phase=state.get('phase', 'assessment_complete'),
                        session_id=trace_context.trace_id,
                        agent_name='OrchestratorAgent',
                    )
                    return {
                        'response': keep_response,
                        'logs': [],
                        'phase': state.get('phase', 'assessment_complete'),
                        'trace_id': trace_context.trace_id,
                        'tokens': {},
                        'mode': 'azure_openai',
                    }

                if decision == 'new_plan':
                    state['next_step'] = None
                    preparation_context = self._build_reinforcement_context(
                        state=state,
                        user_message=message,
                    )
                    result = self._execute_sub_workflow_1(
                        preparation_context,
                        student_id,
                        trace_context,
                    )
                    response = result['response']
                    state['phase'] = result.get(
                        'phase', 'preparation_complete')
                    state['next_step'] = 'await_assessment_confirmation'
                    if 'curator_data' in result:
                        state['curator_data'] = result['curator_data']

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
                        'mode': 'azure_openai',
                    }

                ask_response = (
                    "Para continuar de forma clara, elige una opción:\n"
                    "1) **Nuevo plan basado en la última evaluación**\n"
                    "2) **Continuar con el plan actual**\n\n"
                    "Responde: *nuevo plan* o *continuar plan*."
                )
                self._save_interaction_to_db(
                    student_id=student_id,
                    user_msg=message,
                    assistant_msg=ask_response,
                    phase=state.get('phase', 'assessment_complete'),
                    session_id=trace_context.trace_id,
                    agent_name='OrchestratorAgent',
                )
                return {
                    'response': ask_response,
                    'logs': [],
                    'phase': state.get('phase', 'assessment_complete'),
                    'trace_id': trace_context.trace_id,
                    'tokens': {},
                    'mode': 'azure_openai',
                }

            # HITL: await_assessment_confirmation
            # El Sub-Workflow 1 acaba de terminar y pregunto al usuario si quiere evaluarse.
            # Interceptamos aqui para no caer en el routing de intents.
            if state.get('next_step') == 'await_assessment_confirmation':
                replan_choice = self._parse_replan_choice(message)
                if replan_choice == 'new_plan':
                    state['next_step'] = None
                    preparation_context = self._build_reinforcement_context(
                        state=state,
                        user_message=message,
                    )
                    result = self._execute_sub_workflow_1(
                        preparation_context,
                        student_id,
                        trace_context,
                    )
                    response = result['response']
                    state['phase'] = result.get(
                        'phase', 'preparation_complete')
                    state['next_step'] = 'await_assessment_confirmation'
                    if 'curator_data' in result:
                        state['curator_data'] = result['curator_data']

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
                        'mode': 'azure_openai',
                    }

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
                    state['phase'] = result.get('phase', 'assessment_complete')
                    if result.get('passed', False):
                        state['next_step'] = 'await_certification_choice'
                        state['last_assessment_feedback'] = ''
                    else:
                        state['next_step'] = 'await_replan_choice'
                        state['last_assessment_feedback'] = (
                            f"Assessment:\n{result.get('assessment_response', '')}\n\n"
                            f"Critic:\n{result.get('critic_response', '')}"
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
                if self._should_update_certification_goal(message, detected_topic, state):
                    explicit_code = self._extract_certification_code(message)
                    topic_code = self._extract_certification_code(
                        detected_topic)
                    chosen_topic = explicit_code or topic_code or detected_topic
                    state['chosen_certification'] = chosen_topic
                    if persistence_tool:
                        try:
                            persistence_tool.save_student_profile(student_id, {
                                'student_id': student_id,
                                'chosen_certification': state['chosen_certification'],
                                'phase': 'preparation',
                                'updated_at': datetime.now().isoformat()
                            })
                        except Exception as _pe:
                            logger.warning(
                                f"⚠️ Error guardando perfil de certificación: {_pe}")

                normalized_message = message.strip().lower()
                generic_preparation_messages = {
                    'preparar', 'preparame', 'prepárame', 'reforzar',
                    'refuerzo', 'continuar', 'seguir',
                }
                preparation_context = message
                if (
                    normalized_message in generic_preparation_messages
                    and state.get('chosen_certification')
                ):
                    preparation_context = self._build_reinforcement_context(
                        state=state,
                        user_message=message,
                    )

                # Sub-workflow completo (Curator → Study Plan → Engagement) sin pausa HITL
                result = self._execute_sub_workflow_1(
                    preparation_context, student_id, trace_context)
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
                        structured_plan['sessions'] = (
                            OrchestratorAgent._enrich_sessions_with_learn_urls(
                                structured_plan.get('sessions', []),
                                chosen_cert,
                            )
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
                        state['last_assessment_feedback'] = ''
                    else:
                        state['next_step'] = 'await_replan_choice'
                        state['last_assessment_feedback'] = (
                            f"Assessment:\n{result.get('assessment_response', '')}\n\n"
                            f"Critic:\n{result.get('critic_response', '')}"
                        )
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
                self._persist_agent_response(
                    student_id=student_id,
                    session_id=trace_context.trace_id,
                    agent_name='CertAdvisorAgent',
                    phase='certification',
                    result=cert_result if isinstance(cert_result, dict) else {
                        'response': cert_response,
                        'tokens': {},
                    },
                )
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

    def _persist_agent_response(
        self,
        student_id: str,
        session_id: str,
        agent_name: str,
        phase: str,
        result: dict,
    ) -> None:
        """Guarda en BD la salida de un agente especializado."""
        if not persistence_tool or not isinstance(result, dict):
            return

        try:
            persistence_tool.save_conversation_message(
                student_id=student_id,
                role='assistant',
                content=str(result.get('response', '')),
                session_id=session_id,
                agent_name=agent_name,
                phase=phase,
                tokens=result.get('tokens', {}),
            )
        except Exception as _pe:
            logger.warning(
                "⚠️ Error persistiendo respuesta de %s: %s",
                agent_name,
                _pe,
            )

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

        self._persist_agent_response(
            student_id=student_id,
            session_id=trace_context.trace_id,
            agent_name='CuratorAgent',
            phase='preparation',
            result=curator_result,
        )

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

        self._persist_agent_response(
            student_id=student_id,
            session_id=trace_context.trace_id,
            agent_name='StudyPlanAgent',
            phase='preparation',
            result=sp_result,
        )

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

        self._persist_agent_response(
            student_id=student_id,
            session_id=trace_context.trace_id,
            agent_name='EngagementAgent',
            phase='preparation_complete',
            result=eng_result,
        )

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

        self._persist_agent_response(
            student_id=student_id,
            session_id=trace_context.trace_id,
            agent_name='StudyPlanAgent',
            phase='preparation',
            result=sp_result,
        )

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

        self._persist_agent_response(
            student_id=student_id,
            session_id=trace_context.trace_id,
            agent_name='EngagementAgent',
            phase='preparation_complete',
            result=eng_result,
        )

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

        self._persist_agent_response(
            student_id=student_id,
            session_id=trace_context.trace_id,
            agent_name='AssessmentAgent',
            phase='assessment',
            result=ass_result,
        )

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

        self._persist_agent_response(
            student_id=student_id,
            session_id=trace_context.trace_id,
            agent_name='CriticAgent',
            phase='assessment_complete',
            result=cr_result,
        )

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
        cert_result = None
        cert_block = ""

        if passed:
            cert_context = (
                "El estudiante ha aprobado la evaluación técnica.\n"
                f"Certificación/ruta objetivo actual: "
                f"{self.conversation_state.get(student_id, {}).get('chosen_certification', 'Business Central')}\n\n"
                f"Resultado del Assessment Agent:\n{ass_result.get('response', '')}\n\n"
                f"Análisis del Critic Agent:\n{cr_result.get('response', '')}\n\n"
                "Recomienda la certificación Microsoft más adecuada y planifica el examen con acciones concretas."
            )
            self._emit_agent_active('cert_advisor', '🎓 Cert Advisor Agent')
            cert_result = self.agents['cert_advisor'].execute(
                cert_context,
                student_id,
            )
            cert_result = cert_result if isinstance(cert_result, dict) else {
                'response': cert_result,
                'agent_name': 'Cert Advisor Agent',
                'tools_used': [],
            }
            self._persist_agent_response(
                student_id=student_id,
                session_id=trace_context.trace_id,
                agent_name='CertAdvisorAgent',
                phase='certification',
                result=cert_result,
            )
            cert_block = self._fmt_agent_block(cert_result, 3, 3)

        if passed:
            response = (
                f"\U0001f9e0 **Sub-Workflow #2: Evaluación completada** ({'3/3' if cert_result else '2/2'} agentes, {total_time:.1f}s)\n"
                f"Veredicto del Orchestrator: **APROBADO ✅**\n\n"
                + ass_block + "\n\n"
                + cr_block + "\n\n"
                + cert_block +
                f"\n\n---\n"
                "🎉 ¡Felicitaciones! Ya tienes recomendación de certificación y planificación de examen. "
                "Si quieres, te preparo el plan detallado para esa fecha."
            )
        else:
            response = (
                f"\U0001f9e0 **Sub-Workflow #2: Evaluación completada** (2/2 agentes, {total_time:.1f}s)\n"
                f"Veredicto del Orchestrator: **NECESITA REFUERZO ⚠️**\n\n"
                + ass_block + "\n\n"
                + cr_block +
                f"\n\n---\n"
                "📌 Elige cómo continuar:\n"
                "1) **Nuevo plan basado en la última evaluación**\n"
                "2) **Continuar con el plan actual**\n\n"
                "Responde: **nuevo plan** o **continuar plan**."
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

        return {
            'response': response,
            'passed': passed,
            'phase': 'certification' if passed else 'assessment_complete',
            'tokens': self._sum_tokens(ass_result, cr_result, cert_result),
            'assessment_response': ass_result.get('response', ''),
            'critic_response': cr_result.get('response', ''),
            'cert_advisor_response': cert_result.get('response', '') if isinstance(cert_result, dict) else '',
        }

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
        """Evalúa si el estudiante pasó la evaluación.

        Regla principal: aprobado cuando hay al menos 3 respuestas correctas
        sobre 5 preguntas (>= 60%).
        """
        import os as _os
        import json as _json

        assessment_text = assessment_result.get('response', '') if isinstance(
            assessment_result, dict) else str(assessment_result)
        critic_text = critic_result.get('response', '') if isinstance(
            critic_result, dict) else str(critic_result)

        correct_matches = re.findall(
            r'evaluaci[oó]n\s*:\s*✅|tu\s+respuesta\s*:\s*[a-d]\)\s*\n?\s*evaluaci[oó]n\s*:\s*✅',
            assessment_text,
            flags=re.IGNORECASE,
        )
        incorrect_matches = re.findall(
            r'evaluaci[oó]n\s*:\s*❌|tu\s+respuesta\s*:\s*[a-d]\)\s*\n?\s*evaluaci[oó]n\s*:\s*❌',
            assessment_text,
            flags=re.IGNORECASE,
        )
        correct_count = len(correct_matches)
        incorrect_count = len(incorrect_matches)
        answered_count = correct_count + incorrect_count

        if answered_count > 0:
            threshold = max(3, int((answered_count * 0.6) + 0.9999))
            return correct_count >= threshold

        ready_score_match = re.search(
            r'(?:score\s+estimado\s+de\s+ready-for-exam|ready-for-exam|listo\s+para\s+examen)\s*[:=]?\s*(\d{1,3})(?:\s*/\s*100|\s*%)?',
            assessment_text,
            re.IGNORECASE,
        )
        if ready_score_match:
            try:
                return float(ready_score_match.group(1)) >= 60.0
            except (TypeError, ValueError):
                pass

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
