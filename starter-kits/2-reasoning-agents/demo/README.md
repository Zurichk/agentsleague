# AEP CertMaster — Reasoning Agents Submission

Proyecto para Agents League · Track Reasoning Agents.

AEP CertMaster implementa un sistema multi-agente para preparar certificaciones Microsoft con un flujo completo:

1. Curación de rutas de aprendizaje.
2. Plan de estudio estructurado con calendario.
3. Engagement y recordatorios por email.
4. Evaluación adaptativa.
5. Análisis crítico de brechas.
6. Recomendación de certificación y planificación de examen.

---

## 1) Resumen del problema y solución

### Problema
Un estudiante suele tener fricción en 3 puntos: no saber por dónde empezar, no sostener constancia y no saber si ya está listo para examen.

### Solución
AEP CertMaster coordina agentes especializados mediante un Orchestrator que aplica razonamiento multi-paso y decisiones HITL (human-in-the-loop) para guiar al estudiante desde intención libre hasta decisión de examen.

---

## 2) Arquitectura funcional

### Flujo principal
1. Input libre del estudiante (ej. “Quiero aprender Business Central”).
2. Sub-Workflow 1 (secuencial): Curator → Study Plan → Engagement.
3. Confirmación humana para iniciar evaluación.
4. Sub-Workflow 2: Assessment → Critic → decisión.
5. Si aprueba: Cert Advisor recomienda certificación y planifica examen.
6. Si no aprueba: vuelve a preparación con refuerzo.

### Patrones de razonamiento implementados
- Planner-Executor: Orchestrator detecta intención y enruta.
- Critic/Verifier: Critic evalúa calidad y gaps.
- HITL: confirmaciones antes de evaluaciones y decisiones de continuidad.
- Role-based specialization: responsabilidades claras por agente.

---

## 3) Agentes y tools

| Agente | Rol | Tools declaradas |
|---|---|---|
| Orchestrator | Coordina fases, estado e intención | intent detection, state machine, HITL |
| Curator | Recomienda rutas Microsoft Learn | `microsoft_learn_search`, `certification_catalog`, `relevance_scorer` |
| Study Plan | Convierte rutas en plan accionable | `calendar_generator`, `workload_calculator`, `milestone_planner`, `bloom_taxonomy_mapper` |
| Engagement | Constancia y recordatorios | `email_reminder_service`, `motivation_personalizer` |
| Assessment | Evaluación adaptativa | `question_generator`, `knowledge_mapper`, `adaptive_engine`, `scoring_rubric` |
| Critic | Análisis de brechas y calidad | `quality_scorer`, `gap_analyzer`, `benchmark_comparator`, `improvement_suggester` |
| Cert Advisor | Recomendación de certificación y examen | `certification_roadmap`, `career_path_analyzer`, `exam_scheduler`, `roi_calculator` |

### Integraciones del proyecto
- Azure OpenAI (modo principal de ejecución).
- Persistencia SQLite para historial, planes, evaluaciones y métricas.
- Calendario `.ics` exportable.
- Envío de email (real por SMTP o simulado según configuración).

---

## 4) Qué puedes ver en Insights

En la sección de Insights / Progress se muestran:

- Progreso global y evolución de estudio.
- Historial de evaluaciones con puntuaciones.
- Áreas fuertes y débiles detectadas.
- Recomendaciones accionables del Critic.
- Predicción orientativa de preparación para examen.

Además, el sistema conserva trazabilidad por sesión para explicar decisiones y facilitar debugging.

---

## 5) Mapeo directo a criterios de evaluación

### Precisión y relevancia (25%)
- Flujo alineado al escenario oficial (preparar → evaluar → decidir certificación).
- Resultados accionables: ruta, plan, evaluación, feedback y siguiente paso.

### Razonamiento y pensamiento de múltiples pasos (25%)
- Orquestación secuencial con estado conversacional.
- Sub-workflows especializados + decisión condicional por resultado.

### Creatividad y originalidad (15%)
- Combinación de agentes pedagógicos + agente crítico + asesor de certificación.
- Integración de plan de estudio con calendario y recordatorios.

### Experiencia de usuario y presentación (15%)
- Chat con ejecución visible por etapas.
- Flujo comprensible, demostrable y centrado en estudiante.

### Confiabilidad y seguridad (20%)
- Persistencia robusta de historial/evaluaciones.
- Validación de entradas y fallback de integración.
- Recomendación explícita de no exponer secretos.

---

## 6) Demo

- Video de demostración: [demo/demo-agents.mkv](demo/demo-agents.mkv)
- Si el archivo aún no está subido, revisa [demo/README.md](demo/README.md).

---

## 7) Ejecución local

Desde `starter-kits/2-reasoning-agents`:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run_flask_dashboard.py
```

La app inicia en el puerto configurado del dashboard.

---

## 8) Variables de entorno mínimas

```env
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o-agentsleague

# Email real opcional
USE_REAL_EMAIL=false
SMTP_HOST=
SMTP_PORT=587
SMTP_USER=
SMTP_PASSWORD=
SMTP_FROM_NAME=AEP CertMaster
```

---

## 9) Entrega oficial (submission)

1. Repositorio público en GitHub.
2. README claro (este archivo).
3. Video demo enlazado.
4. Crear issue en `microsoft/agentsleague` con plantilla de project submission.
5. Seleccionar track Reasoning Agents.
6. Incluir enlace del repositorio y enlace de demo.
7. Confirmar registro en `aka.ms/agentsleague/register`.

Fecha límite indicada: 1 de marzo de 2026, 11:59 PM PT.

---

## 10) Seguridad y buenas prácticas

- No subir llaves/API keys ni `.env`.
- Usar datos de demo.
- Revisar git history antes de publicar.
- Verificar que logs no contengan secretos.

---

## 11) Documentación complementaria

- Funcionamiento técnico de agentes: [docs/funcionamiento agentes.md](docs/funcionamiento%20agentes.md)
- Guion de presentación: [docs/guion video.md](docs/guion%20video.md)
- Notas operativas locales: [docs/info.txt](docs/info.txt)



# Demo de presentación

Coloca aquí el video de demostración final con este nombre:

- `demo-agents.mkv`

Ruta esperada en el repositorio:

- `starter-kits/2-reasoning-agents/demo/demo-agents.mkv`
