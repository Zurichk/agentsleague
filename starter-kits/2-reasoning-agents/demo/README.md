# AEP CertMaster — Reasoning Agents Submission

Proyecto para Agents League · Track Reasoning Agents.

AEP CertMaster implementa un sistema multi-agente para preparar certificaciones Microsoft con un flujo completo:

1. Gestión de rutas de aprendizaje.
2. Plan de estudio estructurado con calendario.
3. Engagement y recordatorios por email.
4. Evaluación adaptativa.
5. Análisis crítico de brechas.
6. Recomendación de certificación y planificación de examen.

---

## 1) Resumen del problema y solución

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


## 4) Demo

- Video de demostración: [demo-agents.mkv](demo-agents.mkv)

---

## 5) Ejecución local

Desde `starter-kits/2-reasoning-agents`:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run_flask_dashboard.py
```

La app se levanta en el puerto configurado del dashboard.

### Otras opciones
- **Docker**: hay un `Dockerfile` en la raíz. Puedes construir y ejecutar el contenedor con:

  ```bash
docker build -t aep-certmaster .
docker run -p 5000:5000 --env-file .env aep-certmaster
```

- **Deployment público**: si sólo quieres verla en acción, está desplegada en Coolify aquí:

  http://78.47.111.58:5033/

  (el servicio puede tardar unos segundos en arrancar si está inactivo)

---

## 6) Variables de entorno mínimas

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

## 7) Seguridad y buenas prácticas

- No subir llaves/API keys ni `.env`.
- Usar datos de demo.
- Revisar git history antes de publicar.
- Verificar que logs no contengan secretos.
