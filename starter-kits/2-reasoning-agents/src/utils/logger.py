"""
Utilidades de logging para AEP CertMaster.

Configura logging estructurado con salida a consola y archivo,
usando niveles apropiados y formato consistente con trazabilidad completa.
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from src.config.settings import AEP_PROJECT_ROOT, load_settings


AEP_LOG_FORMAT = (
    "%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s"
)
AEP_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Formato JSON estructurado para análisis
AEP_JSON_FORMAT = {
    "timestamp": "%(asctime)s",
    "logger": "%(name)s",
    "level": "%(levelname)s",
    "message": "%(message)s",
    "trace_id": "%(trace_id)s",
    "user_id": "%(user_id)s",
    "agent": "%(agent)s",
    "phase": "%(phase)s",
    "session_id": "%(session_id)s",
    "correlation_id": "%(correlation_id)s",
    "performance": "%(performance)s"
}


class StructuredLogRecord(logging.LogRecord):
    """LogRecord personalizado con campos adicionales para trazabilidad."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Campos adicionales para trazabilidad
        self.trace_id = getattr(self, 'trace_id', '')
        self.user_id = getattr(self, 'user_id', '')
        self.agent = getattr(self, 'agent', '')
        self.phase = getattr(self, 'phase', '')
        self.session_id = getattr(self, 'session_id', '')
        self.correlation_id = getattr(self, 'correlation_id', '')
        self.performance = getattr(self, 'performance', {})


class StructuredFormatter(logging.Formatter):
    """Formatter que incluye campos adicionales para trazabilidad."""

    def __init__(self, fmt=None, datefmt=None, style='%', use_json=False):
        super().__init__(fmt, datefmt, style)
        self.use_json = use_json

    def format(self, record: StructuredLogRecord) -> str:
        if self.use_json:
            log_entry = {
                "timestamp": self.formatTime(record, self.datefmt),
                "logger": record.name,
                "level": record.levelname,
                "message": record.getMessage(),
                "trace_id": getattr(record, 'trace_id', ''),
                "user_id": getattr(record, 'user_id', ''),
                "agent": getattr(record, 'agent', ''),
                "phase": getattr(record, 'phase', ''),
                "session_id": getattr(record, 'session_id', ''),
                "correlation_id": getattr(record, 'correlation_id', ''),
                "performance": getattr(record, 'performance', {}),
                "extra": getattr(record, 'extra', {})
            }
            return json.dumps(log_entry, ensure_ascii=False)
        else:
            # Formato de texto enriquecido
            extra_info = []
            if hasattr(record, 'trace_id') and record.trace_id:
                extra_info.append(f"TRACE:{record.trace_id[:8]}")
            if hasattr(record, 'agent') and record.agent:
                extra_info.append(f"AGENT:{record.agent}")
            if hasattr(record, 'phase') and record.phase:
                extra_info.append(f"PHASE:{record.phase}")
            if hasattr(record, 'user_id') and record.user_id:
                extra_info.append(f"USER:{record.user_id}")

            extra_str = f" [{' | '.join(extra_info)}]" if extra_info else ""
            return f"{super().format(record)}{extra_str}"


class TraceContext:
    """Contexto de trazabilidad para una sesión de usuario."""

    def __init__(
        self,
        user_id: str = "",
        session_id: str = "",
        correlation_id: str = "",
        operation: str = "",
    ):
        self.trace_id = str(uuid.uuid4())
        self.user_id = user_id
        self.session_id = session_id or str(uuid.uuid4())
        self.correlation_id = correlation_id or self.trace_id
        self.operation = operation
        self.start_time = datetime.now()
        self.steps = []

    def add_step(self, agent: str, phase: str, action: str, performance: Dict[str, Any] = None):
        """Agrega un paso al contexto de trazabilidad."""
        step = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent,
            "phase": phase,
            "action": action,
            "performance": performance or {}
        }
        self.steps.append(step)

    def get_context_dict(self) -> Dict[str, Any]:
        """Retorna el contexto completo como diccionario."""
        return {
            "trace_id": self.trace_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "correlation_id": self.correlation_id,
            "operation": self.operation,
            "start_time": self.start_time.isoformat(),
            "duration": (datetime.now() - self.start_time).total_seconds(),
            "steps": self.steps
        }


# Contexto global de trazabilidad
_current_trace_context: Optional[TraceContext] = None


def get_current_trace_context() -> Optional[TraceContext]:
    """Obtiene el contexto de trazabilidad actual."""
    return _current_trace_context


def set_trace_context(context: TraceContext):
    """Establece el contexto de trazabilidad actual."""
    global _current_trace_context
    _current_trace_context = context


def create_trace_context(
    user_id: str = "",
    session_id: str = "",
    operation: str = "",
) -> TraceContext:
    """Crea un nuevo contexto de trazabilidad."""
    context = TraceContext(
        user_id=user_id,
        session_id=session_id,
        operation=operation,
    )
    set_trace_context(context)
    return context


def _trace_log_enabled() -> bool:
    """
    Indica si el logging local de trazas está habilitado.

    Lee ``AEP_TRACE_LOG_ENABLED`` del entorno en cada llamada para
    respetar el valor del .env sin necesidad de reiniciar el módulo.

    Returns:
        True si se deben escribir ficheros y emitir INFO de traza.
    """
    try:
        return load_settings().aep_trace_log_enabled
    except Exception:
        return False


def log_with_trace(
    logger: logging.Logger,
    level: int,
    message: str,
    agent: str = "",
    phase: str = "",
    performance: Dict[str, Any] | None = None,
    extra: Dict[str, Any] | None = None,
    trace_context: TraceContext | None = None,
) -> None:
    """Log con información de trazabilidad."""
    # Cuando las trazas locales están desactivadas, los niveles INFO
    # se degradan a DEBUG para no saturar la consola.  ERROR y WARNING
    # siempre se emiten.
    if level == logging.INFO and not _trace_log_enabled():
        level = logging.DEBUG

    if trace_context is not None:
        set_trace_context(trace_context)

    context = get_current_trace_context()

    # Preparar campos adicionales
    extra_fields = {
        'trace_id': context.trace_id if context else '',
        'user_id': context.user_id if context else '',
        'session_id': context.session_id if context else '',
        'correlation_id': context.correlation_id if context else '',
        'agent': agent,
        'phase': phase,
        'performance': performance or {},
        'extra': extra or {}
    }

    # Agregar paso al contexto si hay información relevante
    if context and (agent or phase):
        context.add_step(agent, phase, message, performance)

    logger.log(level, message, extra=extra_fields)


def setup_logger(
    name: str,
    log_file: str | None = None,
    use_json: bool = False,
) -> logging.Logger:
    """
    Configura y retorna un logger para un módulo dado con soporte para logging estructurado.

    Args:
        name: Nombre del logger (generalmente __name__).
        log_file: Ruta opcional para archivo de log.
        use_json: Si usar formato JSON estructurado.

    Returns:
        Logger configurado con handlers de consola y
        opcionalmente archivo con trazabilidad.
    """
    settings = load_settings()
    level = getattr(
        logging, settings.aep_log_level.upper(), logging.INFO
    )

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Evitar duplicar handlers
    if logger.handlers:
        return logger

    # Formatter estructurado
    formatter = StructuredFormatter(
        fmt=AEP_LOG_FORMAT if not use_json else None,
        datefmt=AEP_LOG_DATE_FORMAT,
        use_json=use_json
    )

    # Handler de consola
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Handler de archivo (opcional)
    if log_file:
        log_path = AEP_PROJECT_ROOT / "logs"
        log_path.mkdir(exist_ok=True)

        # Handler para logs de texto
        text_file_handler = logging.FileHandler(
            log_path / log_file, encoding="utf-8"
        )
        text_file_handler.setLevel(level)
        text_formatter = StructuredFormatter(
            fmt=AEP_LOG_FORMAT, datefmt=AEP_LOG_DATE_FORMAT, use_json=False
        )
        text_file_handler.setFormatter(text_formatter)
        logger.addHandler(text_file_handler)

        # Handler para logs JSON estructurados
        json_log_file = log_path / f"{Path(log_file).stem}.jsonl"
        json_file_handler = logging.FileHandler(
            json_log_file, encoding="utf-8"
        )
        json_file_handler.setLevel(level)
        json_formatter = StructuredFormatter(use_json=True)
        json_file_handler.setFormatter(json_formatter)
        logger.addHandler(json_file_handler)

    return logger


def get_agent_logger(agent_name: str, use_json: bool = True) -> logging.Logger:
    """
    Retorna un logger específico para un agente con trazabilidad completa.

    Args:
        agent_name: Nombre del agente.
        use_json: Si usar formato JSON para análisis.

    Returns:
        Logger con prefijo del nombre del agente y trazabilidad.
    """
    log_file = f"agents_{agent_name}.log" if _trace_log_enabled() else None
    return setup_logger(
        f"aep.agent.{agent_name}",
        log_file=log_file,
        use_json=use_json
    )


def get_orchestrator_logger(use_json: bool = True) -> logging.Logger:
    """
    Retorna el logger del orquestador principal con trazabilidad completa.

    Args:
        use_json: Si usar formato JSON para análisis.

    Returns:
        Logger configurado para el orquestador con trazabilidad.
    """
    log_file = "orchestrator.log" if _trace_log_enabled() else None
    return setup_logger(
        "aep.orchestrator",
        log_file=log_file,
        use_json=use_json
    )


def get_workflow_logger(workflow_name: str, use_json: bool = True) -> logging.Logger:
    """
    Retorna un logger para un workflow específico.

    Args:
        workflow_name: Nombre del workflow.
        use_json: Si usar formato JSON para análisis.

    Returns:
        Logger configurado para el workflow.
    """
    log_file = (
        f"workflow_{workflow_name}.log" if _trace_log_enabled() else None
    )
    return setup_logger(
        f"aep.workflow.{workflow_name}",
        log_file=log_file,
        use_json=use_json
    )


def log_agent_action(
    logger: logging.Logger,
    agent: str,
    action: str,
    message: str,
    extra: Dict[str, Any] | None = None,
    trace_context: TraceContext | None = None,
    phase: str = "",
    performance: Dict[str, Any] | None = None,
) -> None:
    """
    Log estandarizado para acciones de agentes con trazabilidad completa.

    Args:
        logger: Logger del agente.
        agent: Nombre del agente.
        action: Acción realizada.
        message: Mensaje o detalle principal.
        extra: Contexto adicional.
        trace_context: Contexto de trazabilidad opcional.
        phase: Fase del proceso.
        performance: Métricas de rendimiento.
    """
    log_with_trace(
        logger,
        logging.INFO,
        f"{action}: {message}",
        agent=agent,
        phase=phase,
        performance=performance,
        extra=extra,
        trace_context=trace_context,
    )


def log_workflow_transition(
    logger: logging.Logger,
    *args: Any,
    **kwargs: Any,
) -> None:
    """
    Log estandarizado para transiciones de workflow.

    Soporta dos firmas:
    1) (logger, from_phase, to_phase, trigger, context)
    2) (logger, trace_context, transition_type, message, extra)
    """
    if args and isinstance(args[0], TraceContext):
        trace_context = args[0]
        transition_type = args[1] if len(args) > 1 else "transition"
        message = args[2] if len(args) > 2 else ""
        extra = args[3] if len(args) > 3 else None
        log_with_trace(
            logger,
            logging.INFO,
            f"{transition_type}: {message}",
            extra=extra or {},
            trace_context=trace_context,
        )
        return

    from_phase = args[0] if len(args) > 0 else ""
    to_phase = args[1] if len(args) > 1 else ""
    trigger = args[2] if len(args) > 2 else ""
    context = args[3] if len(args) > 3 else None
    message = f"Workflow transition: {from_phase} → {to_phase} (trigger: {trigger})"
    log_with_trace(
        logger,
        logging.INFO,
        message,
        phase=to_phase,
        extra=context or {},
    )


def log_performance_metrics(
    logger: logging.Logger,
    *args: Any,
    **kwargs: Any,
) -> None:
    """
    Log estandarizado para métricas de rendimiento.

    Soporta dos firmas:
    1) (logger, operation, metrics, agent)
    2) (logger, trace_context, operation, duration, extra)
    """
    if args and isinstance(args[0], TraceContext):
        trace_context = args[0]
        operation = args[1] if len(args) > 1 else "operation"
        duration = args[2] if len(args) > 2 else 0.0
        extra = args[3] if len(args) > 3 else None
        metrics = {"duration_sec": duration, **(extra or {})}
        log_with_trace(
            logger,
            logging.DEBUG,
            f"Performance metrics for {operation}: {metrics}",
            performance=metrics,
            trace_context=trace_context,
        )
        return

    operation = args[0] if len(args) > 0 else "operation"
    metrics = args[1] if len(args) > 1 else {}
    agent = args[2] if len(args) > 2 else ""
    log_with_trace(
        logger,
        logging.DEBUG,
        f"Performance metrics for {operation}: {metrics}",
        agent=agent,
        performance=metrics,
    )


def log_agent_response(
    logger: logging.Logger,
    agent: str,
    action: str,
    message: str,
    extra: Dict[str, Any] | None = None,
    trace_context: TraceContext | None = None,
    phase: str = "",
) -> None:
    """
    Log estandarizado para respuestas de agentes.

    Args:
        logger: Logger del agente.
        agent: Nombre del agente.
        action: Acción asociada.
        message: Mensaje de respuesta.
        extra: Contexto adicional.
        trace_context: Contexto de trazabilidad opcional.
        phase: Fase del proceso.
    """
    log_with_trace(
        logger,
        logging.INFO,
        f"{action}: {message}",
        agent=agent,
        phase=phase,
        extra=extra,
        trace_context=trace_context,
    )


def log_error(
    logger: logging.Logger,
    component: str,
    operation: str,
    message: str,
    extra: Dict[str, Any] | None = None,
    trace_context: TraceContext | None = None,
    phase: str = "",
) -> None:
    """
    Log estandarizado para errores.

    Args:
        logger: Logger del componente.
        component: Nombre del componente.
        operation: Operacion donde ocurre el error.
        message: Mensaje de error.
        extra: Contexto adicional.
        trace_context: Contexto de trazabilidad opcional.
        phase: Fase del proceso.
    """
    log_with_trace(
        logger,
        logging.ERROR,
        f"{operation}: {message}",
        agent=component,
        phase=phase,
        extra=extra,
        trace_context=trace_context,
    )


def save_trace_context(context: TraceContext, log_path: Path = None):
    """
    Guarda el contexto de trazabilidad completo en un archivo JSON.

    Si ``AEP_TRACE_LOG_ENABLED=false`` (por defecto) la función es un
    no-op: las trazas ya se envían a Application Insights y no es
    necesario duplicarlas en disco.

    Args:
        context: Contexto de trazabilidad a guardar.
        log_path: Ruta donde guardar (opcional, usa default si None).
    """
    if not _trace_log_enabled():
        return

    if not log_path:
        log_path = AEP_PROJECT_ROOT / "logs" / "traces"
        log_path.mkdir(exist_ok=True)

    trace_file = log_path / f"trace_{context.trace_id}.json"

    with open(trace_file, 'w', encoding='utf-8') as f:
        json.dump(context.get_context_dict(), f, indent=2, ensure_ascii=False)

    logger = logging.getLogger("aep.tracer")
    logger.debug(f"Trace context saved: {trace_file}")
