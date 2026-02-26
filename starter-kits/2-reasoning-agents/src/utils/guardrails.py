"""
Guardrails y validaciones de entrada para AEP CertMaster.

Incluye validacion basica, sanitizacion y filtros para entradas
potencialmente maliciosas o fuera de rango.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


AEP_MAX_INPUT_LENGTH = 2000
AEP_MIN_INPUT_LENGTH = 1
AEP_MAX_WEEKLY_HOURS = 80
AEP_MIN_WEEKLY_HOURS = 1

AEP_ALLOWED_LEVELS = {"beginner", "intermediate", "advanced"}

AEP_BLOCKED_PATTERNS = [
    "ignore previous instructions",
    "system prompt",
    "developer message",
    "jailbreak",
    "bypass safety",
]


def sanitize_text(text: str) -> str:
    """
    Sanitiza un texto eliminando espacios extras y caracteres de control.

    Args:
        text: Texto original.

    Returns:
        Texto sanitizado.
    """
    cleaned = " ".join(text.split())
    return cleaned.strip()


def contains_blocked_pattern(text: str) -> bool:
    """
    Verifica si el texto contiene patrones bloqueados.

    Args:
        text: Texto a revisar.

    Returns:
        True si encuentra patrones bloqueados.
    """
    lowered = text.lower()
    return any(pattern in lowered for pattern in AEP_BLOCKED_PATTERNS)


def validate_user_message(message: str) -> Tuple[bool, str, str]:
    """
    Valida un mensaje de usuario para el chat.

    Args:
        message: Mensaje original.

    Returns:
        Tupla (es_valido, error, mensaje_sanitizado).
    """
    if message is None:
        return False, "Mensaje requerido.", ""

    cleaned = sanitize_text(message)
    if len(cleaned) < AEP_MIN_INPUT_LENGTH:
        return False, "Mensaje vacio.", ""

    if len(cleaned) > AEP_MAX_INPUT_LENGTH:
        return False, "Mensaje demasiado largo.", cleaned[:AEP_MAX_INPUT_LENGTH]

    if contains_blocked_pattern(cleaned):
        return False, "Mensaje no permitido por politicas de seguridad.", ""

    return True, "", cleaned


def validate_student_payload(
    payload: Dict[str, Any]
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Valida el payload de inicio de certificacion.

    Args:
        payload: Diccionario con datos del estudiante.

    Returns:
        Tupla (es_valido, errores, payload_sanitizado).
    """
    errors: List[str] = []
    sanitized: Dict[str, Any] = {}

    required_fields = [
        "student_name",
        "current_level",
        "weekly_hours",
        "target_certification",
    ]

    for field in required_fields:
        if field not in payload or not payload[field]:
            errors.append(f"Campo requerido faltante: {field}")

    if errors:
        return False, errors, sanitized

    student_name = sanitize_text(str(payload.get("student_name", "")))
    if not student_name:
        errors.append("Nombre del estudiante invalido.")

    current_level = sanitize_text(
        str(payload.get("current_level", ""))
    ).lower()
    if current_level not in AEP_ALLOWED_LEVELS:
        errors.append("Nivel de conocimientos invalido.")

    try:
        weekly_hours = int(payload.get("weekly_hours"))
        if (
            weekly_hours < AEP_MIN_WEEKLY_HOURS
            or weekly_hours > AEP_MAX_WEEKLY_HOURS
        ):
            errors.append("Horas semanales fuera de rango.")
    except (TypeError, ValueError):
        errors.append("Horas semanales invalidas.")
        weekly_hours = 0

    target_certification = sanitize_text(
        str(payload.get("target_certification", ""))
    )
    if not target_certification:
        errors.append("Certificacion objetivo invalida.")

    experience = sanitize_text(str(payload.get("experience", "")))
    goals = sanitize_text(str(payload.get("goals", "")))

    if errors:
        return False, errors, sanitized

    sanitized = {
        "student_name": student_name,
        "current_level": current_level,
        "weekly_hours": weekly_hours,
        "experience": experience,
        "goals": goals,
        "target_certification": target_certification,
    }

    return True, [], sanitized
