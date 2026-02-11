"""Control de acceso RBAC (Role-Based Access Control) — ISO 27001 A.9.2.

Implementa políticas granulares de acceso a datos de series temporales
y operaciones ML.  Cada operación se valida contra el rol del usuario
antes de ejecutarse.

Roles:
- ``viewer``: Solo lectura de predicciones y métricas.
- ``operator``: Lectura + ejecución de predicciones.
- ``admin``: Todo + cambios de configuración.
- ``auditor``: Lectura de audit logs + métricas.

Políticas por recurso:
- ``series_data``: Acceso a datos de series temporales.
- ``predictions``: Acceso a predicciones.
- ``config``: Cambios de configuración.
- ``audit_logs``: Lectura de logs de auditoría.
- ``models``: Gestión de modelos ML.

ISO 27001 A.9.2.1: Registro y des-registro de usuarios.
ISO 27001 A.9.2.2: Provisión de acceso de usuario.
ISO 27001 A.9.4.1: Restricción de acceso a información.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, FrozenSet, List, Optional, Set

logger = logging.getLogger(__name__)


class Role(Enum):
    """Roles del sistema."""

    VIEWER = "viewer"
    OPERATOR = "operator"
    ADMIN = "admin"
    AUDITOR = "auditor"


class Permission(Enum):
    """Permisos granulares."""

    READ_SERIES_DATA = "read_series_data"
    READ_SENSOR_DATA = "read_series_data"  # Alias legacy
    READ_PREDICTIONS = "read_predictions"
    EXECUTE_PREDICTION = "execute_prediction"
    READ_ANOMALIES = "read_anomalies"
    EXECUTE_ANOMALY_DETECTION = "execute_anomaly_detection"
    READ_CONFIG = "read_config"
    WRITE_CONFIG = "write_config"
    READ_AUDIT_LOGS = "read_audit_logs"
    MANAGE_MODELS = "manage_models"
    READ_METRICS = "read_metrics"


# Mapeo de roles a permisos
_ROLE_PERMISSIONS: Dict[Role, FrozenSet[Permission]] = {
    Role.VIEWER: frozenset({
        Permission.READ_PREDICTIONS,
        Permission.READ_ANOMALIES,
        Permission.READ_METRICS,
    }),
    Role.OPERATOR: frozenset({
        Permission.READ_SENSOR_DATA,
        Permission.READ_PREDICTIONS,
        Permission.EXECUTE_PREDICTION,
        Permission.READ_ANOMALIES,
        Permission.EXECUTE_ANOMALY_DETECTION,
        Permission.READ_CONFIG,
        Permission.READ_METRICS,
    }),
    Role.ADMIN: frozenset({
        Permission.READ_SENSOR_DATA,
        Permission.READ_PREDICTIONS,
        Permission.EXECUTE_PREDICTION,
        Permission.READ_ANOMALIES,
        Permission.EXECUTE_ANOMALY_DETECTION,
        Permission.READ_CONFIG,
        Permission.WRITE_CONFIG,
        Permission.READ_AUDIT_LOGS,
        Permission.MANAGE_MODELS,
        Permission.READ_METRICS,
    }),
    Role.AUDITOR: frozenset({
        Permission.READ_PREDICTIONS,
        Permission.READ_ANOMALIES,
        Permission.READ_AUDIT_LOGS,
        Permission.READ_METRICS,
        Permission.READ_CONFIG,
    }),
}


@dataclass
class UserContext:
    """Contexto de usuario para validación de acceso.

    Attributes:
        user_id: Identificador único del usuario/proceso.
        roles: Roles asignados al usuario.
        allowed_sensor_ids: Sensores a los que tiene acceso.
            Si está vacío, tiene acceso a todos (para admin).
        allowed_device_ids: Dispositivos a los que tiene acceso.
    """

    user_id: str
    roles: Set[Role] = field(default_factory=lambda: {Role.VIEWER})
    allowed_series_ids: Set[str] = field(default_factory=set)
    allowed_sensor_ids: Set[int] = field(default_factory=set)  # Legacy IoT
    allowed_device_ids: Set[int] = field(default_factory=set)  # Legacy IoT

    @property
    def permissions(self) -> FrozenSet[Permission]:
        """Permisos efectivos (unión de todos los roles)."""
        perms: Set[Permission] = set()
        for role in self.roles:
            perms.update(_ROLE_PERMISSIONS.get(role, frozenset()))
        return frozenset(perms)

    def has_permission(self, permission: Permission) -> bool:
        """Verifica si el usuario tiene un permiso específico."""
        return permission in self.permissions

    def can_access_series(self, series_id: str) -> bool:
        """Verifica si el usuario puede acceder a una serie.

        Admin sin restricciones puede acceder a todas.
        Combina ``allowed_series_ids`` (agnóstico) y ``allowed_sensor_ids``
        (legacy IoT, convertido a str).
        """
        # Combinar ambos sets
        combined = self.allowed_series_ids | {str(s) for s in self.allowed_sensor_ids}
        if Role.ADMIN in self.roles and not combined:
            return True
        if not combined:
            return True
        return series_id in combined

    def can_access_sensor(self, sensor_id: int) -> bool:
        """Legacy: verifica acceso a sensor por ID numérico."""
        return self.can_access_series(str(sensor_id))


class AccessDeniedError(Exception):
    """Error de acceso denegado (ISO 27001 A.9.4.1)."""

    def __init__(self, user_id: str, permission: str, resource: str) -> None:
        self.user_id = user_id
        self.permission = permission
        self.resource = resource
        super().__init__(
            f"Acceso denegado: usuario '{user_id}' no tiene permiso "
            f"'{permission}' sobre recurso '{resource}'"
        )


class AccessControlService:
    """Servicio de control de acceso RBAC.

    Valida permisos antes de ejecutar operaciones.
    Loggea intentos de acceso (exitosos y denegados) para auditoría.

    Attributes:
        _users: Registro de usuarios y sus contextos.
        _audit_port: Port de auditoría para loggear accesos.
    """

    def __init__(self, audit_port: Optional[object] = None) -> None:
        self._users: Dict[str, UserContext] = {}
        self._audit_port = audit_port

    def register_user(self, context: UserContext) -> None:
        """Registra un usuario en el sistema (ISO 27001 A.9.2.1).

        Args:
            context: Contexto del usuario con roles y restricciones.
        """
        self._users[context.user_id] = context
        logger.info(
            "user_registered",
            extra={
                "user_id": context.user_id,
                "roles": [r.value for r in context.roles],
            },
        )

    def deregister_user(self, user_id: str) -> None:
        """Des-registra un usuario (ISO 27001 A.9.2.1)."""
        self._users.pop(user_id, None)
        logger.info("user_deregistered", extra={"user_id": user_id})

    def get_user(self, user_id: str) -> Optional[UserContext]:
        """Obtiene contexto de usuario."""
        return self._users.get(user_id)

    def check_permission(
        self,
        user_id: str,
        permission: Permission,
        resource: str = "",
    ) -> None:
        """Valida que el usuario tenga el permiso requerido.

        Args:
            user_id: ID del usuario.
            permission: Permiso requerido.
            resource: Recurso al que se intenta acceder.

        Raises:
            AccessDeniedError: Si el usuario no tiene el permiso.
        """
        user = self._users.get(user_id)

        if user is None:
            logger.warning(
                "access_denied_unknown_user",
                extra={
                    "user_id": user_id,
                    "permission": permission.value,
                    "resource": resource,
                },
            )
            raise AccessDeniedError(user_id, permission.value, resource)

        if not user.has_permission(permission):
            logger.warning(
                "access_denied",
                extra={
                    "user_id": user_id,
                    "permission": permission.value,
                    "resource": resource,
                    "roles": [r.value for r in user.roles],
                },
            )
            raise AccessDeniedError(user_id, permission.value, resource)

        logger.debug(
            "access_granted",
            extra={
                "user_id": user_id,
                "permission": permission.value,
                "resource": resource,
            },
        )

    def check_series_access(
        self,
        user_id: str,
        series_id: str,
    ) -> None:
        """Valida acceso a una serie temporal específica.

        Args:
            user_id: ID del usuario.
            series_id: Identificador de la serie.

        Raises:
            AccessDeniedError: Si no tiene acceso a la serie.
        """
        user = self._users.get(user_id)

        if user is None:
            raise AccessDeniedError(user_id, "series_access", f"series_{series_id}")

        if not user.can_access_series(series_id):
            logger.warning(
                "series_access_denied",
                extra={
                    "user_id": user_id,
                    "series_id": series_id,
                    "allowed_series": list(user.allowed_series_ids),
                },
            )
            raise AccessDeniedError(
                user_id, "series_access", f"series_{series_id}"
            )

    def check_sensor_access(
        self,
        user_id: str,
        sensor_id: int,
    ) -> None:
        """Legacy: valida acceso a sensor por ID numérico."""
        self.check_series_access(user_id, str(sensor_id))


# --- Contexto de sistema (para procesos internos) ---
SYSTEM_CONTEXT = UserContext(
    user_id="system",
    roles={Role.ADMIN},
    allowed_sensor_ids=set(),  # Sin restricciones
)
