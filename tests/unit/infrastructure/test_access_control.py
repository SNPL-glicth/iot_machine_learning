"""Tests para AccessControlService y RBAC.

Verifica:
- Roles y permisos correctos
- Acceso denegado para usuarios sin permiso
- Acceso a sensores por whitelist
- Registro/des-registro de usuarios
- Contexto de sistema (admin sin restricciones)
"""

from __future__ import annotations

import pytest

from iot_machine_learning.infrastructure.security.access_control import (
    AccessControlService,
    AccessDeniedError,
    Permission,
    Role,
    SYSTEM_CONTEXT,
    UserContext,
)


class TestUserContext:
    """Tests para UserContext."""

    def test_viewer_permissions(self) -> None:
        """Viewer solo tiene permisos de lectura."""
        ctx = UserContext(user_id="viewer1", roles={Role.VIEWER})

        assert ctx.has_permission(Permission.READ_PREDICTIONS) is True
        assert ctx.has_permission(Permission.READ_METRICS) is True
        assert ctx.has_permission(Permission.EXECUTE_PREDICTION) is False
        assert ctx.has_permission(Permission.WRITE_CONFIG) is False

    def test_operator_permissions(self) -> None:
        """Operator puede leer y ejecutar predicciones."""
        ctx = UserContext(user_id="op1", roles={Role.OPERATOR})

        assert ctx.has_permission(Permission.READ_SENSOR_DATA) is True
        assert ctx.has_permission(Permission.EXECUTE_PREDICTION) is True
        assert ctx.has_permission(Permission.WRITE_CONFIG) is False
        assert ctx.has_permission(Permission.READ_AUDIT_LOGS) is False

    def test_admin_has_all_permissions(self) -> None:
        """Admin tiene todos los permisos."""
        ctx = UserContext(user_id="admin1", roles={Role.ADMIN})

        for perm in Permission:
            assert ctx.has_permission(perm) is True, f"Admin sin permiso {perm}"

    def test_auditor_permissions(self) -> None:
        """Auditor puede leer logs y métricas."""
        ctx = UserContext(user_id="aud1", roles={Role.AUDITOR})

        assert ctx.has_permission(Permission.READ_AUDIT_LOGS) is True
        assert ctx.has_permission(Permission.READ_METRICS) is True
        assert ctx.has_permission(Permission.EXECUTE_PREDICTION) is False
        assert ctx.has_permission(Permission.WRITE_CONFIG) is False

    def test_multiple_roles_union(self) -> None:
        """Múltiples roles combinan permisos."""
        ctx = UserContext(user_id="multi", roles={Role.VIEWER, Role.AUDITOR})

        assert ctx.has_permission(Permission.READ_PREDICTIONS) is True
        assert ctx.has_permission(Permission.READ_AUDIT_LOGS) is True
        assert ctx.has_permission(Permission.EXECUTE_PREDICTION) is False

    def test_sensor_access_with_whitelist(self) -> None:
        """Acceso a sensores restringido por whitelist."""
        ctx = UserContext(
            user_id="op1",
            roles={Role.OPERATOR},
            allowed_sensor_ids={1, 5, 42},
        )

        assert ctx.can_access_sensor(1) is True
        assert ctx.can_access_sensor(5) is True
        assert ctx.can_access_sensor(99) is False

    def test_sensor_access_no_whitelist_allows_all(self) -> None:
        """Sin whitelist → acceso a todos los sensores."""
        ctx = UserContext(user_id="op1", roles={Role.OPERATOR})

        assert ctx.can_access_sensor(1) is True
        assert ctx.can_access_sensor(999) is True

    def test_admin_no_whitelist_allows_all(self) -> None:
        """Admin sin whitelist → acceso a todos."""
        ctx = UserContext(user_id="admin1", roles={Role.ADMIN})

        assert ctx.can_access_sensor(1) is True
        assert ctx.can_access_sensor(999) is True


class TestAccessControlService:
    """Tests para AccessControlService."""

    def test_register_and_check(self) -> None:
        """Registrar usuario y verificar permiso."""
        svc = AccessControlService()
        ctx = UserContext(user_id="op1", roles={Role.OPERATOR})
        svc.register_user(ctx)

        # No debe lanzar excepción
        svc.check_permission("op1", Permission.EXECUTE_PREDICTION)

    def test_check_denied_raises(self) -> None:
        """Permiso denegado debe lanzar AccessDeniedError."""
        svc = AccessControlService()
        ctx = UserContext(user_id="viewer1", roles={Role.VIEWER})
        svc.register_user(ctx)

        with pytest.raises(AccessDeniedError) as exc_info:
            svc.check_permission("viewer1", Permission.WRITE_CONFIG, "ML_TAYLOR_ORDER")

        assert "viewer1" in str(exc_info.value)
        assert "write_config" in str(exc_info.value)

    def test_unknown_user_denied(self) -> None:
        """Usuario no registrado debe ser denegado."""
        svc = AccessControlService()

        with pytest.raises(AccessDeniedError):
            svc.check_permission("unknown", Permission.READ_PREDICTIONS)

    def test_deregister_user(self) -> None:
        """Des-registrar usuario debe revocar acceso."""
        svc = AccessControlService()
        ctx = UserContext(user_id="op1", roles={Role.OPERATOR})
        svc.register_user(ctx)

        svc.deregister_user("op1")

        with pytest.raises(AccessDeniedError):
            svc.check_permission("op1", Permission.READ_PREDICTIONS)

    def test_check_sensor_access(self) -> None:
        """Verificar acceso a sensor específico."""
        svc = AccessControlService()
        ctx = UserContext(
            user_id="op1",
            roles={Role.OPERATOR},
            allowed_sensor_ids={1, 5},
        )
        svc.register_user(ctx)

        svc.check_sensor_access("op1", 1)  # OK

        with pytest.raises(AccessDeniedError):
            svc.check_sensor_access("op1", 99)

    def test_get_user(self) -> None:
        """get_user retorna contexto o None."""
        svc = AccessControlService()
        ctx = UserContext(user_id="op1", roles={Role.OPERATOR})
        svc.register_user(ctx)

        assert svc.get_user("op1") is ctx
        assert svc.get_user("unknown") is None


class TestSystemContext:
    """Tests para el contexto de sistema."""

    def test_system_is_admin(self) -> None:
        """Contexto de sistema debe ser admin."""
        assert Role.ADMIN in SYSTEM_CONTEXT.roles

    def test_system_accesses_all_sensors(self) -> None:
        """Sistema accede a todos los sensores."""
        assert SYSTEM_CONTEXT.can_access_sensor(1) is True
        assert SYSTEM_CONTEXT.can_access_sensor(999) is True
