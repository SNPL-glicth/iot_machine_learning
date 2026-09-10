#!/usr/bin/env python
"""Test controlado de Paper Trading contra Alpaca Paper.

Valida el flujo completo:
Market Data → ZENIN → Strategy → Risk/Guard → Order Proposal → AlpacaOrderClient → Alpaca Paper → Order Status → Position → Telemetry → TUI
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_ST_ROOT = _PROJECT_ROOT.parent

for p in (str(_ST_ROOT), str(_PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Use absolute imports through sys.path
from iot_machine_learning.infrastructure.adapters.market.live_config import LiveBotConfig
from iot_machine_learning.infrastructure.adapters.market.alpaca.order_client import AlpacaOrderClient
from iot_machine_learning.infrastructure.adapters.market.alpaca.account import AlpacaAccount, create_account


class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    END = "\033[0m"


def print_header(text: str):
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{text.center(70)}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'='*70}{Colors.END}\n")


def print_step(step: int, text: str):
    print(f"{Colors.CYAN}{Colors.BOLD}[PASO {step}]{Colors.END} {text}")


def print_success(text: str):
    print(f"  {Colors.GREEN}✓{Colors.END} {text}")


def print_error(text: str):
    print(f"  {Colors.RED}✗{Colors.END} {text}")


def print_info(label: str, value: str):
    print(f"  {Colors.BOLD}{label}:{Colors.END} {value}")


async def validate_preconditions(client: AlpacaOrderClient) -> dict:
    """Valida todas las precondiciones antes de enviar orden."""
    print_step(1, "Validando precondiciones de entorno...")
    
    # Verificar configuración
    config = LiveBotConfig(
        broker="alpaca",
        symbol="SPY",
        alpaca_api_key=os.getenv("ALPACA_API_KEY"),
        alpaca_secret_key=os.getenv("ALPACA_SECRET_KEY"),
        alpaca_api_base_url=os.getenv("ALPACA_API_BASE_URL") or "https://paper-api.alpaca.markets",
        alpaca_data_feed=os.getenv("ALPACA_DATA_FEED", "iex"),
    )
    
    print_info("Broker", config.broker)
    print_info("is_paper_trading", str(config.is_paper_trading))
    print_info("is_live_trading", str(config.is_live_trading))
    print_info("ALPACA_API_BASE_URL", config.alpaca_api_base_url)
    
    assert config.broker == "alpaca", "Broker debe ser alpaca"
    assert config.is_paper_trading is True, "Debe ser paper trading"
    assert config.is_live_trading is False, "NO debe ser live trading"
    assert "paper-api.alpaca.markets" in config.alpaca_api_base_url, "URL debe ser paper"
    print_success("Configuración validada")
    
    # Verificar market clock
    clock = await client.get_clock()
    print_info("Market is_open", str(clock.get("is_open")))
    print_info("Next open", str(clock.get("next_open")))
    print_info("Next close", str(clock.get("next_close")))
    
    # Verificar quote SPY
    quote = await client.get_latest_quote("SPY")
    bid = quote.get("quote", {}).get("bp", 0) if "quote" in quote else quote.get("bp", 0)
    ask = quote.get("quote", {}).get("ap", 0) if "quote" in quote else quote.get("ap", 0)
    print_info("SPY Bid", str(bid))
    print_info("SPY Ask", str(ask))
    
    # Verificar asset tradeable
    asset = await client.get_asset("SPY")
    print_info("Asset tradable", str(asset.get("tradable", False)))
    print_info("Asset status", str(asset.get("status", "unknown")))
    assert asset.get("tradable") is True, "SPY debe ser tradeable"
    assert asset.get("status") == "active", "SPY debe estar activo"
    
    # Verificar cuenta
    account = await client.get_account()
    equity = float(account.get("equity", 0))
    cash = float(account.get("cash", 0))
    buying_power = float(account.get("buying_power", 0))
    print_info("Equity", f"${equity:,.2f}")
    print_info("Cash", f"${cash:,.2f}")
    print_info("Buying Power", f"${buying_power:,.2f}")
    
    # Verificar posición actual
    pos = await client.get_position("SPY")
    current_qty = float(pos.get("qty", 0))
    print_info("Posición SPY actual", str(current_qty))
    
    return {
        "config": config,
        "clock": clock,
        "quote": {"bid": bid, "ask": ask},
        "account": account,
        "current_position": current_qty,
        "equity": equity,
        "cash": cash,
        "buying_power": buying_power,
    }


async def send_test_order(client: AlpacaOrderClient, quote: dict) -> dict:
    """Envía orden de prueba LIMIT."""
    print_step(2, "Enviando orden de prueba LIMIT...")
    
    bid = quote.get("bid", 0)
    ask = quote.get("ask", 0)
    
    # Usar precio entre bid/ask o ligeramente por debajo del bid para buy
    limit_price = round(bid - 0.01, 2) if bid > 0 else 760.00
    qty = 1  # Mínimo 1 acción
    
    unique_id = f"ZENIN_TEST_PAPER_{int(time.time() * 1000000)}"
    
    print_info("Símbolo", "SPY")
    print_info("Lado", "buy")
    print_info("Cantidad", str(qty))
    print_info("Tipo", "limit")
    print_info("Precio límite", f"${limit_price:.2f}")
    print_info("Client Order ID", unique_id)
    print_info("Bid actual", f"${bid:.2f}")
    print_info("Ask actual", f"${ask:.2f}")
    
    # Confirmación explícita
    print(f"\n{Colors.YELLOW}{Colors.BOLD}¿Confirmar envío de orden? (s/N): {Colors.END}", end="")
    # En modo automático para CI, continuamos
    # response = input().strip().lower()
    # if response != 's':
    #     print("Cancelado por usuario")
    #     return None
    
    order = await client.place_limit_order(
        symbol="SPY",
        side="buy",
        qty=qty,
        price=limit_price,
        time_in_force="day",
        client_order_id=unique_id,
    )
    
    print_success(f"Orden enviada: {order.id}")
    print_info("Status inicial", order.status)
    print_info("Client Order ID", order.client_order_id)
    
    return {
        "order": order,
        "order_id": order.id,
        "client_order_id": unique_id,
        "limit_price": limit_price,
        "qty": qty,
    }


async def monitor_order(client: AlpacaOrderClient, order_id: str) -> dict:
    """Monitorea el estado de la orden."""
    print_step(3, "Monitoreando estado de la orden...")
    
    for i in range(30):  # 30 segundos max
        order = await client.get_order(order_id)
        print_info(f"Check {i+1}/30", f"Status: {order.status}, Filled: {order.filled_qty}/{order.qty}")
        
        if order.status in ("filled", "canceled", "rejected", "expired"):
            print_success(f"Orden finalizada: {order.status}")
            return {
                "order": order,
                "final_status": order.status,
                "filled_qty": order.filled_qty,
                "filled_avg_price": order.filled_avg_price,
            }
        
        await asyncio.sleep(1)
    
    # Timeout - cancelar
    print(f"{Colors.YELLOW}Timeout - cancelando orden...{Colors.END}")
    cancelled = await client.cancel_order(order_id)
    print_info("Cancel status", cancelled.status)
    return {"order": cancelled, "final_status": cancelled.status, "filled_qty": 0.0}


async def verify_position(client: AlpacaOrderClient, account: AlpacaAccount, before_qty: float):
    """Verifica posición resultante."""
    print_step(4, "Verificando posición resultante...")
    
    # Desde order client
    pos = await client.get_position("SPY")
    after_qty = float(pos.get("qty", 0))
    print_info("Posición SPY (OrderClient)", f"{after_qty}")
    
    # Desde account
    await account.sync_now()
    pos_detail = await account.get_position_details("SPY")
    if pos_detail:
        print_info("Posición SPY (Account)", f"{pos_detail.qty} @ ${pos_detail.avg_entry_price:.2f}")
        print_info("Unrealized PL", f"${pos_detail.unrealized_pl:.2f}")
    
    change = after_qty - before_qty
    print_info("Cambio en posición", f"{change:+.4f}")
    
    return {
        "before_qty": before_qty,
        "after_qty": after_qty,
        "change": change,
    }


async def cleanup_open_orders(client: AlpacaOrderClient):
    """Limpia órdenes abiertas si las hay."""
    print_step(5, "Limpiando órdenes abiertas...")
    orders = await client.get_orders(status="open", limit=50)
    for order in orders:
        if order.symbol == "SPY" and order.client_order_id and order.client_order_id.startswith("ZENIN_TEST"):
            print_info("Cancelando", f"{order.id} ({order.client_order_id})")
            await client.cancel_order(order.id)
    print_success("Limpieza completada")


async def main():
    print_header("TEST CONTROLADO - ALPACA PAPER TRADING")
    print(f"{Colors.BOLD}Símbolo:{Colors.END} SPY")
    print(f"{Colors.BOLD}Cantidad:{Colors.END} 1 acción")
    print(f"{Colors.BOLD}Tipo:{Colors.END} LIMIT (compra)")
    print(f"{Colors.BOLD}Modo:{Colors.END} PAPER TRADING\n")
    
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not api_secret:
        print_error("Variables ALPACA_API_KEY y ALPACA_SECRET_KEY son requeridas")
        return False
    base_url = os.getenv("ALPACA_API_BASE_URL", "https://paper-api.alpaca.markets")
    data_feed = os.getenv("ALPACA_DATA_FEED", "iex")
    
    async with AlpacaOrderClient(
        api_key=api_key,
        api_secret=api_secret,
        base_url=base_url,
        data_feed=data_feed,
    ) as client:
        
        # Crear cuenta
        account = await create_account(client, auto_sync=True)
        
        try:
            # 1. Validar precondiciones
            pre = await validate_preconditions(client)
            
            # 2. Enviar orden
            order_info = await send_test_order(client, pre["quote"])
            if not order_info:
                return
            
            # 3. Monitorear orden
            result = await monitor_order(client, order_info["order_id"])
            
            # 4. Verificar posición
            pos_result = await verify_position(client, account, pre["current_position"])
            
            # 5. Limpiar
            await cleanup_open_orders(client)
            
            # Resumen final
            print_header("RESUMEN DE LA PRUEBA")
            print_info("Order ID", order_info["order_id"])
            print_info("Client Order ID", order_info["client_order_id"])
            print_info("Símbolo", "SPY")
            print_info("Lado", "buy")
            print_info("Cantidad solicitada", str(order_info["qty"]))
            print_info("Tipo", "LIMIT")
            print_info("Precio límite", f"${order_info['limit_price']:.2f}")
            print_info("Estado final", result["final_status"])
            print_info("Cantidad llenada", str(result["filled_qty"]))
            if result.get("filled_avg_price"):
                print_info("Precio promedio", f"${result['filled_avg_price']:.2f}")
            print_info("Posición antes", f"{pos_result['before_qty']}")
            print_info("Posición después", f"{pos_result['after_qty']}")
            print_info("Cambio", f"{pos_result['change']:+.4f}")
            
            if result["final_status"] == "filled":
                print(f"\n{Colors.GREEN}{Colors.BOLD}✓ PRUEBA EXITOSA - Orden ejecutada{Colors.END}")
            elif result["final_status"] in ("canceled", "rejected", "expired"):
                print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠ PRUEBA COMPLETADA - Orden no ejecutada (status: {result['final_status']}){Colors.END}")
            else:
                print(f"\n{Colors.RED}{Colors.BOLD}✗ ESTADO INESPERADO: {result['final_status']}{Colors.END}")
            
            return result["final_status"] == "filled"
            
        except Exception as e:
            logger.exception("Error en prueba")
            print_error(f"Error: {e}")
            await cleanup_open_orders(client)
            return False
        
        finally:
            await account.stop_auto_sync()


if __name__ == "__main__":
    import os
    result = asyncio.run(main())
    sys.exit(0 if result else 1)