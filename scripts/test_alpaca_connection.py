#!/usr/bin/env python
"""Test de conectividad Alpaca Paper Trading.

Verifica:
1. Autenticación
2. Acceso a la cuenta
3. Estado de la cuenta
4. Buying power / Equity
5. Acceso a market data
6. Manejo correcto de errores de autenticación/network/API

Uso:
    python scripts/test_alpaca_connection.py
    python scripts/test_alpaca_connection.py --verbose
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load order client directly (avoiding package init issues)
import importlib.util

spec = importlib.util.spec_from_file_location(
    "alpaca_order_client",
    Path(__file__).resolve().parent.parent / "infrastructure" / "adapters" / "market" / "alpaca" / "order_client.py"
)
order_client_module = importlib.util.module_from_spec(spec)
sys.modules["alpaca_order_client"] = order_client_module
spec.loader.exec_module(order_client_module)
AlpacaOrderClient = order_client_module.AlpacaOrderClient

from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    END = "\033[0m"


def print_header(text: str):
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{text.center(60)}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'='*60}{Colors.END}\n")


def print_test(name: str, passed: bool, details: str = ""):
    status = f"{Colors.GREEN}✓ PASS{Colors.END}" if passed else f"{Colors.RED}✗ FAIL{Colors.END}"
    print(f"  {status} {name}")
    if details:
        color = Colors.CYAN if passed else Colors.YELLOW
        print(f"    {color}{details}{Colors.END}")


def print_info(label: str, value: str):
    print(f"  {Colors.BOLD}{label}:{Colors.END} {value}")


async def test_authentication(client: AlpacaOrderClient) -> tuple[bool, str]:
    """Test 1: Autenticación básica."""
    try:
        account = await client.get_account()
        account_id = account.get("id", "unknown")
        return True, f"Account ID: {account_id}"
    except RuntimeError as e:
        if "401" in str(e) or "Authentication" in str(e):
            return False, f"Authentication failed: {e}"
        raise


async def test_account_access(client: AlpacaOrderClient) -> tuple[bool, str]:
    """Test 2: Acceso a datos de cuenta."""
    try:
        account = await client.get_account()
        required_fields = ["id", "equity", "cash", "buying_power", "portfolio_value", "status"]
        missing = [f for f in required_fields if f not in account]
        if missing:
            return False, f"Missing fields: {missing}"
        return True, f"All required fields present ({len(required_fields)} fields)"
    except Exception as e:
        return False, f"Account access failed: {e}"


async def test_account_status(client: AlpacaOrderClient) -> tuple[bool, str]:
    """Test 3: Estado de la cuenta (trading_blocked, PDT, etc.)."""
    try:
        account = await client.get_account()
        status_info = {
            "status": account.get("status"),
            "trading_blocked": account.get("trading_blocked"),
            "account_blocked": account.get("account_blocked"),
            "transfers_blocked": account.get("transfers_blocked"),
            "pattern_day_trader": account.get("pattern_day_trader"),
        }
        details = ", ".join(f"{k}={v}" for k, v in status_info.items())
        is_tradeable = not any([
            status_info["trading_blocked"],
            status_info["account_blocked"],
            status_info["transfers_blocked"],
        ])
        return is_tradeable, details
    except Exception as e:
        return False, f"Account status check failed: {e}"


async def test_equity_buying_power(client: AlpacaOrderClient) -> tuple[bool, str]:
    """Test 4: Equity y Buying Power."""
    try:
        account = await client.get_account()
        equity = float(account.get("equity", 0))
        cash = float(account.get("cash", 0))
        buying_power = float(account.get("buying_power", 0))
        portfolio_value = float(account.get("portfolio_value", 0))

        details = f"Equity: ${equity:,.2f}, Cash: ${cash:,.2f}, Buying Power: ${buying_power:,.2f}, Portfolio: ${portfolio_value:,.2f}"

        if equity <= 0:
            return False, f"Equity inválido: ${equity}"
        if buying_power < 0:
            return False, f"Buying power negativo: ${buying_power}"

        return True, details
    except Exception as e:
        return False, f"Equity/buying power check failed: {e}"


async def test_positions(client: AlpacaOrderClient) -> tuple[bool, str]:
    """Test 5: Acceso a posiciones."""
    try:
        positions = await client.get_positions()
        details = f"Posiciones abiertas: {len(positions)}"
        for pos in positions:
            details += f"\n    {pos['symbol']}: {pos['qty']} @ ${pos['avg_entry_price']} (PL: ${pos['unrealized_pl']})"
        return True, details
    except Exception as e:
        return False, f"Positions access failed: {e}"


async def test_market_data(client: AlpacaOrderClient) -> tuple[bool, str]:
    """Test 6: Acceso a market data (quotes, trades, bars)."""
    test_symbol = "SPY"
    try:
        quote = await client.get_latest_quote(test_symbol)
        bid = quote.get("quote", {}).get("bp", 0) if "quote" in quote else quote.get("bp", 0)
        ask = quote.get("quote", {}).get("ap", 0) if "quote" in quote else quote.get("ap", 0)

        trade = await client.get_latest_trade(test_symbol)
        price = trade.get("trade", {}).get("p", 0) if "trade" in trade else trade.get("p", 0)

        bars = await client.get_bars(test_symbol, timeframe="1Min", limit=1)
        bar_count = len(bars.get("bars", [])) if "bars" in bars else len(bars)

        details = f"Quote: bid={bid}, ask={ask} | Trade: ${price} | Bars: {bar_count}"
        return True, details
    except Exception as e:
        return False, f"Market data access failed: {e}"


async def test_clock(client: AlpacaOrderClient) -> tuple[bool, str]:
    """Test 7: Reloj del mercado (horario de trading)."""
    try:
        clock = await client.get_clock()
        is_open = clock.get("is_open", False)
        next_open = clock.get("next_open", "N/A")
        next_close = clock.get("next_close", "N/A")
        details = f"Market open: {is_open} | Next open: {next_open} | Next close: {next_close}"
        return True, details
    except Exception as e:
        return False, f"Clock access failed: {e}"


async def test_order_submission_dry_run(client: AlpacaOrderClient) -> tuple[bool, str]:
    """Test 8: Simulación de envío de orden (dry-run, cancelación inmediata)."""
    test_symbol = "SPY"
    unique_id = f"ZENIN_TEST_CONNECT_{int(time.time() * 1000)}"
    try:
        # Intentar colocar orden LIMIT con precio muy bajo para que no se ejecute
        order = await client.place_limit_order(
            symbol=test_symbol,
            side="buy",
            qty=1,
            price=0.01,
            time_in_force="day",
            client_order_id=unique_id,
        )

        order_id = order.id
        status = order.status

        # Cancelar inmediatamente
        cancelled_order = await client.cancel_order(order_id)
        cancelled_status = cancelled_order.status

        details = f"Order ID: {order_id} | Status: {status} -> Cancelled: {cancelled_status}"
        return cancelled_status in ("canceled", "cancelled"), details
    except Exception as e:
        return False, f"Order submission test failed: {e}"


async def test_error_handling(client: AlpacaOrderClient) -> tuple[bool, str]:
    """Test 9: Manejo de errores (asset inválido)."""
    try:
        await client.get_asset("THIS_SYMBOL_DOES_NOT_EXIST_12345")
        return False, "Debería haber fallado con asset inválido"
    except RuntimeError as e:
        if "404" in str(e) or "not found" in str(e).lower():
            return True, f"Correctamente maneja 404: {e}"
        return False, f"Error inesperado: {e}"
    except Exception as e:
        return False, f"Error type inesperado: {type(e).__name__}: {e}"


async def test_rate_limiting(client: AlpacaOrderClient) -> tuple[bool, str]:
    """Test 10: Rate limiting básico (verificar que no falla bajo carga ligera)."""
    try:
        tasks = [client.get_clock() for _ in range(5)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        success = sum(1 for r in results if not isinstance(r, Exception))
        details = f"Requests exitosas: {success}/5"
        return success == 5, details
    except Exception as e:
        return False, f"Rate limiting test failed: {e}"


async def run_all_tests(verbose: bool = False) -> dict:
    """Ejecuta todos los tests de conectividad."""

    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_SECRET_KEY")
    base_url = os.getenv("ALPACA_API_BASE_URL", "https://paper-api.alpaca.markets/v2")
    data_feed = os.getenv("ALPACA_DATA_FEED", "iex")

    if not api_key or not api_secret:
        print(f"{Colors.RED}ERROR: ALPACA_API_KEY y ALPACA_SECRET_KEY requeridos en .env{Colors.END}")
        return {"passed": 0, "failed": 0, "total": 0}

    if "paper-api.alpaca.markets" not in base_url:
        print(f"{Colors.RED}ERROR: ALPACA_API_BASE_URL debe ser paper-api.alpaca.markets{Colors.END}")
        return {"passed": 0, "failed": 0, "total": 0}

    print_header("ALPACA PAPER TRADING - CONNECTIVITY TEST")
    print_info("Base URL", base_url)
    print_info("Data Feed", data_feed)
    print_info("API Key", f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "***")

    async with AlpacaOrderClient(
        api_key=api_key,
        api_secret=api_secret,
        base_url=base_url,
        data_feed=data_feed,
    ) as client:

        tests = [
            ("Authentication", test_authentication(client)),
            ("Account Access", test_account_access(client)),
            ("Account Status", test_account_status(client)),
            ("Equity & Buying Power", test_equity_buying_power(client)),
            ("Positions", test_positions(client)),
            ("Market Data", test_market_data(client)),
            ("Market Clock", test_clock(client)),
            ("Order Submission (dry-run)", test_order_submission_dry_run(client)),
            ("Error Handling", test_error_handling(client)),
            ("Rate Limiting", test_rate_limiting(client)),
        ]

        results = {}
        passed = 0
        failed = 0

        for name, coro in tests:
            if verbose:
                print(f"\n{Colors.YELLOW}Running: {name}...{Colors.END}")
            try:
                success, details = await coro
                results[name] = (success, details)
                if success:
                    passed += 1
                else:
                    failed += 1
                print_test(name, success, details)
            except Exception as e:
                failed += 1
                results[name] = (False, f"Exception: {e}")
                print_test(name, False, f"Exception: {e}")

    print_header("RESUMEN")
    total = passed + failed
    print_info("Total tests", str(total))
    print_info(f"{Colors.GREEN}Passed{Colors.END}", str(passed))
    print_info(f"{Colors.RED}Failed{Colors.END}", str(failed))

    if failed == 0:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ TODOS LOS TESTS PASARON{Colors.END}")
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}✗ {failed} TEST(S) FALLARON{Colors.END}")

    return {"passed": passed, "failed": failed, "total": total, "results": results}


def main():
    parser = argparse.ArgumentParser(description="Test Alpaca Paper Trading connectivity")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    result = asyncio.run(run_all_tests(verbose=args.verbose))
    sys.exit(0 if result["failed"] == 0 else 1)


if __name__ == "__main__":
    main()