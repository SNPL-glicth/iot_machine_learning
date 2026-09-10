import asyncio
import os
from dotenv import load_dotenv

load_dotenv('../.env')
load_dotenv('.env')

from iot_machine_learning.infrastructure.adapters.market.alpaca.order_client import AlpacaOrderClient
from iot_machine_learning.infrastructure.adapters.market.alpaca.account import AlpacaAccount

async def main():
    client = AlpacaOrderClient(
        os.getenv('ALPACA_API_KEY'),
        os.getenv('ALPACA_SECRET_KEY'),
        base_url=os.getenv('ALPACA_API_BASE_URL', 'https://paper-api.alpaca.markets/v2')
    )
    acc = AlpacaAccount(client)
    pos = await acc.get_all_positions()
    print("=== POSICIONES ABIERTAS ===")
    for sym, p in pos.items():
        print(f"Símbolo: {sym} | Lado: {p.side} | Cantidad: {p.qty} | Entrada: ${p.avg_entry_price:.2f} | Actual: ${p.current_price:.2f} | PnL No Realizado: ${p.unrealized_pl:+.2f} ({p.unrealized_plpc*100:+.2f}%)")
    if not pos:
        print("Ninguna posición abierta (Flat).")

    raw_acc = await client.get_account()
    equity = float(raw_acc.get('equity', 0))
    cash = float(raw_acc.get('cash', 0))
    print(f"\n=== ESTADO DE LA CUENTA ===")
    print(f"Equity: ${equity:,.2f} USD")
    print(f"Cash:   ${cash:,.2f} USD")
    print(f"Ganancia neta total hoy vs $100k: ${equity - 100000.0:+.2f} USD")

    orders = await client.get_orders(status='all', limit=20)
    print(f"\n=== ÚLTIMAS ÓRDENES (Últimas 15) ===")
    for o in orders[:15]:
        print(f"ID: {o.id[:8]}.. | Sym: {o.symbol:4} | Lado: {o.side:4} | Qty: {o.qty:4} | Tipo: {o.order_type:6} | Estado: {o.status:9} | Precio Ejec: {o.filled_avg_price}")

if __name__ == '__main__':
    asyncio.run(main())
