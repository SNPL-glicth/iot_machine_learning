#!/usr/bin/env python3
"""Entry point para ejecutar ZeninQueuePoller como daemon."""

import logging
import sys
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Import poller
from iot_machine_learning.ml_service.workers.zenin_queue_poller import ZeninQueuePoller

def main():
    print("Starting ZeninQueuePoller...", flush=True)
    poller = ZeninQueuePoller()
    print(f"Poller configured: interval={poller.poll_interval}s, batch={poller.batch_size}", flush=True)
    print("Starting polling loop (Ctrl+C to stop)...", flush=True)
    poller.start()

if __name__ == "__main__":
    main()
