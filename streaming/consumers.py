import io
import json
from typing import Any, List, Tuple

from channels.db import database_sync_to_async
from channels.generic.websocket import AsyncWebsocketConsumer

from simulation.models import SimulationRun, SimulationTick
import numpy as np


class SimulationConsumer(AsyncWebsocketConsumer):
    async def connect(self) -> None:
        run_id = self.scope['url_route']['kwargs']['run_id']
        self.run_group_name = f'simulation_{run_id}'
        self.run_id = run_id

        run_exists = await self._run_exists(run_id)
        if not run_exists:
            await self.close(code=4040)
            return

        await self.channel_layer.group_add(self.run_group_name, self.channel_name)
        await self.accept()
        await self._send_snapshot()

    async def disconnect(self, close_code: int) -> None:
        await self.channel_layer.group_discard(self.run_group_name, self.channel_name)

    async def receive(self, text_data: str | None = None, bytes_data: bytes | None = None) -> None:
        if text_data:
            payload = json.loads(text_data)
            if payload.get('type') == 'catchup':
                await self._send_recent_ticks(payload.get('from', 0))

    async def simulation_tick(self, event: dict[str, Any]) -> None:
        await self.send(text_data=json.dumps({'type': 'tick', **event['message']}))

    async def simulation_completed(self, event: dict[str, Any]) -> None:
        await self.send(text_data=json.dumps({'type': 'completed', **event['message']}))

    async def simulation_failed(self, event: dict[str, Any]) -> None:
        await self.send(text_data=json.dumps({'type': 'failed', **event['message']}))

    async def simulation_paused(self, event: dict[str, Any]) -> None:
        await self.send(text_data=json.dumps({'type': 'paused', **event['message']}))

    @database_sync_to_async
    def _run_exists(self, run_id: str) -> bool:
        return SimulationRun.objects.filter(id=run_id).exists()

    async def _send_recent_ticks(self, from_index: int) -> None:
        ticks, latest_grid = await self._collect_ticks(from_index)
        if not ticks:
            return
        await self.send(text_data=json.dumps({
            'type': 'catchup',
            'ticks': ticks,
            'latestGrid': latest_grid,
        }))

    async def _send_snapshot(self) -> None:
        ticks, latest_grid = await self._collect_ticks(0)
        if not ticks:
            return
        await self.send(text_data=json.dumps({
            'type': 'snapshot',
            'ticks': ticks,
            'latestGrid': latest_grid,
        }))

    @database_sync_to_async
    def _collect_ticks(self, from_index: int) -> Tuple[List[dict], List[List[int]] | None]:
        qs = SimulationTick.objects.filter(run_id=self.run_id)
        if from_index:
            qs = qs.filter(tick_index__gte=from_index)
        ticks = list(qs.order_by('tick_index'))
        payload = []
        latest_grid = None
        for tick in ticks:
            payload.append({
                'tick': tick.tick_index,
                'activeCells': tick.active_cells,
                'burnedCells': tick.burned_cells,
                'suppressedCells': tick.suppressed_cells,
                'footprint': tick.footprint,
                'extinguished': bool(tick.active_cells == 0),
            })
            latest_grid = self._decode_grid(tick.grid_payload)
        return payload, latest_grid

    @staticmethod
    def _decode_grid(binary: bytes) -> List[List[int]]:
        if not binary:
            return None
        buffer = io.BytesIO(binary)
        array = np.load(buffer, allow_pickle=False)
        return array.astype(int).tolist()
