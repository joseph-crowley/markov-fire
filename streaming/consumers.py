import json
from typing import Any

from channels.db import database_sync_to_async
from channels.generic.websocket import AsyncWebsocketConsumer

from simulation.models import SimulationRun, SimulationTick


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

    @database_sync_to_async
    def _run_exists(self, run_id: str) -> bool:
        return SimulationRun.objects.filter(id=run_id).exists()

    @database_sync_to_async
    def _fetch_ticks(self, from_index: int) -> list[dict[str, Any]]:
        ticks = SimulationTick.objects.filter(run_id=self.run_id, tick_index__gte=from_index).order_by('tick_index')[:200]
        return [
            {
                'tick': tick.tick_index,
                'activeCells': tick.active_cells,
                'burnedCells': tick.burned_cells,
                'suppressedCells': tick.suppressed_cells,
                'footprint': tick.footprint,
                'extinguished': False,
            }
            for tick in ticks
        ]

    async def _send_recent_ticks(self, from_index: int) -> None:
        ticks = await self._fetch_ticks(from_index)
        for tick in ticks:
            await self.send(text_data=json.dumps({'type': 'tick', **tick}))
