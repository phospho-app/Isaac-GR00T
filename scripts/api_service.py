#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import threading
import time
import uuid
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from uvicorn import Config, Server

from gr00t.eval.robot import RobotInferenceServer
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy


class ServerConfig(BaseModel):
    model_path: str
    embodiment_tag: str = "gr1"
    data_config: str = "gr1_arms_waist"
    denoising_steps: int = 4
    port: int = 0  # 0 means auto-assign
    host: str = "localhost"


class ServerInfo(BaseModel):
    id: str
    config: ServerConfig
    status: str
    assigned_port: int
    start_time: float


class ServerResponse(BaseModel):
    id: str
    host: str
    port: int
    status: str


class GR00TServerManager:
    def __init__(self):
        self.servers: Dict[str, ServerInfo] = {}
        self.server_threads: Dict[str, threading.Thread] = {}
        self.next_port = 5555
        self.lock = threading.Lock()

    def _get_next_port(self) -> int:
        with self.lock:
            port = self.next_port
            self.next_port += 1
            return port

    def launch_server(self, config: ServerConfig) -> ServerInfo:
        server_id = str(uuid.uuid4())

        # Assign port if not specified
        if config.port == 0:
            config.port = self._get_next_port()

        # Create data config and policy
        data_config = DATA_CONFIG_MAP[config.data_config]
        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()

        policy = Gr00tPolicy(
            model_path=config.model_path,
            modality_config=modality_config,
            modality_transform=modality_transform,
            embodiment_tag=config.embodiment_tag,
            denoising_steps=config.denoising_steps,
        )

        # Create server info
        server_info = ServerInfo(
            id=server_id, config=config, status="starting", assigned_port=config.port, start_time=time.time()
        )
        self.servers[server_id] = server_info

        # Start server in a separate thread
        server_thread = threading.Thread(
            target=self._run_server, args=(server_id, policy, config.host, config.port), daemon=True
        )
        self.server_threads[server_id] = server_thread
        server_thread.start()

        # Wait briefly to ensure server starts
        time.sleep(1)
        server_info.status = "running"

        return server_info

    def _run_server(self, server_id: str, policy, host: str, port: int):
        try:
            server = RobotInferenceServer(policy, host=host, port=port)
            self.servers[server_id].status = "running"
            server.run()
        except Exception as e:
            print(f"Server {server_id} error: {e}")
            self.servers[server_id].status = f"error: {str(e)}"

    def stop_server(self, server_id: str) -> bool:
        if server_id not in self.servers:
            return False

        # TODO: Implement proper server shutdown mechanism
        # For now, we just mark it as stopped
        self.servers[server_id].status = "stopped"
        return True

    def get_server_info(self, server_id: str) -> Optional[ServerInfo]:
        return self.servers.get(server_id)

    def list_servers(self) -> List[ServerInfo]:
        return list(self.servers.values())


# Create FastAPI app
app = FastAPI(title="GR00T Inference API")
server_manager = GR00TServerManager()


@app.post("/servers", response_model=ServerResponse)
async def create_server(config: ServerConfig):
    """Launch a new inference server with the specified configuration."""
    try:
        server_info = server_manager.launch_server(config)
        return ServerResponse(
            id=server_info.id, host=config.host, port=server_info.assigned_port, status=server_info.status
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/servers", response_model=List[ServerResponse])
async def list_servers():
    """List all running inference servers."""
    servers = server_manager.list_servers()
    return [ServerResponse(id=s.id, host=s.config.host, port=s.assigned_port, status=s.status) for s in servers]


@app.get("/servers/{server_id}", response_model=ServerResponse)
async def get_server(server_id: str):
    """Get information about a specific inference server."""
    server_info = server_manager.get_server_info(server_id)
    if not server_info:
        raise HTTPException(status_code=404, detail="Server not found")

    return ServerResponse(
        id=server_info.id, host=server_info.config.host, port=server_info.assigned_port, status=server_info.status
    )


@app.delete("/servers/{server_id}", response_model=dict)
async def stop_server(server_id: str):
    """Stop a specific inference server."""
    success = server_manager.stop_server(server_id)
    if not success:
        raise HTTPException(status_code=404, detail="Server not found")

    return {"status": "success", "message": f"Server {server_id} stopped"}


@app.get("/data_configs", response_model=List[str])
async def list_data_configs():
    """List all available data configurations."""
    return list(DATA_CONFIG_MAP.keys())


def run_api_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the API server."""
    config = Config(app=app, host=host, port=port)
    server = Server(config=config)
    server.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GR00T Inference API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="API server host")
    parser.add_argument("--port", type=int, default=8000, help="API server port")
    args = parser.parse_args()

    print(f"Starting GR00T Inference API server on {args.host}:{args.port}...")
    run_api_server(host=args.host, port=args.port)
