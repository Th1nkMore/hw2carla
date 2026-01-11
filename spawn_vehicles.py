#!/usr/bin/env python3
"""Spawn CARLA vehicles for debugging.

Supports:
- Spawning vehicles from dataset (player + NPCs at t=0)
- Spawning vehicles from map spawn points
- Spawning one-by-one (interactive) or all-at-once

Examples:
  # Spawn player + all NPCs from dataset (frame 0)
  python3 spawn_vehicles.py from-data --ip 10.16.90.246 --town Town06 --scene ChangeLane --mode all

  # Spawn one-by-one (press Enter each time)
  python3 spawn_vehicles.py from-data --ip 10.16.90.246 --town Town06 --scene IntersectionMerge --mode one

  # Spawn N vehicles at map spawn points
  python3 spawn_vehicles.py from-map --ip 10.16.90.246 --town Town06 --count 10 --mode all
"""

import argparse
import math
import time
from typing import List, Optional, Tuple

import carla

import data


IMPLIED_MIN_X = 103.92


def _resolve_blueprint(blueprint_library, token_or_pattern: str) -> Tuple[Optional[carla.ActorBlueprint], str, List[str]]:
    """Return (blueprint or None, pattern_used, sample_candidates)."""
    if not token_or_pattern:
        return None, token_or_pattern, []

    pattern = token_or_pattern if ("*" in token_or_pattern or "?" in token_or_pattern) else f"*{token_or_pattern}*"
    bp_list = blueprint_library.filter(pattern)
    if not bp_list:
        candidates = [bp.id for bp in blueprint_library.filter(f"*{token_or_pattern}*")]
        return None, pattern, candidates[:20]
    return bp_list[0], pattern, []


def _spawn_actor(world: carla.World, bp: carla.ActorBlueprint, transform: carla.Transform, name: str) -> Optional[carla.Actor]:
    """Spawn with helpful error messages."""
    actor = world.try_spawn_actor(bp, transform)
    if actor is not None:
        return actor

    try:
        return world.spawn_actor(bp, transform)
    except Exception as e:
        loc = transform.location
        rot = transform.rotation
        print(
            f"FAILED spawn {name}: blueprint={bp.id} "
            f"loc=({loc.x:.2f},{loc.y:.2f},{loc.z:.2f}) "
            f"rot=(pitch={rot.pitch:.1f},yaw={rot.yaw:.1f},roll={rot.roll:.1f})"
        )
        print(f"Spawn exception: {type(e).__name__}: {e}")
        return None


def _convert_highway_point_to_town(point: List[float], town_id: str) -> List[float]:
    """Convert [frame, x, y, yaw(rad)] -> [frame, x, y, z, pitch, yaw(deg), roll]."""
    if town_id in ("Town06", "Town06_Opt"):
        init_pose = [209, 66, 0.08]
        min_x = IMPLIED_MIN_X
    elif town_id in ("Town03", "Town03_Opt"):
        init_pose = [0, -1.5, 0.08]
        min_x = 0
    else:
        raise ValueError(f"Unsupported town_id: {town_id}")

    frame, x, y, yaw_rad = point
    return [
        frame,
        x + init_pose[0] - min_x,
        y + init_pose[1],
        init_pose[2],
        0.0,
        float(yaw_rad) * 180.0 / math.pi,
        0.0,
    ]


def _to_transform(p: List[float]) -> carla.Transform:
    # p: [frame, x, y, z, pitch, yaw, roll]
    return carla.Transform(
        carla.Location(x=float(p[1]), y=float(p[2]), z=float(p[3])),
        carla.Rotation(pitch=float(p[4]), yaw=float(p[5]), roll=float(p[6])),
    )


def _interactive_pause(i: int, label: str) -> bool:
    """Return False to stop."""
    try:
        s = input(f"[{i}] Ready to spawn {label}. Enter=spawn, q=quit: ").strip().lower()
    except EOFError:
        return True
    if s in ("q", "quit", "exit"):
        return False
    return True


def cmd_from_data(args) -> int:
    client = carla.Client(args.ip, args.port)
    client.set_timeout(args.timeout)

    world = client.load_world(args.town) if args.town else client.get_world()
    blueprint_library = world.get_blueprint_library()

    if args.sync:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

    # Load dataset
    raw = data.data_mix(scene=args.scene)
    hero, npcs = data.player_data_split(raw)

    # Convert first frame positions
    hero_pose = _convert_highway_point_to_town(hero[0], args.town)
    npc_poses = [_convert_highway_point_to_town(path[0], args.town) for path in npcs]

    spawned: List[carla.Actor] = []

    try:
        # NPCs
        npc_bp, npc_pattern, npc_candidates = _resolve_blueprint(blueprint_library, args.npc_model)
        if npc_bp is None:
            print(f"NPC model '{args.npc_model}' not found (pattern '{npc_pattern}').")
            if npc_candidates:
                print("Candidates:", npc_candidates)
            return 2

        if args.mode == "one":
            for i, pose in enumerate(npc_poses):
                if not _interactive_pause(i, f"NPC[{i}]"):
                    break
                actor = _spawn_actor(world, npc_bp, _to_transform(pose), name=f"NPC[{i}]")
                if actor is not None:
                    actor.set_simulate_physics(False)
                    actor.set_enable_gravity(False)
                    spawned.append(actor)
        else:
            for i, pose in enumerate(npc_poses):
                actor = _spawn_actor(world, npc_bp, _to_transform(pose), name=f"NPC[{i}]")
                if actor is not None:
                    actor.set_simulate_physics(False)
                    actor.set_enable_gravity(False)
                    spawned.append(actor)

        # Player
        if args.include_player:
            player_bp, player_pattern, player_candidates = _resolve_blueprint(blueprint_library, args.player_model)
            if player_bp is None:
                print(f"Player model '{args.player_model}' not found (pattern '{player_pattern}').")
                if player_candidates:
                    print("Candidates:", player_candidates)
                return 2

            if args.mode != "one" or _interactive_pause(999, "PLAYER"):
                actor = _spawn_actor(world, player_bp, _to_transform(hero_pose), name="PLAYER")
                if actor is not None:
                    actor.set_simulate_physics(False)
                    actor.set_enable_gravity(False)
                    spawned.append(actor)

        print(f"Spawned {len(spawned)} actors.")
        if args.keep_seconds is None:
            print("Keeping actors alive. Ctrl-C to cleanup.")
            while True:
                if args.sync:
                    world.tick()
                else:
                    world.wait_for_tick()
        else:
            end = time.time() + float(args.keep_seconds)
            while time.time() < end:
                if args.sync:
                    world.tick()
                else:
                    world.wait_for_tick()

        return 0

    except KeyboardInterrupt:
        return 0

    finally:
        for a in spawned:
            try:
                a.destroy()
            except Exception:
                pass


def cmd_from_map(args) -> int:
    client = carla.Client(args.ip, args.port)
    client.set_timeout(args.timeout)

    world = client.load_world(args.town) if args.town else client.get_world()
    blueprint_library = world.get_blueprint_library()

    if args.sync:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

    bp, pattern, candidates = _resolve_blueprint(blueprint_library, args.model)
    if bp is None:
        print(f"Model '{args.model}' not found (pattern '{pattern}').")
        if candidates:
            print("Candidates:", candidates)
        return 2

    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        print("No map spawn points found.")
        return 2

    count = min(args.count, len(spawn_points))
    spawned: List[carla.Actor] = []

    try:
        if args.mode == "one":
            for i in range(count):
                if not _interactive_pause(i, f"MAP[{i}]"):
                    break
                actor = _spawn_actor(world, bp, spawn_points[i], name=f"MAP[{i}]")
                if actor is not None:
                    actor.set_simulate_physics(False)
                    actor.set_enable_gravity(False)
                    spawned.append(actor)
        else:
            for i in range(count):
                actor = _spawn_actor(world, bp, spawn_points[i], name=f"MAP[{i}]")
                if actor is not None:
                    actor.set_simulate_physics(False)
                    actor.set_enable_gravity(False)
                    spawned.append(actor)

        print(f"Spawned {len(spawned)} actors at map spawn points.")
        if args.keep_seconds is None:
            print("Keeping actors alive. Ctrl-C to cleanup.")
            while True:
                if args.sync:
                    world.tick()
                else:
                    world.wait_for_tick()
        else:
            end = time.time() + float(args.keep_seconds)
            while time.time() < end:
                if args.sync:
                    world.tick()
                else:
                    world.wait_for_tick()

        return 0

    except KeyboardInterrupt:
        return 0

    finally:
        for a in spawned:
            try:
                a.destroy()
            except Exception:
                pass


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Spawn CARLA vehicles for debugging")
    p.add_argument("--ip", default="localhost")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--timeout", type=float, default=10.0)
    p.add_argument("--town", default=None, help="Town map to load, e.g. Town06")
    p.add_argument("--mode", choices=["one", "all"], default="all")
    p.add_argument("--keep-seconds", type=float, default=None, help="How long to keep actors alive (default: until Ctrl-C)")
    p.add_argument("--sync", action="store_true", help="Enable synchronous mode")

    sub = p.add_subparsers(dest="cmd", required=True)

    p_data = sub.add_parser("from-data", help="Spawn from dataset at frame 0")
    p_data.add_argument("--scene", default="ChangeLane")
    p_data.add_argument("--npc-model", default="model3")
    p_data.add_argument("--player-model", default="audi")
    p_data.add_argument("--include-player", dest="include_player", action="store_true", default=True)
    p_data.add_argument("--no-player", dest="include_player", action="store_false", help="Do not spawn the player vehicle")
    p_data.set_defaults(func=cmd_from_data)

    p_map = sub.add_parser("from-map", help="Spawn at CARLA map spawn points")
    p_map.add_argument("--model", default="model3")
    p_map.add_argument("--count", type=int, default=10)
    p_map.set_defaults(func=cmd_from_map)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
