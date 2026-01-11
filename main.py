import copy
import glob
import math
import os
import sys

import data

import threading
import time

from carla import Transform, Location, Rotation
import time
import numpy as np
import cv2
import carla

IM_WIDTH = 1280
IM_HEIIGHT = 720
RECORDING = False

def clean_up():
    file_list = glob.glob('test/*.jpg')
    for f in file_list:
        os.remove(f)
    os.makedirs('test', exist_ok=True)

def process_img(data):
    if not RECORDING:
        return
    frame = data.frame
    i = np.array(data.raw_data)
    i2 = i.reshape((IM_HEIIGHT, IM_WIDTH, 4))
    i3 = i2[:, :, :3]
    filePath = os.path.join('test', 'test' + f'{frame:09d}' + '.jpg')
    # Save image in a separate thread
    threading.Thread(target=cv2.imwrite, args=(filePath, i3)).start()
    return i3 / 255.0

def img2video(scene='ChangeLane', view='Top'):
    img_array = []
    file_list = sorted(glob.glob('test/*.jpg'), key=lambda x: int(x.split('/')[-1].split('.')[0].split('t')[-1]))
    
    if not file_list:
        print("No images found to create video")
        return
    
    for filename in file_list:
        img = cv2.imread(filename)
        height, width, _ = img.shape
        size = (width, height)
        img_array.append(img)
    videoName = f'highway2carla_{scene}_{view}.mp4'
    out = cv2.VideoWriter(videoName, cv2.VideoWriter_fourcc(*'mp4v'), 15, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

class CarlaControl():
    def __init__(self, ip='localhost', port=2000, view='Top'):
        self.client = carla.Client(ip, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()

        self.settings = self.world.get_settings()
        self.settings.synchronous_mode = True # Enables synchronous mode
        self.settings.fixed_delta_seconds = 0.01
        self.world.apply_settings(self.settings)

        self.view = view

        self.actor_list =  []

    def _log_spawn_context(self, car_name, spawn_point, bp=None, car_model=None, err=None):
        try:
            loc = spawn_point.location
            rot = spawn_point.rotation
            bp_id = getattr(bp, "id", None)
            print(
                "Spawn context: "
                f"car_name={car_name} car_model={car_model} blueprint={bp_id} "
                f"loc=({loc.x:.2f},{loc.y:.2f},{loc.z:.2f}) rot=(pitch={rot.pitch:.1f},yaw={rot.yaw:.1f},roll={rot.roll:.1f})"
            )
            if err is not None:
                print(f"Spawn exception: {type(err).__name__}: {err}")

            # Best-effort hint: nearest existing vehicle
            try:
                vehicles = self.world.get_actors().filter('vehicle.*')
                min_d2 = None
                min_id = None
                for a in vehicles:
                    aloc = a.get_location()
                    dx = aloc.x - loc.x
                    dy = aloc.y - loc.y
                    dz = aloc.z - loc.z
                    d2 = dx * dx + dy * dy + dz * dz
                    if min_d2 is None or d2 < min_d2:
                        min_d2 = d2
                        min_id = a.id
                if min_d2 is not None:
                    print(f"Nearest existing vehicle: id={min_id} dist={min_d2 ** 0.5:.2f}m")
            except Exception:
                pass
        except Exception:
            # Never let logging break simulation
            pass

    def change_map(self, TOWN='Town05'):
        self.world = self.client.load_world(TOWN)
    
    def untoggle_layer(self, layer=carla.MapLayer.Buildings):
        self.world.unload_map_layer(layer)

    def create_car(self, car_name, position_x, position_y, position_z, position_p, position_yaw, position_r, car_model="audi"):
        spawn_point = Transform(Location(x=position_x, y=position_y, z=position_z), Rotation(pitch=position_p, yaw=position_yaw, roll=position_r))
        blueprint_library = self.world.get_blueprint_library()
        # CARLA blueprint filter matches patterns like 'vehicle.*' or '*model3*'.
        # If user passes a short token like 'audi'/'model3', treat it as a substring match.
        pattern = car_model if ('*' in car_model or '?' in car_model) else f"*{car_model}*"
        bp_list = blueprint_library.filter(pattern)
        if len(bp_list) == 0:
            candidates = [bp.id for bp in blueprint_library.filter(f"*{car_model}*")]
            print(
                f'Car model "{car_model}" not found. '
                f'Tried pattern "{pattern}". '
                f'Candidates containing token: {candidates[:15]}'
                + (" ..." if len(candidates) > 15 else "")
            )
            return None
        bp = bp_list[0]
        vehicle = self.world.try_spawn_actor(bp, spawn_point)
        if vehicle is None:
            # try_spawn_actor gives no reason; spawn_actor often raises with a message.
            try:
                vehicle = self.world.spawn_actor(bp, spawn_point)
            except Exception as e:
                print(f'Failed to create car {car_name} (try_spawn_actor returned None).')
                self._log_spawn_context(car_name, spawn_point, bp=bp, car_model=car_model, err=e)
                return None

        vehicle.set_simulate_physics(False)
        vehicle.set_enable_gravity(False)
        self.actor_list.append([car_name, vehicle])
        print(f'Car {car_name} created! Type: {vehicle} (blueprint={bp.id})')
        return vehicle

    def close(self):
        for _, actor in self.actor_list:
            actor.destroy()
        print("All cleaned up!")

    def move_car(self, car_name, position_x, position_y, position_z, position_p, position_yaw, position_r):
        spawn_point = Transform(Location(x=position_x, y=position_y, z=position_z), Rotation(pitch=position_p, yaw=position_yaw, roll=position_r))
        for i in range(len(self.actor_list)):
            if self.actor_list[i][0] == car_name:
                self.actor_list[i][1].set_transform(spawn_point)
                return

    def setup_sensors(self, player_car):
        blueprint_library = self.world.get_blueprint_library()
        cam_bp = blueprint_library.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
        cam_bp.set_attribute("image_size_y", f"{IM_HEIIGHT}")
        cam_bp.set_attribute("fov", "110")
        cam_bp.set_attribute("sensor_tick", "0.05")

        if self.view == 'Front':
            # Place the camera slightly above and behind the vehicle and tilt it down
            # so the vehicle's front/hood is visible along with the road ahead.
            spawn_point = carla.Transform(
                carla.Location(x=-3.5, z=4),
                carla.Rotation(pitch=-15, yaw=0, roll=0)
            )
            sensor = self.world.try_spawn_actor(cam_bp, spawn_point, attach_to=player_car)
            if sensor is not None:
                sensor.listen(lambda data: process_img(data))
                self.actor_list.append([-100, sensor])
            else:
                raise ValueError('Failed to create front view camera sensor')

        elif self.view == 'Top':
            spawn_point = carla.Transform(carla.Location(x=0, z=25), Rotation(yaw=90, pitch=-90))
            sensor = self.world.try_spawn_actor(cam_bp, spawn_point, attach_to=player_car)
            if sensor is not None:
                sensor.listen(lambda data: process_img(data))
                self.actor_list.append([-110, sensor])
            else:
                raise ValueError('Failed to create top view camera sensor')
        else:
            raise ValueError(f"Unsupported view: {self.view}")

    def play_video(self, my_car, npc_cars, player_car_model='audi'):

        print('create npc cars')
        for i in range(len(npc_cars)):
            self.create_car(i, npc_cars[i][0][1], npc_cars[i][0][2], npc_cars[i][0][3], npc_cars[i][0][4], npc_cars[i][0][5], npc_cars[i][0][6], car_model="model3")
        
        print('create player car')
        print(f"my_car length: {len(my_car)}")
        print(f"my_car[0]: {my_car[0]}")
        print(f"Accessing indices - [1]:{my_car[0][1]}, [2]:{my_car[0][2]}, [3]:{my_car[0][3]}, [4]:{my_car[0][4]}, [5]:{my_car[0][5]}, [6]:{my_car[0][6]}")
        player_car = self.create_car(-1, my_car[0][1], my_car[0][2], my_car[0][3], my_car[0][4], my_car[0][5], my_car[0][6], car_model=player_car_model)

        if player_car is None:
            print('Failed to create player car')
            return
        

        # Wait for car to be created
        time.sleep(1)

        print('create camera')
        self.setup_sensors(player_car)

        input("Press Enter to start moving cars...")
        global RECORDING
        RECORDING = True
        print('moving car')
        car_names = set([actor[0] for actor in self.actor_list])
        for time_count in range(1, len(my_car)):
            
            self.move_car(-1, my_car[time_count][1], my_car[time_count][2], my_car[time_count][3], my_car[time_count][4], my_car[time_count][5], my_car[time_count][6])
            for i in range(len(npc_cars)):
                if i not in car_names:
                    continue
                if time_count < len(npc_cars[i]):
                    self.move_car(i, npc_cars[i][time_count][1], npc_cars[i][time_count][2], npc_cars[i][time_count][3], npc_cars[i][time_count][4], npc_cars[i][time_count][5], npc_cars[i][time_count][6])
                
            # Wait for the simulator to tick
            self.world.tick()


class HighwayPathToCarlaPath():
    def __init__(self, path_lists):
        self.path_list = path_lists
        self.min_x = 103.92

        # for path_planning in self.path_list:
        #     for point in path_planning:
        #         point[1] -= min_x

    def exchange_to_town(self, town_id, yaw_offset_deg=0.0, pitch_deg=0.0, roll_deg=0.0):
        if town_id == 'Town06' or town_id == 'Town06_Opt':
            # init_pose = [x_offset, y_offset, z_height]
            # NOTE: the 3rd value is Z (height), NOT orientation.
            # self.init_pose = [280, 38, 0.08]
            self.init_pose = [260, 38, 0.08]
            # self.init_pose = [20, 140, 0.08]
            min_x = self.min_x
        elif town_id == 'Town03' or town_id == 'Town03_Opt':
            self.init_pose = [0, -1.5, 0.08]
            min_x = 0
        else:
            raise ValueError(f"Unsupported town_id: {town_id}")

        town_path = []
        for point_list in self.path_list:
            tmp_path = []
            for point in point_list:
                # [frame, x, y, z, pitch, yaw, roll]
                yaw_deg = (point[3] * 180 / math.pi) + yaw_offset_deg
                tmp_path.append([
                    point[0],
                    point[1] + self.init_pose[0] - min_x,
                    point[2] + self.init_pose[1],
                    self.init_pose[2],
                    pitch_deg,
                    yaw_deg,
                    roll_deg,
                ])
            town_path.append(tmp_path)

        return town_path

if __name__ == '__main__':
    scene = 'IntersectionMerge'
    view = 'Top'
    town_id = 'Town06'

    # Orientation tuning (degrees).
    # - "global_*" applies to BOTH hero + NPCs (keeps same reference frame)
    # - "hero_extra_*" applies only to the hero on top of global
    # Common fixes: set yaw offset to 90 or -90 if cars face sideways.
    global_yaw_offset_deg = 0.0
    global_pitch_deg = 0.0
    global_roll_deg = 0.0

    hero_extra_yaw_offset_deg = 0.0
    hero_extra_pitch_deg = 0.0
    hero_extra_roll_deg = 0.0

    carla_control = None
    try:
        self_list, actor_list = data.player_data_split(data.data_mix(scene=scene))
        
        print(f"Player trajectory points: {len(self_list)}")
        print(f"Number of NPC cars: {len(actor_list)}")
        if len(self_list) > 0:
            print(f"First player point: {self_list[0]}")
        if len(actor_list) > 0 and len(actor_list[0]) > 0:
            print(f"First NPC point: {actor_list[0][0]}")

        carla_path = HighwayPathToCarlaPath(actor_list).exchange_to_town(
            town_id,
            yaw_offset_deg=global_yaw_offset_deg,
            pitch_deg=global_pitch_deg,
            roll_deg=global_roll_deg,
        )
        player_path = HighwayPathToCarlaPath([self_list]).exchange_to_town(
            town_id,
            yaw_offset_deg=global_yaw_offset_deg + hero_extra_yaw_offset_deg,
            pitch_deg=global_pitch_deg + hero_extra_pitch_deg,
            roll_deg=global_roll_deg + hero_extra_roll_deg,
        )[0]
        
        print(f"Player path length after conversion: {len(player_path)}")
        if len(player_path) > 0:
            print(f"First player path point: {player_path[0]}")

        carla_control = CarlaControl(ip='10.16.90.246', view=view)
        carla_control.change_map(town_id)
        carla_control.untoggle_layer()
        clean_up()
        time.sleep(2)
        carla_control.play_video(player_path, carla_path, player_car_model='model3')
        # carla_control.play_video(player_path, carla_path)

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        if carla_control is not None:
            carla_control.close()
        img2video(scene=scene, view=view)
