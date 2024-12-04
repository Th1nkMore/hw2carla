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
IM_HEIIGHT = 760

def clean_up():
    file_list = glob.glob('test/*.jpg')
    for f in file_list:
        os.remove(f)
    os.makedirs('test', exist_ok=True)

def process_img(data):
    frame = data.frame
    i = np.array(data.raw_data)
    i2 = i.reshape((IM_HEIIGHT, IM_WIDTH, 4))
    i3 = i2[:, :, :3]
    filePath = os.path.join('test', 'test' + f'{frame:09d}' + '.jpg')
    # Save image in a separate thread
    threading.Thread(target=cv2.imwrite, args=(filePath, i3)).start()
    return i3 / 255.0

def img2video():
    img_array = []
    file_list = sorted(glob.glob('test/*.jpg'), key=lambda x: int(x.split('/')[-1].split('.')[0].split('t')[-1]))
    for filename in file_list:
        img = cv2.imread(filename)
        height, width, _ = img.shape
        size = (width, height)
        img_array.append(img)
    out = cv2.VideoWriter('highway2carla.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

class CarlaControl():
    def __init__(self, ip='localhost', port=2000):
        self.client = carla.Client(ip, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()

        self.actor_list =  []

    def change_map(self, TOWN='Town05'):
        self.world = self.client.load_world(TOWN)

    def create_car(self, car_name, position_x, position_y, position_z, position_p, position_yaw, position_r, car_model="audi"):
        spawn_point = Transform(Location(x=position_x, y=position_y, z=position_z), Rotation(pitch=position_p, yaw=position_yaw, roll=position_r))
        blueprint_library = self.world.get_blueprint_library()
        bp = blueprint_library.filter(car_model)[0]
        vehicle = self.world.spawn_actor(bp, spawn_point)
        vehicle.set_simulate_physics(False)
        self.actor_list.append([car_name, vehicle])
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

        # First camera
        # spawn_point = carla.Transform(carla.Location(x=1.5, z=1.5))
        # sensor = self.world.spawn_actor(cam_bp, spawn_point, attach_to=player_car)
        # sensor.listen(lambda data: process_img(data))
        # self.actor_list.append([-100, sensor])

        # Second camera
        spawn_point = carla.Transform(carla.Location(x=0, z=40), Rotation(yaw=0, pitch=-90))
        sensor = self.world.spawn_actor(cam_bp, spawn_point, attach_to=player_car)
        sensor.listen(lambda data: process_img(data))
        self.actor_list.append([-110, sensor])

    def play_video(self, my_car, npc_cars):
        print('create player car')
        player_car = self.create_car(0, my_car[0][1], my_car[0][2], my_car[0][3], my_car[0][4], my_car[0][5], my_car[0][6], car_model='audi')

        print('create camera')
        self.setup_sensors(player_car)

        print('create npc cars')
        for i in range(len(npc_cars)):
            self.create_car(i, npc_cars[i][0][1], npc_cars[i][0][2], npc_cars[i][0][3], npc_cars[i][0][4], npc_cars[i][0][5], npc_cars[i][0][6], car_model="model3")

        print('moving car')
        for time_count in range(1, len(npc_cars[0])):
            self.move_car(0, my_car[time_count][1], my_car[time_count][2], my_car[time_count][3], my_car[time_count][4], my_car[time_count][5], my_car[time_count][6])
            print(f"Self-car position: {my_car[time_count][1]}, {my_car[time_count][2]}")
            for i in range(len(npc_cars)):
                self.move_car(i, npc_cars[i][time_count][1], npc_cars[i][time_count][2], npc_cars[i][time_count][3], npc_cars[i][time_count][4], npc_cars[i][time_count][5], npc_cars[i][time_count][6])

            # Wait for the simulator to tick
            self.world.tick()
            time.sleep(0.05)  # Adjust sleep for desired update rate


class HighwayPathToCarlaPath():
    def __init__(self, path_lists):
        self.path_list = path_lists
        min_x = 103.92

        for path_planning in self.path_list:
            for point in path_planning:
                point[1] -= min_x

    def exchange_to_town06(self):
        self.init_pose = [20, 140, 0.08]
        town06_path = []
        for point_list in self.path_list:
            tmp_path = []
            for point in point_list:
                # [frame, x, y, z, pitch, yaw, roll]
                tmp_path.append([point[0], point[1] + self.init_pose[0], point[2] + self.init_pose[1], self.init_pose[2], 0, point[3] * 180 / math.pi, 0])
            town06_path.append(tmp_path)

        return town06_path

if __name__ == '__main__':
    try:
        self_list, actor_list = data.player_data_split(data.data_mix())

        carla_path = HighwayPathToCarlaPath(actor_list).exchange_to_town06()
        player_path = HighwayPathToCarlaPath([self_list]).exchange_to_town06()[0]

        player_path = sorted(player_path, key=lambda point: point[1])
        carla_control = CarlaControl(ip='10.16.90.246')
        carla_control.change_map('Town06')
        clean_up()
        time.sleep(2)
        carla_control.play_video(player_path, carla_path)

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        carla_control.close()
        img2video()
