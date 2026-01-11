import copy
import os
import numpy as np
from typing import List, Optional
SP_NUM = 20

_DEFAULT_DATA_FILES = (
    "car_data_mix.npy",
    "data.npy",
)


def _resolve_scene_dir(scene: str, data_root: str = "data") -> str:
    """Resolve scenario directory name with a case-insensitive fallback."""
    scene = (scene or "").strip()
    if not scene:
        raise ValueError("scene must be a non-empty string")

    direct = os.path.join(data_root, scene)
    if os.path.isdir(direct):
        return scene

    if os.path.isdir(data_root):
        for entry in os.listdir(data_root):
            if entry.lower() == scene.lower() and os.path.isdir(os.path.join(data_root, entry)):
                return entry

    return scene


def _find_first_existing(path_candidates: List[str]) -> str:
    for p in path_candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "No data file found. Tried: " + ", ".join(path_candidates)
    )


def data_mix(scene: str = "ChangeLane", *, max_frames: Optional[int] = None, data_root: str = "data"):
    """Load a scenario dataset.

    Expected shape: (T, N, 4) where each entry is [frame, x, y, yaw].
    """
    scene_dir = _resolve_scene_dir(scene, data_root=data_root)

    base_dir = os.path.join(data_root, scene_dir)
    path = _find_first_existing([os.path.join(base_dir, f) for f in _DEFAULT_DATA_FILES])

    print(f"reading {scene_dir} from {path}")
    data = np.load(path, allow_pickle=False)

    if not isinstance(data, np.ndarray) or data.ndim != 3 or data.shape[-1] != 4:
        raise ValueError(
            f"Unexpected data format for scene '{scene_dir}': shape={getattr(data,'shape',None)}"
        )

    if max_frames is None and scene_dir.lower() == "changelane":
        max_frames = 200
    if max_frames is not None:
        data = data[:max_frames]
    return data

def player_data_split(datas):
    '''
    INPUT
    datas: [frame, car, [frame, x, y, yaw]]

    OUTPUT
    player_data: [frame, x, y, yaw]
    npc_data: [[frame, x, y, yaw]]    
    '''
    player_data=[]

    npc_data=[]
    for i in range(1,len(datas[0])):
        npc_data.append([])
    for t in range(len(datas)):
        player_data.append(datas[t][0])
        for i in range(1,len(datas[0])):
            npc_data[i-1].append(datas[t][i])

    sp_num=SP_NUM
    player_data1=route_extend(player_data,sp_num)
    player_data=player_data1

    carla_data1=[]
    for path in npc_data:
        carla_data1.append(route_extend(path,sp_num))
    npc_data=carla_data1

    return player_data, npc_data


def route_extend(path,sp_num):
    ans=[]
    count_turn=0
    for t in range(1,len(path)):
        delta_x=path[t][1]-path[t-1][1]
        delta_y=path[t][2]-path[t-1][2]
        delta_theta=path[t][3]-path[t-1][3]
        for count in range(sp_num):
            count_turn+=1
            ans.append([count_turn,path[t-1][1]+delta_x*count/sp_num,path[t-1][2]+delta_y*count/sp_num,path[t-1][3]+delta_theta*count/sp_num])
    final_point=copy.deepcopy(path[len(path)-1])
    final_point[0]=count_turn+1
    ans.append(final_point)
    return ans


if __name__ == '__main__':
    data = data_mix(scene='Roundabout')
    for Fid, frame in enumerate(data):
        for Cid, car in enumerate(frame):
            with open('data/Roundabout/car_data_mix.txt', 'a') as f:
                f.write(f"Frame: {Fid}, Car: {Cid}, Data: {car}\n")
    # hero_path, npcs_path = player_data_split(data)

    # import main
    # carla_path = main.HighwayPathToCarlaPath(npcs_path).exchange_to_town06()
    # player_path = main.HighwayPathToCarlaPath([hero_path]).exchange_to_town06()[0]
    # print(len(carla_path))
    