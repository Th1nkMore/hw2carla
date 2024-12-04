import copy
import numpy as np
import main
SP_NUM = 1

def data_mix():
    data = np.load('car_data_mix.npy')
    data = data[:-300]
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
    data = data_mix()

    hero_path, npcs_path = player_data_split(data)

    carla_path = main.HighwayPathToCarlaPath(npcs_path).exchange_to_town06()
    player_path = main.HighwayPathToCarlaPath([hero_path]).exchange_to_town06()[0]

    