import numpy as np
import os

def main():
    classes = ['bathtub','bed','chair','desk','dresser','monitor','night_stand','sofa','table','toilet']
    data = []
    label = []
    for i,c in enumerate(classes):
        dir_path = f'{c}/train'
        print(len(os.listdir(dir_path)))
        for name in os.listdir(dir_path):
            #print(name)
            file_path = f'{dir_path}/{name}'
            obj = []
            min_num = 1e6
            with open(file_path,'r') as file:
                lines = file.readlines()
                point_num = int(lines[1].split()[0])
                points = []
                print(point_num)
                for i in range(point_num):
                    pos = []
                    for x in lines[i+2].split():
                        pos.append(float(x))
                points.append(np.array(pos,dtype='float'))
            min_num = min(len(points),min_num)
            obj.append(np.array(points,dtype='float'))
            #print(min_num)
            #print(obj)
        break
if __name__=='__main__':
    main()