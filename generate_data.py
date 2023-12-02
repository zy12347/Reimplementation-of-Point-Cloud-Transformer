import numpy as np

def main():
    points = np.random.randint(0,100,size=(32,1024,3))
    
    label = np.random.randint(0,6,size=32)
    np.save("data/cls/train_data.npy",points)
    np.save("data/cls/train_labels.npy",label)
    #print(points)
    
def readnpy(path):
    x = np.load(path)
    print(x.shape)
    print(x)

main()
#readnpy('data/cls/train_data.npy')