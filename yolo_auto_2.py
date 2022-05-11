import numpy as np
import cv2
import glob
import os
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN
import math
from PIL import Image
import keras_yolo3_master.yolo

def detection(images, yolo):
    img, point, score = yolo.detect_image(images)

    return img, point, score
    
def run_1(file, yolo):
    block = [0, 1, 0, 1]
    images = Image.open(file)
    img, point, score = detection(images, yolo)
    return img, point, score, block

def run_2(file, yolo):
    hmx = 1
    h = 0
    found_mx = False
    block_mx = [0, 1, 0, 1]
    img = np.array(Image.open(file))
    img_mx = img
    point_mx = []
    height, width = img.shape[:2]
    df = pd.read_csv('divide.csv')
    case = int(len(df.columns) / 4)
    for n in range(case):
        df_tmp = df.iloc[:,n*4:(n+1)*4].dropna(how='any')
        div = len(df_tmp)
        for m in range(div):
            block = [df_tmp.iloc[m,0], df_tmp.iloc[m,1], df_tmp.iloc[m,2], df_tmp.iloc[m,3]]
            img_tmp = img[int(height*block[0]):int(height*block[1]),int(width*block[2]):int(width*block[3])]
            img_tmp = Image.fromarray(img_tmp)
            img_tmp = img_tmp.resize((width, height))
            image, point = detection(img_tmp, yolo)
            if len(point) > len(point_mx):
                img_mx = image
                point_mx = point
                block_mx = block_mx
        if found_mx:
            break
    if len(point_mx)==3:
        add = np.array([[block_mx[2]*width, block_mx[0]*height, block_mx[2]*width, block_mx[0]*height]] * len(point_mx))
        point_mx = point_mx + add
    return img_mx, point_mx, block

#水位設定関数
def makeLinearEquation(x1, y1, x2, y2):
    a_1 = (y1 - y2) / (x1 - x2)
    b_1 = y1 - (a_1 * x1)
    return a_1,b_1

def predict_wl(point, images_w, buffer, interval):
    xy1 = [0,0]
    xy2 = [0,0]
    xy3 = [0,0]
    xy1[0] = (point[0][1]+point[0][3])/2
    xy1[1] = (point[0][0]+point[0][2])/2
    xy2[0] = (point[1][1]+point[1][3])/2
    xy2[1] = (point[1][0]+point[1][2])/2
    xy3[0] = (point[2][1]+point[2][3])/2
    xy3[1] = (point[2][0]+point[2][2])/2
    y_max = min(xy1[1],xy2[1],xy3[1])
    for xy in [xy1,xy2,xy3]:
        if int(xy[1]) == int(y_max):
            xy_u = xy
    x1 = (xy1[0] + xy2[0] + xy3[0]) / 3
    y1 = (xy1[1] + xy2[1] + xy3[1]) / 3
    x2 = xy_u[0]
    y2 = xy_u[1]
    a_1, b_1 = makeLinearEquation(x1, y1, x2, y2) #a_1：傾き、b_1：切片
    #xy座標リスト
    xy_list=[]
    for y in range(images_w.shape[0]):   #y:0～height
        if y:
            x=(y-b_1)/a_1   #0～719のときのx座標
            x=int(Decimal(str(x)).quantize(Decimal('0'), rounding=ROUND_HALF_UP))  #ピクセルは整数のため、小数点以下四捨五入
            x_y_list=str(x)+","+str(y)   #グラフ上の(x,y)座標
            if x >= 0 and x < images_w.shape[1]:
                xy_list.append(x_y_list)     #リストに追加
    #水面位置の座標リスト
    x_water=[]
    y_water=[]
    for lenxy in range(len(xy_list)):    #グラフ上のxy座標で[0,255,255,255]があるか確認
        len_x=int(xy_list[lenxy].split(",")[0]) #x座標
        len_y=int(xy_list[lenxy].split(",")[1]) #y座標
        pixcelValue1=images_w[len_y,len_x]
        if pixcelValue1[3]==255:         #グラフ上のxy座標で[0,255,255,255]があれば以下の処理
            #print(len_x,len_y)           #x,yの最小値取得(最小値＝水面境界)
            x_water.append(len_x)
            y_water.append(len_y)
    
    if len(y_water)!=0:  #グラフ上に予測水面がある場合
        line1 = math.sqrt(pow(x1-x_water[0],2)+pow(y1-y_water[0],2))   #(x1,y1)から水面までの距離(座標値)
        line2 = math.sqrt(pow(x1-x2,2)+pow(y1-y2,2))   #(x1,y1)から(x2,y2)までの距離(座標値)
        water_line = round(line1 / line2 * interval / 100, 2)   #(x1,y1)から水面までの距離
        WL = 9.995 - buffer - water_line  #水位（設置箇所上部の標高基準）
        print(WL)
    
    if len(y_water)==0:   #グラフ上に予測水面がない場合
        WL = -999
        print(WL)
    return WL

#####MAIN#######
yolo = keras_yolo3_master.yolo.YOLO()
#file_path
area_path = "./output/"
in_path = "./pic_mask_1/"
out_path = "./pic_out_1/"

#param
interval = 0.1333 #中点から下点までの距離(m)
buffer = 0.1 + 0.2/2 + 0.0667  #(m)

#202112150730の結果
point_base = [[181.69174,312.31085,228.50461,388.24393],[188.44395,427.99658,241.20738,496.41882],[238.8278,366.1533,293.10962,442.10477]]
for i in range(len(point_base)):
    point_base[i][0] = int(point_base[i][0] / 5) + 387
    point_base[i][1] = int(point_base[i][0] / 5) + 1150
    point_base[i][2] = int(point_base[i][0] / 5) + 387
    point_base[i][3] = int(point_base[i][0] / 5) + 1150

H_list = []
flist = glob.glob(in_path + "*.jpg")
for file in flist:
    #量水標検知
    score = 0
    img, point, score, block = run_1(file, yolo)
    #if len(point) != 3:
    #    img, point, block = run_2(file, yolo)
    #    if len(point) != 3:
    #        print("NOT FOUND")
    #if len(point) == 3:
    height, width = cv2.imread(file).shape[:2]
        #水位算定
    img_w = Image.open(area_path + os.path.splitext(os.path.basename(file))[0] + '.png')
    img_w = np.array(img_w.resize((width, height)))
    H = predict_wl(point_base, img_w, buffer, interval)
    H_list.append([os.path.basename(file), H, len(point), score, point])
    img.save(out_path + os.path.basename(file))
    #else:
    #    H_list.append([os.path.basename(file), "miss", len(point), score, point])
    #    if len(point) > 0:
    #        img.save(out_path + os.path.basename(file))
df = pd.DataFrame(H_list, columns=['time', 'H', 'point', 'score', 'point'])
df.to_csv('./H_series.csv')

