import numpy as np

pi_array = np.zeros((2,181))
rot_array = np.zeros(181)
for i in range(181):
    pi_array[0][i] = np.sin(np.pi*(90-i)/180)
    pi_array[1][i] = np.cos(np.pi*(90-i)/180)
    rot_array[i] = np.pi*(90-i)/180

def GetOffset():
    # offset = [
    #     [ 0.     , 0.      ,  0.     ],
    #     [ 1.36306, -1.79463,  0.83929],
    #     [ 2.44811, -6.72613,  0.     ],
    #     [ 2.5622 , -7.03959,  0.     ],
    #     [ 0.15764, -0.43311,  2.32255],
    #     [ 0.     , 0.      ,  0.     ],
    #     [-1.30552, -1.79463,  0.83929],
    #     [-2.54253, -6.98555,  0.     ],
    #     [-2.56826, -7.05623,  0.     ],
    #     [-0.16473, -0.45259,  2.36315],
    #     [ 0.     , 0.      ,  0.     ],
    #     [ 0.02827,  2.03559, -0.19338],
    #     [ 0.05672,  2.04885, -0.04275],
    #     [ 0.     , 0.      ,  0.     ],
    #     [-0.05417,  1.74624,  0.17202],
    #     [ 0.10407,  1.76136, -0.12397],
    #     [ 0.     , 0.      ,  0.     ],
    #     [ 3.36241,  1.20089, -0.31121],
    #     [ 4.983  , -0.     , -0.     ],
    #     [ 3.48356, -0.     , -0.     ],
    #     [ 0.71526, -0.     , -0.     ],
    #     [ 0.     , 0.      ,  0.     ],
    #     [-3.1366 ,  1.37405, -0.40465],
    #     [-5.2419 , -0.     , -0.     ],
    #     [-3.44417, -0.     , -0.     ],
    #     [-0.62253, -0.     , -0.     ],
    #     [ 0.     , 0.      ,  0.     ]
    #     ]
    # offset = [
    #     [ 0.     , 0.      ,  0.     ],
    #     [ 1.36306, -1.79463,  0.     ],
    #     [ 0.     , -6.72613,  0.     ],
    #     [ 0.     , -7.03959,  0.     ],
    #     [ 0.15764, -0.43311,  2.32255],
    #     [ 0.     , 0.      ,  0.     ],
    #     [-1.30552, -1.79463,  0.     ],
    #     [ 0.     , -6.98555,  0.     ],
    #     [ 0.     , -7.05623,  0.     ],
    #     [-0.16473, -0.45259,  2.36315],
    #     [ 0.     , 0.      ,  0.     ],
    #     [ 0.02827,  2.03559, -0.19338],
    #     [ 0.05672,  2.04885, -0.04275],
    #     [ 0.     , 0.      ,  0.     ],
    #     [-0.05417,  1.74624,  0.17202],
    #     [ 0.10407,  1.76136, -0.12397],
    #     [ 0.     , 0.      ,  0.     ],
    #     [ 3.36241,  1.20089, -0.31121],
    #     [ 4.983  , -0.     , -0.     ],
    #     [ 3.48356, -0.     , -0.     ],
    #     [ 0.71526, -0.     , -0.     ],
    #     [ 0.     , 0.      ,  0.     ],
    #     [-3.1366 ,  1.37405, -0.40465],
    #     [-5.2419 , -0.     , -0.     ],
    #     [-3.44417, -0.     , -0.     ],
    #     [-0.62253, -0.     , -0.     ],
    #     [ 0.     , 0.      ,  0.     ]
    #     ]
    offset = [
        [ 0.      ,  0.      ,  0.      ],
        [ 1.903060, -1.794630,  0.      ],
        [ 0.      , -5.985550,  0.      ],
        [ 0.      , -7.556230,  0.      ],
        [ 0.      ,  0.      ,  2.363150],
        [ 0.      ,  0.      ,  0.      ],
        [-1.903060, -1.794630,  0.      ],
        [ 0.      , -5.985550,  0.      ],
        [ 0.      , -7.556230,  0.      ],
        [ 0.      ,  0.      ,  2.363150],
        [ 0.      ,  0.      ,  0.      ],
        [ 0.      ,  2.035590,  0.      ],
        [ 0.      ,  2.048850,  0.      ],
        [ 0.      ,  0.      ,  0.      ],
        [ 0.      ,  1.746240,  0.      ],
        [ 0.      ,  1.761360,  0.      ],
        [ 0.      ,  0.      ,  0.      ],
        [ 2.136600,  1.20089 ,  0.      ],
        [ 3.983000,  0.      ,  0.      ],
        [ 3.983560,  0.      ,  0.      ],
        [ 0.      ,  0.      ,  0.      ],
        [-2.136600,  1.20089 ,  0.      ],
        [-3.983000,  0.      ,  0.      ],
        [-3.983560,  0.      ,  0.      ],
        [ 0.      ,  0.      ,  0.      ],
        ]
    return np.array(offset)

def GetData():
    #[一個前の関節,骨の始点,骨の終点]
    data = [
        [-1,-1,-1],
        [0,-1,0],
        [1,0,1],
        [2,1,2],
        [3,2,3],
        [4,-1,-1],
        [0,-1,4],
        [6,4,5],
        [7,5,6],
        [8,6,7],
        [9,-1,-1],
        [0,-1,8],
        [11,8,9],
        [12,-1,-1],
        [13,9,10],
        [14,10,11],
        [15,-1,-1],
        [13,9,12],
        [17,12,13],
        [18,13,14],
        [19,-1,-1],
        [13,9,15],
        [21,15,16],
        [22,16,17],
        [23,-1,-1]
    ]
    return np.array(data)

def CalcMat(x,y,z):
    # print(x,y,z)
    sx = np.sin(x)
    cx = np.cos(x)
    sy = np.sin(y)
    cy = np.cos(y)
    sz = np.sin(z)
    cz = np.cos(z)

    mat_tmp = np.zeros((3,3))

    mat_tmp[0][0] = cy*cz
    mat_tmp[0][1] = sx*sy*cz - cx*sz
    mat_tmp[0][2] = cx*sy*cz + sx*sz
    mat_tmp[1][0] = sz*cy
    mat_tmp[1][1] = sx*sy*sz + cx*cz
    mat_tmp[1][2] = cx*sy*sz - sx*cz
    mat_tmp[2][0] = -sy
    mat_tmp[2][1] = sx*cy
    mat_tmp[2][2] = cx*cy

    return mat_tmp

def NewCalcMat(x,y,z):
    # print(x,y,z)
    sx = np.sin(x)
    cx = np.cos(x)
    sy = np.sin(y)
    cy = np.cos(y)
    sz = np.sin(z)
    cz = np.cos(z)

    mat_tmp = np.zeros((3,3))

    mat_tmp[0][0] = cy*cz - sx*sy*sz
    mat_tmp[0][1] = - cx*sz
    mat_tmp[0][2] = sy*cz + sx*cy*sz
    mat_tmp[1][0] = cy*sz + sx*sy*cz
    mat_tmp[1][1] = cx*cz
    mat_tmp[1][2] = sy*sz - sx*cy*cz
    mat_tmp[2][0] = - cx*sy
    mat_tmp[2][1] = sx
    mat_tmp[2][2] = cx*cy

    return mat_tmp



#fourth
def CalcRotation(b,a):
    if b[0] == 0:
        if b[1] == 0:
            if b[2] > 0:
                #+z_only
                pattern = 5
            else:
                #error
                pattern = 0
                
        elif b[1] > 0:
            if b[2] == 0:
                #+y_only
                pattern = 3
            else:
                #error
                pattern = 0
                
        else:
            if b[2] == 0:
                #-y_only
                pattern = 4
            else:
                #error
                pattern = 0

    elif b[0] > 0:
        if b[1] == 0:
            if b[2] == 0:
                #+x_only
                pattern = 1
            else:
                #error
                pattern = 0
        else:
            if b[2] == 0:
                #+x,y
                pattern = 6
            else:
                #error
                pattern = 0

    else:
        if b[1] == 0:
            if b[2] == 0:
                #-x_only
                pattern = 2
            else:
                #error
                pattern = 0
        else:
            if b[2] == 0:
                #-x,y
                pattern = 7
            else:
                #error
                pattern = 0
    
    rotX = 0
    rotY = 0
    rotZ = 0

    x0,y0,z0 = b
    x1,y1,z1 = a
    if x1 == 0:
        x1 = 0.00001
    if y1 == 0:
        y1 = 0.00001
    if z1 == 0:
        z1 = 0.00001

    if pattern == 0:
        #error
        print(b)
        return Exception('Error!')
    
    elif pattern == 1:
        #+Xonly
        if x1 > 0:
            rotY = np.arcsin(-z1 / np.sqrt(x1**2+z1**2))
            rotZ = np.arcsin( y1 / np.sqrt(x1**2+y1**2))
        else:
            if z1 > 0:
                rotY = -np.pi - np.arcsin(-z1 / np.sqrt(x1**2+z1**2))
            else:
                rotY = np.pi - np.arcsin(-z1 / np.sqrt(x1**2+z1**2))
            rotZ = np.arcsin(-y1 / np.sqrt(x1**2+y1**2))

    elif pattern == 2:
        #-Xonly
        if x1 < 0:
            rotY = np.arcsin( z1 / np.sqrt(x1**2+z1**2))
            rotZ = np.arcsin(-y1 / np.sqrt(x1**2+y1**2))
        else:
            if z1 > 0:
                rotY = np.pi - np.arcsin( z1 / np.sqrt(x1**2+z1**2))
            else:
                rotY = -np.pi - np.arcsin( z1 / np.sqrt(x1**2+z1**2))
            rotZ = np.arcsin( y1 / np.sqrt(x1**2+y1**2))

    elif pattern == 3:
        #+Yonly
        if y1 > 0:
            rotX = np.arcsin( z1 / np.sqrt(y1**2 + z1**2))
            rotZ = np.arcsin(-x1 / np.sqrt(x1**2 + y1**2))
        else:
            if z1 > 0:
                rotX  = np.pi - np.arcsin( z1 / np.sqrt(y1**2 + z1**2))
            else:
                rotX  = -np.pi - np.arcsin( z1 / np.sqrt(y1**2 + z1**2))
            rotZ = np.arcsin( x1 / np.sqrt(x1**2 + y1**2))

    elif pattern == 4:
        #+Yonly
        if y1 < 0:
            rotX = np.arcsin(-z1 / np.sqrt(y1**2 + z1**2))
            rotZ = np.arcsin( x1 / np.sqrt(x1**2 + y1**2))
        else:
            if z1 > 0:
                rotX  = -np.pi - np.arcsin(-z1 / np.sqrt(y1**2 + z1**2))
            else:
                rotX  = np.pi - np.arcsin(-z1 / np.sqrt(y1**2 + z1**2))
            rotZ = np.arcsin(-x1 / np.sqrt(x1**2 + y1**2))
    
    elif pattern == 5:
        #+Zonly
        if z1 > 0:
            rotX = np.arcsin(-y1 / np.sqrt(y1**2 + z1**2))
            rotY = np.arcsin( x1 / np.sqrt(x1**2 + z1**2))
        else:
            if y1 > 0:
                rotX = -np.pi - np.arcsin(-y1 / np.sqrt(y1**2 + z1**2))
            else:
                rotX = np.pi - np.arcsin(-y1 / np.sqrt(y1**2 + z1**2))
            rotY = np.arcsin(-x1 / np.sqrt(x1**2 + z1**2))

    elif pattern == 6:
        #+X,Y
        if x1 > 0:
            rotY = np.arcsin(-z1 / np.sqrt(x1**2 + z1**2))
            rotZ = np.arcsin( y1 / np.sqrt(x1**2 + y1**2)) - np.arcsin( y0 / np.sqrt(((x0**2)*(x1**2))/(x1**2 + z1**2) + y0**2))
            # rotZ = np.arcsin( y1 / np.sqrt(x1**2 + y1**2)) - np.arcsin( y0 / np.sqrt(x1**2 + y0**2))
        else:
            if z1 > 0:
                rotY = -np.pi - np.arcsin(-z1 / np.sqrt(x1**2 + z1**2))
            else:
                rotY = np.pi - np.arcsin(-z1 / np.sqrt(x1**2 + z1**2))
            rotZ = -np.arcsin( y1 / np.sqrt(x1**2 + y1**2)) + np.arcsin( y0 / np.sqrt(((x0**2)*(x1**2))/(x1**2 + z1**2) + y0**2))
            # rotZ = -np.arcsin( y1 / np.sqrt(x1**2 + y1**2)) + np.arcsin( y0 / np.sqrt(x1**2 + y0**2))

    elif pattern == 7:
        #-X,Y
        if x1 < 0:
            rotY = np.arcsin( z1 / np.sqrt(x1**2 + z1**2))
            rotZ = np.arcsin(-y1 / np.sqrt(x1**2 + y1**2)) - np.arcsin(-y0 / np.sqrt(((x0**2)*(x1**2))/(x1**2 + z1**2) + y0**2))
        else:
            if z1 > 0:
                rotY = np.pi - np.arcsin( z1 / np.sqrt(x1**2 + z1**2))
            else:
                rotY = -np.pi - np.arcsin( z1 / np.sqrt(x1**2 + z1**2))
            rotZ = -np.arcsin(-y1 / np.sqrt(x1**2 + y1**2)) + np.arcsin(-y0 / np.sqrt(((x0**2)*(x1**2))/(x1**2 + z1**2) + y0**2))
    return rotX,rotY,rotZ





#third#########################################################################################3
# def NewCalcRotation(b,a):
#     if b[0] == 0:
#         if b[1] == 0:
#             if b[2] > 0:
#                 #+z_only
#                 pattern = 5
#             else:
#                 #error
#                 pattern = 0
                
#         elif b[1] > 0:
#             if b[2] == 0:
#                 #+y_only
#                 pattern = 3
#             else:
#                 #error
#                 pattern = 0
                
#         else:
#             if b[2] == 0:
#                 #-y_only
#                 pattern = 4
#             else:
#                 #error
#                 pattern = 0

#     elif b[0] > 0:
#         if b[1] == 0:
#             if b[2] == 0:
#                 #+x_only
#                 pattern = 1
#             else:
#                 #error
#                 pattern = 0
#         else:
#             if b[2] == 0:
#                 #+x,y
#                 pattern = 6
#             else:
#                 #error
#                 pattern = 0

#     else:
#         if b[1] == 0:
#             if b[2] == 0:
#                 #-x_only
#                 pattern = 2
#             else:
#                 #error
#                 pattern = 0
#         else:
#             if b[2] == 0:
#                 #-x,y
#                 pattern = 7
#             else:
#                 #error
#                 pattern = 0
    
#     rotX = 0
#     rotY = 0
#     rotZ = 0

#     x0,y0,z0 = b
#     x1,y1,z1 = a
#     if x1 == 0:
#         x1 = 0.00001
#     if y1 == 0:
#         y1 = 0.00001
#     if z1 == 0:
#         z1 = 0.00001

#     if pattern == 0:
#         #error
#         print(b)
#         return Exception('Error!')
    
#     elif pattern == 1:
#         #+x_only
#         rotX = 0
#         if x1 > 0:
#             rotY = np.arcsin(-z1 / np.sqrt(x1**2 + z1**2))
#         else:
#             if z1 > 0:
#                 rotY = -np.pi - np.arcsin(-z1 / np.sqrt(x1**2 + z1**2))
#             else:
#                 rotY = np.pi - np.arcsin(-z1 / np.sqrt(x1**2 + z1**2))
#         rotZ = np.arcsin(y1)

#     elif pattern == 2:
#         #-x_only
#         rotX = 0
#         if x1 < 0:
#             rotY = np.arcsin(z1 / np.sqrt(x1**2 + z1**2))
#         else:
#             if z1 > 0:
#                 rotY = np.pi - np.arcsin(z1 / np.sqrt(x1**2 + z1**2))
#             else:
#                 rotY = -np.pi - np.arcsin(z1 / np.sqrt(x1**2 + z1**2))
#         rotZ = np.arcsin(-y1)

#     elif pattern == 3:
#         #+y_only
#         rotX = 0
#         if z1 > 0:
#             rotY = np.arcsin(x1 / np.sqrt(x1**2 + z1**2))
#         else:
#             rotY = np.arcsin(-x1 / np.sqrt(x1**2 + z1**2))
#         if y1 > 0:
#             rotX = np.arcsin(np.sqrt(x1**2 + z1**2))
#         else:
#             if z1 > 0:
#                 rotX = np.pi - np.arcsin(np.sqrt(x1**2 + z1**2))
#             else:
#                 rotX = -np.pi - np.arcsin(np.sqrt(x1**2 + z1**2))
#         rotZ = 0

#     elif pattern == 4:
#         #-y_only
#         rotX = 0
#         if z1 > 0:
#             rotY = np.arcsin(x1 / np.sqrt(x1**2 + z1**2))
#         else:
#             rotY = np.arcsin(-x1 / np.sqrt(x1**2 + z1**2))
#         if y1 < 0:
#             rotX = np.arcsin(-np.sqrt(x1**2 + z1**2))
#         else:
#             if z1 > 0:
#                 rotX = -np.pi - np.arcsin(-np.sqrt(x1**2 + z1**2))
#             else:
#                 rotX = np.pi - np.arcsin(-np.sqrt(x1**2 + z1**2))
#         rotZ = 0
    
#     elif pattern == 5:
#         #+z_only
#         rotX = 0
#         if z1 > 0:
#             rotY = np.arcsin(x1 / np.sqrt(x1**2 + z1**2))
#         else:
#             if x1 > 0:
#                 rotY = np.pi - np.arcsin(x1 / np.sqrt(x1**2 + z1**2))
#             else:
#                 rotY = -np.pi - np.arcsin(x1 / np.sqrt(x1**2 + z1**2))
#         rotZ = np.arcsin(-y1)

#     elif pattern == 6:
#         #+x,y
#         rotX = 0
#         if x1 > 0:
#             rotY = np.arcsin(-z1 / np.sqrt(x1**2 + z1**2))
#         else:
#             if z1 > 0:
#                 rotY = -np.pi - np.arcsin(-z1 / np.sqrt(x1**2 + z1**2))
#             else:
#                 rotY = np.pi - np.arcsin(-z1 / np.sqrt(x1**2 + z1**2))
#         rotZ = np.arcsin(y1) - np.arcsin(y0)

#     elif pattern == 7:
#         #+x,y
#         rotX = 0
#         if x1 < 0:
#             rotY = np.arcsin(z1 / np.sqrt(x1**2 + z1**2))
#         else:
#             if z1 > 0:
#                 rotY = np.pi - np.arcsin(z1 / np.sqrt(x1**2 + z1**2))
#             else:
#                 rotY = -np.pi - np.arcsin(z1 / np.sqrt(x1**2 + z1**2))
#         rotZ = np.arcsin(-y1) - np.arcsin(-y0)

#     else:
#         print(pattern)
#         return Exception('Error!')
#     print(pattern)
#     print(b)
#     print(a)
#     print(rotX, rotY, rotZ)
#     return rotX, rotY, rotZ

            
    







##Second#######################################################################################################################################################################
# def CalcRotationX(x,y):
#     a = np.zeros(181)
#     b = np.zeros(181)
#     x_tmp = rot_array
#     y_tmp = np.zeros(181)
#     z_tmp = np.zeros(181)
#     norm_tmp = np.zeros(181)
#     a = x[1]*pi_array[0] + x[2]*pi_array[1]
#     b = -x[1]*pi_array[1] + x[2]*pi_array[0]
#     print(a)
#     if x[0] > 0:
#         y_tmp = np.arcsin(-y[2]/np.sqrt(x[0]**2 + a**2)) + (a/np.sqrt(x[0]**2 + a**2))
#     else:
#         y_tmp = np.arcsin(-y[2]/np.sqrt(x[0]**2 + a**2)) - (a/np.sqrt(x[0]**2 + a**2)) + np.pi
    
#     if y[0] > 0:
#         z_tmp = np.arcsin(b/np.sqrt(y[0]**2 + y[1]**2)) + np.arcsin(y[1]/np.sqrt(y[0]**2 + y[1]**2))
#     else:
#         z_tmp = np.arcsin(b/np.sqrt(y[0]**2 + y[1]**2)) - np.arcsin(y[1]/np.sqrt(y[0]**2 + y[1]**2)) + np.pi

#     print(y_tmp)
#     for i in range(181):
#         if y_tmp[i] > np.pi:
#             y_tmp[i] -= 2*np.pi
#         elif y_tmp[i] < - np.pi:
#             y_tmp[i] += 2*np.pi
#         if z_tmp[i] > np.pi:
#             z_tmp[i] -= 2*np.pi
#         elif z_tmp[i] < - np.pi:
#             z_tmp[i] += 2*np.pi
#         if np.isnan(y_tmp[i]):
#             y_tmp[i] = 1000
#         if np.isnan(z_tmp[i]):
#             z_tmp[i] = 1000

#     norm_tmp = x_tmp**4 + y_tmp**4 + z_tmp**4

#     argmax_tmp = np.argmin(norm_tmp)

#     print("X")
#     for i in range(len(norm_tmp)):
#         print(norm_tmp[i],x_tmp[i],y_tmp[i],z_tmp[i])
#     print("X")
        

#     return min(norm_tmp),x_tmp[argmax_tmp],y_tmp[argmax_tmp],z_tmp[argmax_tmp]



# def CalcRotationY(x,y):
#     x_tmp = np.zeros(181)
#     y_tmp = rot_array
#     z_tmp = np.zeros(181)
#     norm_tmp = np.zeros(181)
    
#     if x[1] > 0:
#         x_tmp = np.arcsin((y[2]+x[0]*pi_array[0]) / (abs(pi_array[1])*np.sqrt(x[1]**2+x[2]**2))) - np.arcsin((x[2]*pi_array[1]) / (abs(pi_array[1])*np.sqrt(x[1]**2+x[2]**2)))
#     else:
#         x_tmp = np.arcsin((y[2]+x[0]*pi_array[0]) / (abs(pi_array[1])*np.sqrt(x[1]**2+x[2]**2))) + np.arcsin((x[2]*pi_array[1]) / (abs(pi_array[1])*np.sqrt(x[1]**2+x[2]**2))) + np.pi
    
#     if y[0] > 0:
#         z_tmp = np.arcsin((-x[1]*np.cos(x_tmp)+x[2]*np.sin(x_tmp)) / (np.sqrt(y[0]**2+y[1]**2))) + np.arcsin(y[1] / np.sqrt(y[0]**2+y[1]**2))
#     else:
#         z_tmp = np.arcsin((-x[1]*np.cos(x_tmp)+x[2]*np.sin(x_tmp)) / (np.sqrt(y[0]**2+y[1]**2))) - np.arcsin(y[1] / np.sqrt(y[0]**2+y[1]**2)) + np.pi

#     for i in range(181):
#         if x_tmp[i] > np.pi:
#             x_tmp[i] -= 2*np.pi
#         elif x_tmp[i] < - np.pi:
#             x_tmp[i] += 2*np.pi
#         if z_tmp[i] > np.pi:
#             z_tmp[i] -= 2*np.pi
#         elif z_tmp[i] < - np.pi:
#             z_tmp[i] += 2*np.pi
#         if np.isnan(x_tmp[i]):
#             x_tmp[i] = 1000
#         if np.isnan(z_tmp[i]):
#             z_tmp[i] = 1000
#     norm_tmp = x_tmp**4 + y_tmp**4 + z_tmp**4

#     argmax_tmp = np.argmin(norm_tmp)

#     print("Y")
#     for i in range(len(norm_tmp)):
#         print(norm_tmp[i],x_tmp[i],y_tmp[i],z_tmp[i])
#     print("Y")
        

#     return min(norm_tmp),x_tmp[argmax_tmp],y_tmp[argmax_tmp],z_tmp[argmax_tmp]



# def CalcRotationZ(x,y):
#     x_tmp = np.zeros(181)
#     y_tmp = np.zeros(181)
#     z_tmp = rot_array
#     norm_tmp = np.zeros(181)
    
#     if x[2] > 0:
#         x_tmp = np.arcsin((y[0]*pi_array[0]+y[1]*pi_array[1]) / np.sqrt(x[1]**2+x[2]**2)) + np.arcsin((x[1]) / np.sqrt(x[1]**2+x[2]**2))
#     else:
#         x_tmp = np.arcsin((y[0]*pi_array[0]+y[1]*pi_array[1]) / np.sqrt(x[1]**2+x[2]**2)) - np.arcsin((x[1]) / np.sqrt(x[1]**2+x[2]**2)) + np.pi
    
#     a = x[1]*np.sin(x_tmp) + x[2]*np.cos(x_tmp) 

#     if x[0] > 0:
#         y_tmp = np.arcsin(-y[2] / np.sqrt(x[0]**2+a**2)) + np.arcsin(a / np.sqrt(x[0]**2+a**2))
#     else:
#         y_tmp = np.arcsin(-y[2] / np.sqrt(x[0]**2+a**2)) + np.arcsin(a / np.sqrt(x[0]**2+a**2)) + np.pi


#     for i in range(181):
#         if x_tmp[i] > np.pi:
#             x_tmp[i] -= 2*np.pi
#         elif x_tmp[i] < - np.pi:
#             x_tmp[i] += 2*np.pi
#         if y_tmp[i] > np.pi:
#             y_tmp[i] -= 2*np.pi
#         elif y_tmp[i] < - np.pi:
#             y_tmp[i] += 2*np.pi
#         if np.isnan(x_tmp[i]):
#             x_tmp[i] = 1000
#         if np.isnan(y_tmp[i]):
#             y_tmp[i] = 1000
#     norm_tmp = x_tmp**4 + y_tmp**4 + z_tmp**4

#     argmax_tmp = np.argmin(norm_tmp)

#     print("Z")
#     for i in range(len(norm_tmp)):
#         print(norm_tmp[i],x_tmp[i],y_tmp[i],z_tmp[i])
#     print("Z")

#     return min(norm_tmp),x_tmp[argmax_tmp],y_tmp[argmax_tmp],z_tmp[argmax_tmp]



# def CalcRotation(x,y):
#     norm_X, x_X, y_X, z_X = CalcRotationX(x,y)
#     norm_Y, x_Y, y_Y, z_Y = CalcRotationY(x,y)
#     norm_Z, x_Z, y_Z, z_Z = CalcRotationZ(x,y)
#     if norm_X <= norm_Y and norm_X <= norm_Z:
#         return x_X,y_X,z_X
#     elif norm_Y <= norm_X and norm_Y <= norm_Z:
#         return x_Y,y_Y,z_Y
#     else:
#         return x_Z,y_Z,z_Z




# ##First#######################################################################################################################################################################
def CalcRotation_OLD(before_vector,after_vector):
    rotate_list = []
    rotate_l2 = []

    if abs(np.sqrt(before_vector[1]**2 + before_vector[2]**2)) > abs(after_vector[2]) and abs(np.sqrt(after_vector[0]**2 + after_vector[1]**2)) >abs(before_vector[0]):

        if before_vector[1] > 0:
            rotate_X = np.arcsin(after_vector[2]/np.sqrt(before_vector[1]**2 + before_vector[2]**2)) - np.arcsin(before_vector[2]/np.sqrt(before_vector[1]**2 + before_vector[2]**2))
        else:
            rotate_X = np.arcsin(after_vector[2]/np.sqrt(before_vector[1]**2 + before_vector[2]**2)) + np.arcsin(before_vector[2]/np.sqrt(before_vector[1]**2 + before_vector[2]**2))
            if rotate_X > 0:
                rotate_X -= np.pi
            else:
                rotate_X += np.pi
        
        rotate_Y = 0

        if after_vector[1] > 0:
            rotate_Z = np.arcsin(before_vector[0]/np.sqrt(after_vector[0]**2 + after_vector[1]**2)) - np.arcsin(after_vector[0]/np.sqrt(after_vector[0]**2 + after_vector[1]**2))
        else:
            rotate_Z = np.arcsin(before_vector[0]/np.sqrt(after_vector[0]**2 + after_vector[1]**2)) + np.arcsin(after_vector[0]/np.sqrt(after_vector[0]**2 + after_vector[1]**2))
            if rotate_Z > 0:
                rotate_Z -= np.pi
            else:
                rotate_Z += np.pi
        
        rotate_list.append([rotate_X,rotate_Y,rotate_Z])
        l2_tmp = np.degrees(rotate_X)**2 + np.degrees(rotate_Y)**2 + np.degrees(rotate_Z)**2
        rotate_l2.append(l2_tmp)


    if abs(np.sqrt(before_vector[0]**2 + before_vector[2]**2)) > abs(after_vector[2]) and abs(np.sqrt(after_vector[0]**2 + after_vector[1]**2)) >abs(before_vector[1]):

        rotate_X = 0

        if before_vector[0] > 0:
            rotate_Y = np.arcsin(-after_vector[2]/np.sqrt(before_vector[0]**2 + before_vector[2]**2)) + np.arcsin(before_vector[2]/np.sqrt(before_vector[0]**2 + before_vector[2]**2))
        else:
            rotate_Y = np.arcsin(-after_vector[2]/np.sqrt(before_vector[0]**2 + before_vector[2]**2)) - np.arcsin(before_vector[2]/np.sqrt(before_vector[0]**2 + before_vector[2]**2))
            if rotate_Y > 0:
                rotate_Y -= np.pi
            else:
                rotate_Y += np.pi
        
        if after_vector[0] > 0:
            rotate_Z = np.arcsin(-before_vector[1]/np.sqrt(after_vector[0]**2 + after_vector[1]**2)) + np.arcsin(after_vector[1]/np.sqrt(after_vector[0]**2 + after_vector[1]**2))
        else:
            rotate_Z = np.arcsin(-before_vector[1]/np.sqrt(after_vector[0]**2 + after_vector[1]**2)) - np.arcsin(after_vector[1]/np.sqrt(after_vector[0]**2 + after_vector[1]**2))
            if rotate_Z > 0:
                rotate_Z -= np.pi
            else:
                rotate_Z += np.pi
        
        rotate_list.append([rotate_X,rotate_Y,rotate_Z])
        l2_tmp = np.degrees(rotate_X)**2 + np.degrees(rotate_Y)**2 + np.degrees(rotate_Z)**2
        rotate_l2.append(l2_tmp)


    if abs(np.sqrt(before_vector[1]**2 + before_vector[2]**2)) > abs(after_vector[1]) and abs(np.sqrt(after_vector[0]**2 + after_vector[2]**2)) > abs(before_vector[0]):

        if before_vector[2] > 0:
            rotate_X = np.arcsin(-after_vector[1]/np.sqrt(before_vector[1]**2 + before_vector[2]**2)) + np.arcsin(before_vector[1]/np.sqrt(before_vector[1]**2 + before_vector[2]**2))
        else:
            rotate_X = np.arcsin(-after_vector[1]/np.sqrt(before_vector[1]**2 + before_vector[2]**2)) - np.arcsin(before_vector[1]/np.sqrt(before_vector[1]**2 + before_vector[2]**2))
            if rotate_X > 0:
                rotate_X -= np.pi
            else:
                rotate_X += np.pi

        if after_vector[2] > 0:
            rotate_Y = np.arcsin(-before_vector[0]/np.sqrt(after_vector[0]**2 + after_vector[2]**2)) + np.arcsin(after_vector[0]/np.sqrt(after_vector[0]**2 + after_vector[2]**2))
        else:
            rotate_Y = np.arcsin(-before_vector[0]/np.sqrt(after_vector[0]**2 + after_vector[2]**2)) - np.arcsin(after_vector[0]/np.sqrt(after_vector[0]**2 + after_vector[2]**2))
            if rotate_Y > 0:
                rotate_Y -= np.pi
            else:
                rotate_Y += np.pi

        rotate_Z = 0

        rotate_list.append([rotate_X,rotate_Y,rotate_Z])
        l2_tmp = np.degrees(rotate_X)**2 + np.degrees(rotate_Y)**2 + np.degrees(rotate_Z)**2
        rotate_l2.append(l2_tmp)


    if abs(np.sqrt(before_vector[0]**2 + before_vector[1]**2)) > abs(after_vector[2]) and abs(np.sqrt(after_vector[0]**2 + after_vector[1]**2)) > abs(after_vector[2]):

        rotate_X = 0.5 * np.pi

        if before_vector[0] > 0:
            rotate_Y = np.arcsin(-after_vector[2]/np.sqrt(before_vector[0]**2 + before_vector[1]**2)) + np.arcsin(before_vector[1]/np.sqrt(before_vector[0]**2 + before_vector[1]**2))
        else:
            rotate_Y = np.arcsin(-after_vector[2]/np.sqrt(before_vector[0]**2 + before_vector[1]**2)) - np.arcsin(before_vector[1]/np.sqrt(before_vector[0]**2 + before_vector[1]**2))
            if rotate_Y > 0:
                rotate_Y -= np.pi
            else:
                rotate_Y += np.pi

        if after_vector[0] > 0:
            rotate_Z = np.arcsin(before_vector[2]/np.sqrt(after_vector[0]**2 + after_vector[1]**2)) + np.arcsin(after_vector[1]/np.sqrt(after_vector[0]**2 + after_vector[1]**2))
        else:
            rotate_Z = np.arcsin(before_vector[2]/np.sqrt(after_vector[0]**2 + after_vector[1]**2)) - np.arcsin(after_vector[1]/np.sqrt(after_vector[0]**2 + after_vector[1]**2))
            if rotate_Z > 0:
                rotate_Z -= np.pi
            else:
                rotate_Z += np.pi

        rotate_list.append([rotate_X,rotate_Y,rotate_Z])
        l2_tmp = np.degrees(rotate_X)**2 + np.degrees(rotate_Y)**2 + np.degrees(rotate_Z)**2
        rotate_l2.append(l2_tmp)


        rotate_X = -0.5 * np.pi

        if before_vector[0] > 0:
            rotate_Y = np.arcsin(-after_vector[2]/np.sqrt(before_vector[0]**2 + before_vector[1]**2)) - np.arcsin(before_vector[1]/np.sqrt(before_vector[0]**2 + before_vector[1]**2))
        else:
            rotate_Y = np.arcsin(-after_vector[2]/np.sqrt(before_vector[0]**2 + before_vector[1]**2)) + np.arcsin(before_vector[1]/np.sqrt(before_vector[0]**2 + before_vector[1]**2))
            if rotate_Y > 0:
                rotate_Y -= np.pi
            else:
                rotate_Y += np.pi

        if after_vector[0] > 0:
            rotate_Z = np.arcsin(-before_vector[2]/np.sqrt(after_vector[0]**2 + after_vector[1]**2)) + np.arcsin(after_vector[1]/np.sqrt(after_vector[0]**2 + after_vector[1]**2))
        else:
            rotate_Z = np.arcsin(-before_vector[2]/np.sqrt(after_vector[0]**2 + after_vector[1]**2)) - np.arcsin(after_vector[1]/np.sqrt(after_vector[0]**2 + after_vector[1]**2))
            if rotate_Z > 0:
                rotate_Z -= np.pi
            else:
                rotate_Z += np.pi

        rotate_list.append([rotate_X,rotate_Y,rotate_Z])
        l2_tmp = np.degrees(rotate_X)**2 + np.degrees(rotate_Y)**2 + np.degrees(rotate_Z)**2
        rotate_l2.append(l2_tmp)


    if abs(np.sqrt(before_vector[1]**2 + before_vector[2]**2)) > abs(after_vector[0]) and abs(np.sqrt(after_vector[1]**2 + after_vector[2]**2)) > abs(before_vector[0]):
        if before_vector[2] > 0:
            rotate_X = np.arcsin(after_vector[0]/np.sqrt(before_vector[1]**2 + before_vector[2]**2)) + np.arcsin(before_vector[1]/np.sqrt(before_vector[1]**2 + before_vector[2]**2))
        else:
            rotate_X = np.arcsin(after_vector[0]/np.sqrt(before_vector[1]**2 + before_vector[2]**2)) - np.arcsin(before_vector[1]/np.sqrt(before_vector[1]**2 + before_vector[2]**2))
            if rotate_X > 0:
                rotate_X -= np.pi
            else:
                rotate_X += np.pi

        if after_vector[2] > 0:
            rotate_Y = np.arcsin(-before_vector[0]/np.sqrt(after_vector[1]**2 + after_vector[2]**2)) + np.arcsin(after_vector[1]/np.sqrt(after_vector[1]**2 + after_vector[2]**2))
        else:
            rotate_Y = np.arcsin(-before_vector[0]/np.sqrt(after_vector[1]**2 + after_vector[2]**2)) - np.arcsin(after_vector[1]/np.sqrt(after_vector[1]**2 + after_vector[2]**2))
            if rotate_Y > 0:
                rotate_Y -= np.pi
            else:
                rotate_Y += np.pi

        rotate_Z = 0.5 *np.pi

        rotate_list.append([rotate_X,rotate_Y,rotate_Z])
        l2_tmp = np.degrees(rotate_X)**2 + np.degrees(rotate_Y)**2 + np.degrees(rotate_Z)**2
        rotate_l2.append(l2_tmp)


        if before_vector[2] > 0:
            rotate_X = np.arcsin(-after_vector[0]/np.sqrt(before_vector[1]**2 + before_vector[2]**2)) + np.arcsin(before_vector[1]/np.sqrt(before_vector[1]**2 + before_vector[2]**2))
        else:
            rotate_X = np.arcsin(-after_vector[0]/np.sqrt(before_vector[1]**2 + before_vector[2]**2)) - np.arcsin(before_vector[1]/np.sqrt(before_vector[1]**2 + before_vector[2]**2))
            if rotate_X > 0:
                rotate_X -= np.pi
            else:
                rotate_X += np.pi

        if after_vector[2] > 0:
            rotate_Y = np.arcsin(-before_vector[0]/np.sqrt(after_vector[1]**2 + after_vector[2]**2)) - np.arcsin(after_vector[1]/np.sqrt(after_vector[1]**2 + after_vector[2]**2))
        else:
            rotate_Y = np.arcsin(-before_vector[0]/np.sqrt(after_vector[1]**2 + after_vector[2]**2)) + np.arcsin(after_vector[1]/np.sqrt(after_vector[1]**2 + after_vector[2]**2))
            if rotate_Y > 0:
                rotate_Y -= np.pi
            else:
                rotate_Y += np.pi

        rotate_Z = -0.5 * np.pi

        rotate_list.append([rotate_X,rotate_Y,rotate_Z])
        l2_tmp = np.degrees(rotate_X)**2 + np.degrees(rotate_Y)**2 + np.degrees(rotate_Z)**2
        rotate_l2.append(l2_tmp)

    rotate_X,rotate_Y,rotate_Z = rotate_list[np.argmin(np.array(rotate_l2))]
    print(rotate_l2)

    return rotate_X,rotate_Y,rotate_Z