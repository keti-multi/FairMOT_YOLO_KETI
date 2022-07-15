import os
import numpy as np

# outputs = []

with open("/home/hjlee/FairMOT/src/cpp_value1.txt") as data:
    len = 1*1*152*272
    len1 = 1*4*152*272
    output = np.array([[float(i) for i in line.split()] for line in data.readlines()[:len]])
    outputs = np.array([[float(i) for i in line.split()] for line in data.readlines()[len:len+len1]])
    print(output)
    print(outputs)

                # print("outputs: ", outputs)
                # print("outputs[1] shape: ", outputs[1].shape)
                # # et = time.time()
                # # print("inference time: ", et-st)
                # # outputs = self.rknn.inference(inputs=[to_numpy(im_blob)], data_type='float32')
                # # outputs = self.rknn.inference(inputs=[to_numpy(im_blob)], data_type='uint8', data_format='nhwc')
                # # outputs = []

                # with open("/home/hjlee/FairMOT/src/cpp_value1.txt") as data:
                #     len = 1*1*152*272
                #     outputs[0] = np.array([[float(i) for i in line.split()] for line in data.readlines()[:len]])
                #     outputs[0] = outputs[0].reshape(1, 1, 152, 272)
                #     print("outputs[0] shape: ", outputs[0].shape)

                # with open("/home/hjlee/FairMOT/src/cpp_value1.txt") as data:
                #     len1 = 1*4*152*272
                #     outputs[1] = np.array([[float(i) for i in line.split()] for line in data.readlines()[len:len+len1]])
                #     print("outputs1: ", outputs[1])
                #     outputs[1] = outputs[1].reshape(1, 4, 152, 272)
                #     print("outputs[1] shape: ", outputs[1].shape)


                # with open("/home/hjlee/FairMOT/src/cpp_value1.txt") as data:
                #     len2 = 1*64*152*272
                #     outputs[2] = np.array([[float(i) for i in line.split()] for line in data.readlines()[len+len1:len+len1+len2]])
                #     outputs[2] = outputs[2].reshape(1, 64, 152, 272)
                #     print("outputs[2] shape: ", outputs[2].shape)

                # with open("/home/hjlee/FairMOT/src/cpp_value1.txt") as data:
                #     outputs[3] = np.array([[float(i) for i in line.split()] for line in data.readlines()[len+len1+len2:]])
                #     outputs[3] = outputs[3].reshape(1, 2, 152, 272)
                #     print("outputs[3] shape: ", outputs[3].shape)


                # outputs = np.array(outputs)
                # outputs = outputs.reshape(1, 3, 608, 1088)



                # f = open("/home/hjlee/FairMOT/src/cpp_value.txt")
                # lines = f.readlines()
                # lines = list(map(lambda s: s.strip(), lines))

                # len = 1*1*152*272
                # outputs.insert(0, float(lines[:len]))
                # outputs[0] = np.array(outputs[0])

                # len1 = 1*4*152*272
                # outputs.insert(1, float(lines[len:len+len1]))
                # outputs[1] = np.array(outputs[1])

                # len2 = 1*64*152*272
                # outputs.insert(2, float(lines[len+len1:len+len1+len2]))
                # outputs[2] = np.array(outputs[2])

                # len3 = 1*2*152*272
                # outputs.insert(3, float(lines[len+len1+len2:]))
                # outputs[3] = np.array(outputs[3])
                # outputs = np.array(outputs)

                # print(outputs.shape)
                # # print("outputs: ", outputs)
