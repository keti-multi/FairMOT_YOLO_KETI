from rknn.api import RKNN

rknn = RKNN()

print('--> Config model')
rknn.config(#mean_values=[[0, 0, 0]], # [0.408, 0.447, 0.470] [104.04, 113.985, 119.85]
                #std_values=[[255, 255, 255]], # [0.289, 0.274, 0.278] [73.695, 69.87, 70.89]
                # reorder_channel='0 1 2',
                target_platform = 'rk1808',
                optimization_level=3,
                quantized_dtype="dynamic_fixed_point-i16", # asymmetric_quantized-u8 dynamic_fixed_point-i8 dynamic_fixed_point-i16
                quantized_algorithm="normal")  # normal mmse kl_divergence

# Load ONNX model
print('--> Loading model')
load_onnx = "yolov5.onnx"
ret = rknn.load_onnx(model=load_onnx)#, inputs=['modelInput'], input_size_list=[[3, 608, 1088]], outputs=['hm', 'wh', 'id', 'reg'])
if ret != 0:
    print('Load model failed!')
    exit(ret)
print('done')

# Build model
print('--> Building model')
# ret = self.rknn.build(do_quantization=False)
ret = rknn.build(do_quantization=True, dataset='/home/hjlee/FairMOT/Dataset/dataset.txt')
if ret != 0:
    print('Build model failed!')
    exit(ret)
print('done')

# Export RKNN model
print('--> Export RKNN model')
ret = rknn.export_rknn("yolov5.rknn")
if ret != 0:
    print('Export rknn failed!')
    exit(ret)
print('done')

rknn.release()