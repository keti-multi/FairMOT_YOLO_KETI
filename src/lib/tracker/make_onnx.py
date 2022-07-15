import torch
import onnxruntime as ort



model = create_model(opt.arch, opt.heads, opt.head_conv)
model = load_model(self.model, opt.load_model)
model = self.model.to(opt.device)
model.eval()

def Convert_ONNX(self, model): 
    # set the model to inference mode 
    model.eval() 

    # Let's create a dummy input tensor  
    # dummy_input = torch.randn(1, 3, 608, 1088)
    # dummy_input = torch.randn(1, 3, 480, 864)
    # dummy_input = torch.randn(1, 3, 320, 576)
    # dummy_input = torch.randn(1, 3, 160, 288)
    w = self.opt.img_size[0]
    h = self.opt.img_size[1]
    dummy_input = torch.randn(1, 3, h, w)

    # Export the model   
    torch.onnx.export(model,                        # model being run 
        dummy_input,                                # model input (or a tuple for multiple inputs) 
        self.opt.make_onnx,                         # where to save the model  
        export_params=True,                         # store the trained parameter weights inside the model file 
        opset_version=11,                           # the ONNX version to export the model to 
        #do_constant_folding=True,                  # whether to execute constant folding for optimization 
        input_names = ['modelInput'],               # the model's input names 
        output_names = ['hm', 'wh', 'id', 'reg'])   # the model's output names 
        #keep_initializers_as_inputs=True) 
    print(" ") 
    print('Model has been converted to ONNX') 