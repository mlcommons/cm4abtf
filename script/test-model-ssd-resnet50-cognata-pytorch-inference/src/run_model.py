"""
@author: Viet Nguyen <nhviet1009@gmail.com>

Extended by Radoeyh and Grigori Fursin
"""

print ('Initializing packages for ABTF PyTorch model...')
print ('')

import numpy as np
import argparse
import importlib

import torch

import cv2
from PIL import Image

# From ABTF code
from src.transform import SSDTransformer
from src.utils import generate_dboxes, Encoder, colors, coco_classes
from src.model import SSD, ResNet

# Cognata labels
import cognata_labels

def get_args():
    parser = argparse.ArgumentParser("Implementation of SSD")
    parser.add_argument("--input", type=str, required=True, help="the path to input image")
    parser.add_argument("--cls-threshold", type=float, default=0.3)
    parser.add_argument("--nms-threshold", type=float, default=0.5)
    parser.add_argument("--pretrained-model", type=str, default="trained_models/SSD.pth")
    parser.add_argument("--output", type=str, default=None, help="the path to output image")
    parser.add_argument("--dataset", default='Cognata', type=str)
    parser.add_argument("--config", default='config', type=str)
    parser.add_argument("--num-classes", type=int)
    args = parser.parse_args()
    return args


def test(opt):
    import os

    # Check remote debugging via CM (tested with Visual Studio)
    if os.environ.get('CM_TMP_DEBUG_UID', '') == '7cf735bf80204efb':
        import cmind.utils
        cmind.utils.debug_here(__file__, port=5678).breakpoint()

    device = os.environ.get('CM_DEVICE','')
    if device == 'cuda' and not torch.cuda.is_available():
        print ('')
        print ('Error: CUDA is forced but not available...')
        exit(1)

    to_export_model = os.environ.get('CM_ABTF_EXPORT_MODEL_TO_ONNX','')
    exported = False


    config = importlib.import_module('config.' + opt.config)
    image_size = config.model['image_size']

    # Some older ABTF models may have different number of classes.
    # In such cases, we can force the number via command line
    num_classes = opt.num_classes
    if num_classes is None:
        num_classes = len(cognata_labels.label_info)

    print ('')
    print ('Number of classes for the model: {}'.format(num_classes))

    model = SSD(config.model, backbone=ResNet(config.model), num_classes=num_classes)

    checkpoint = torch.load(opt.pretrained_model, map_location=torch.device(device))

    model.load_state_dict(checkpoint["model_state_dict"])

    if device=='cuda':
        model.cuda()

    model.eval()

    dboxes = generate_dboxes(config.model, model="ssd")


    transformer = SSDTransformer(dboxes, image_size, val=True)
    img = Image.open(opt.input).convert("RGB")
    img, _, _, _ = transformer(img, None, torch.zeros(1,4), torch.zeros(1))
    encoder = Encoder(dboxes)


    _, height, width = img.shape



    if torch.cuda.is_available():
        img = img.cuda()

    with torch.no_grad():
        inp = img.unsqueeze(dim=0)

        ###################################################################
        # Save in pickle format for MLPerf loadgen tests
        # https://github.com/mlcommons/ck/tree/dev/cm-mlops/script/app-loadgen-generic-python

        input_pickle_file = opt.input+'.'+device+'.pickle'
        import pickle
        with open(input_pickle_file, 'wb') as handle:
            pickle.dump(inp, handle)

        print ('')
        print ('Recording input image tensor to pickle: {}'.format(input_pickle_file))
        print ('  Input type: {}'.format(type(inp)))
        print ('  Input shape: {}'.format(inp.shape))

        print ('')
        print ('Running ABTF PyTorch model ...')

        import time
        t1 = time.time()
        
        ploc, plabel = model(inp)



        result = encoder.decode_batch(ploc, plabel, opt.nms_threshold, 20)[0]




        if to_export_model!='' and not exported:
            print ('')
            print ('Exporting ABTF PyTorch model to ONNX format ...')

            torch.onnx.export(model,
                 inp,
                 to_export_model,
                 verbose=False,
                 input_names=['input'],
                 output_names=['output'],
                 export_params=True,
                 )


            print ('')
            print ('Loading exported ONNX model ...')

            import onnx
            onnx_model = onnx.load(to_export_model)
            onnx.checker.check_model(onnx_model)

            print ('')
            print ('Running ABTF ONNX model ...')

            import onnxruntime

            onnx_input = inp.numpy() #onnx_program.adapt_torch_inputs_to_onnx(inp)

            ort_session = onnxruntime.InferenceSession(to_export_model, providers=['CPUExecutionProvider'])


            onnxruntime_input = {'input': onnx_input}

            onnxruntime_outputs = ort_session.run(None, onnxruntime_input)

            ploc = torch.from_numpy(np.array(onnxruntime_outputs[0]))
            plabel = torch.from_numpy(np.array(onnxruntime_outputs[1]))


            result = encoder.decode_batch(ploc, plabel, opt.nms_threshold, 20)[0]

            exported = True     
        
        t = time.time() - t1

        print ('')
        print ('Elapsed time: {:0.2f} sec.'.format(t))
                

        # Process result
        loc, label, prob = [r.cpu().numpy() for r in result]

        # Remove boxes with low probability
        best = np.argwhere(prob > opt.cls_threshold).squeeze(axis=1)

        loc = loc[best]
        label = label[best]
        prob = prob[best]

        # Update input image with boxes and predictions
        output_img = cv2.imread(opt.input)

        if len(loc) > 0:
            height, width, _ = output_img.shape

            loc[:, 0::2] *= width
            loc[:, 1::2] *= height

            loc = loc.astype(np.int32)

            for box, lb, pr in zip(loc, label, prob):
                category = cognata_labels.label_info[lb]
                color = colors[lb]

                xmin, ymin, xmax, ymax = box

                cv2.rectangle(output_img, (xmin, ymin), (xmax, ymax), color, 2)

                text_size = cv2.getTextSize(category + " : %.2f" % pr, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]

                cv2.rectangle(output_img, (xmin, ymin), (xmin + text_size[0] + 3, ymin + text_size[1] + 4), color, -1)

                cv2.putText(
                    output_img, category + " : %.2f" % pr,
                    (xmin, ymin + text_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1,
                    (255, 255, 255), 1)

        if opt.output is None:
            output = "{}_prediction.jpg".format(opt.input[:-4])
        else:
            output = opt.output

        print ('')
        print ('Recording output image with detect objects: {}'.format(output))
        cv2.imwrite(output, output_img)


if __name__ == "__main__":
    opt = get_args()
    test(opt)
