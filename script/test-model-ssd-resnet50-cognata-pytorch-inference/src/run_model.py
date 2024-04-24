"""
@author: Viet Nguyen <nhviet1009@gmail.com>

Extended by Radoeyh and Grigori Fursin
"""

print ('Initializing packages for ABTF PyTorch model...')
print ('')

import numpy as np
import argparse
import importlib

import time
import copy

import torch

import cv2
from PIL import Image

# From ABTF code
from src.transform import SSDTransformer
from src.utils import generate_dboxes, Encoder, colors, coco_classes
from src.model import SSD, ResNet

# Cognata dataset labels
import cognata_labels

def get_args():
    parser = argparse.ArgumentParser("Implementation of SSD")
    parser.add_argument("--data-path", type=str, default=None, help="the root folder of dataset")
    parser.add_argument("--input", type=str, default=None, help="the path to input image")
    parser.add_argument("--output", type=str, default=None, help="the path to output image")
    parser.add_argument("--cls-threshold", type=float, default=0.3)
    parser.add_argument("--nms-threshold", type=float, default=0.5)
    parser.add_argument("--pretrained-model", type=str, default="trained_models/SSD.pth")
    parser.add_argument("--dataset", default='Cognata', type=str)
    parser.add_argument("--config", default='config', type=str)
    parser.add_argument("--num-classes", type=int)
    parser.add_argument("--visualize", dest='visualize', action='store_const', default=False, const=True)
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



    # Prepare PyTorch model
    model = SSD(config.model, backbone=ResNet(config.model), num_classes=num_classes)

    checkpoint = torch.load(opt.pretrained_model, map_location=torch.device(device))

    model.load_state_dict(checkpoint["model_state_dict"])

    if device=='cuda':
        model.cuda()

    # Set model to inference
    model.eval()

    data_path = opt.data_path
    input_file = opt.input

    if data_path != '':
        import glob
            
        print ('')
        print ('Searching for images in the data set path ...')
        files = glob.glob(os.path.join(data_path, '**', '*.png'), recursive=True)

        print ('')
        print ('Found {} files!'.format(len(files)))
    else:
        files = [input_file]
    

    # Simple iteration over files without batching/optimizations for testing/demos
    for f in files:
        print ('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print ('Processing input: {}'.format(f))

        # Load image
        img = Image.open(f).convert("RGB")

        width0, height0 = img.size

        # Prepare boxes
        dboxes = generate_dboxes(config.model, model="ssd")
        transformer = SSDTransformer(dboxes, image_size, val=True)
        img, _, _, _ = transformer(img, None, torch.zeros(1,4), torch.zeros(1))
        encoder = Encoder(dboxes)

        _, height, width = img.shape

        print ('')
        print ('Original image size: {} x {}'.format(width0, height0))
        print ('Transformed image size: {} x {}'.format(width, height))

        if torch.cuda.is_available():
            img = img.cuda()

        with torch.no_grad():
            inp = img.unsqueeze(dim=0)

            ###################################################################
            # Save in pickle format for MLPerf loadgen tests
            # https://github.com/mlcommons/ck/tree/dev/cm-mlops/script/app-loadgen-generic-python

            input_pickle_file = f + '.' + device + '.pickle'

            import pickle
            with open(input_pickle_file, 'wb') as handle:
                pickle.dump(inp, handle)

            print ('')
            print ('Recording input image tensor to pickle: {}'.format(input_pickle_file))
            print ('  Input type: {}'.format(type(inp)))
            print ('  Input shape: {}'.format(inp.shape))

            print ('')
            print ('Running ABTF PyTorch model ...')

            t1 = time.time()
            
            ploc, plabel = model(inp)

            print ('')
            print ('ploc:')
            print (ploc)
            print ('')
            print ('plabel:')
            print (plabel)

            # Grigori's note: decode_batch will update ploc2 - copy for later comparison with ONNX if needed
            ploc_copy = copy.deepcopy(ploc)

            result = encoder.decode_batch(ploc, plabel, opt.nms_threshold, 20)[0]

            print ('')
            print ('result:')
            print (result)

            ##########################################################################
            # Grigori used these tutorials: 
            #  * https://pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html
            #  * https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html

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
                     opset_version=17
                     )


     #            print ('')
     #            print ('Loading exported ONNX model ...')
     #            import onnx
     #            onnx_model = onnx.load(to_export_model)
     #            onnx.checker.check_model(onnx_model)

                print ('')
                print ('Running ABTF ONNX model to compare output with PyTorch model ...')

                import onnxruntime

                ort_session = onnxruntime.InferenceSession(to_export_model, providers=['CPUExecutionProvider'])

                def to_numpy(tensor):
                    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

                ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(inp)}

                ort_outputs = ort_session.run(None, ort_inputs)



                ploc2 = torch.from_numpy(ort_outputs[0]) #ploc2_numpy)
                plabel2 = torch.from_numpy(np.array(ort_outputs[1]))


                print ('')
                print ('ORT ploc:')
                print (ploc2)
                print ('')
                print ('ORT plabel:')
                print (plabel2)

                # Grigori's note: decode_batch will update ploc2 !
                result2 = encoder.decode_batch(ploc2, plabel2, opt.nms_threshold, 20)[0]

                print ('')
                print ('ORT result:')
                print (result2)


                try:
                   np.testing.assert_allclose(ploc, ploc2, rtol=1e-03, atol=1e-05)
                except Exception as e:
                   print ('')
                   print ('PyTorch ploc and ONNX ploc differ: {}'.format(e))

                try:
                   np.testing.assert_allclose(plabel, plabel2, rtol=1e-03, atol=1e-05)
                except Exception as e:
                   print ('')
                   print ('PyTorch plabel and ONNX plabel differ: {}'.format(e))


                exported = True     

                result = result2

            
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
            output_img = cv2.imread(f)

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

            # Save image

            if opt.output is None:
                output = "{}_prediction.jpg".format(f[:-4])
            else:
                output = opt.output

            print ('')
            print ('Recording output image with detect objects: {}'.format(output))
            cv2.imwrite(output, output_img)

            # Visualize for demos
            if opt.visualize:
                ratio = height/600
                img_resized = cv2.resize(output_img, (int(width/ratio), int(height/ratio)))

                cv2.imshow("ABTF model testing", img_resized)
                cv2.waitKey(1)



if __name__ == "__main__":
    opt = get_args()
    test(opt)
