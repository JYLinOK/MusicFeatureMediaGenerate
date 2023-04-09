import argparse
from pathlib import Path
import os

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from mynet import transnet as net 



def used_transform(size):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))

    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def style_transfer(vgg, decoder, SCT, content, style, alpha=1.0):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    feat = SCT(content_f, style_f)
    return decoder(feat)


# ===============================================================================
SCT_path = './SCT_decoder/artistic/sct_iter_160000.pth.tar'
decoder_path = './SCT_decoder/artistic/decoder_iter_160000.pth.tar'
testing_mode_used = 'art'
vgg_model_path = './models/vgg_normalised.pth'
output_dir = './output_trans'
# ===============================================================================

parser = argparse.ArgumentParser()

parser.add_argument('--testing_mode', default=testing_mode_used,
                    help='Artistic or Photo-realistic')

parser.add_argument('--vgg', type=str, default=vgg_model_path)
parser.add_argument('--decoder', type=str, default=decoder_path, help='put the trained decoder here')
parser.add_argument('--SCT', type=str, default=SCT_path, help='put the trained SCT module here')

parser.add_argument('--content_size', type=int, default=170)
parser.add_argument('--style_size', type=int, default=170)

parser.add_argument('--img_format', default='.jpg')

parser.add_argument('--output_dir', type=str, default='output')

args = parser.parse_args()

# ===============================================================================

interpolation = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True, parents=True)

decoder = net.decoder
vgg = net.vgg
network = net.Net(vgg, decoder, args.testing_mode)
SCT = network.SCT

SCT.eval()
decoder.eval()
vgg.eval()

decoder.load_state_dict(torch.load(args.decoder))
vgg.load_state_dict(torch.load(args.vgg))
SCT.load_state_dict(torch.load(args.SCT))

vgg = nn.Sequential(*list(vgg.children())[:31]) if args.testing_mode == 'art' else nn.Sequential(*list(vgg.children())[:18])
decoder = decoder if args.testing_mode == 'art' else nn.Sequential(*list(net.decoder.children())[10:])

vgg.to(device)
decoder.to(device)
SCT.to(device)

content_trans = used_transform(args.content_size)
style_trans = used_transform(args.style_size)


def trans_content(content_path, style_path, output_dir, alpha=0.6):
    print(f'\n{content_path = }')
    print(f'{style_path = }')
    print(f'{output_dir = }')

    output_filename = os.path.basename(content_path)
    output_subDir_name = os.path.basename(os.path.dirname(content_path))
    output_subDir = os.path.join(output_dir, output_subDir_name)
    output_path = os.path.join(output_subDir, output_filename)

    if not os.path.exists(output_subDir): os.mkdir(output_subDir)

    print(f'{output_subDir = }')
    print(f'{output_path = }')

    if interpolation:  
        # process one content image with N style image
        style = torch.stack([style_trans(Image.open(str(style_path)))])
        content = content_trans(Image.open(str(content_path))).unsqueeze(0).expand_as(style)
        style = style.to(device)
        content = content.to(device)

        with torch.no_grad():
            output = style_transfer(vgg, decoder, SCT, content, style, alpha)

        output = output.cpu()
        save_image(output, output_path)

    # process one content with only one style
    content = content_trans(Image.open(str(content_path)))
    style = style_trans(Image.open(str(style_path)))
    style = style.to(device).unsqueeze(0)
    content = content.to(device).unsqueeze(0)

    with torch.no_grad():
        output = style_transfer(vgg, decoder, SCT, content, style, alpha)
        
    output = output.cpu()

    save_image(output, str(output_path))



if __name__ == '__main__':
    # transform processing
    Test = False
    if Test:
        trans_content('../music_jpg170/1 A comme amour/1.jpg', '../feature_img/1 A comme amour/1.jpg', 'output_trans')

