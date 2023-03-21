from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import seaborn as sns
import torch
import numpy as np
import tqdm

from matplotlib.transforms import TransformedBbox
from matplotlib.transforms import Bbox
from matplotlib.image import BboxImage
from matplotlib.legend_handler import HandlerBase
from matplotlib._png import read_png

class ImageHandler(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize,
                       trans):

        # enlarge the image by these margins
        sx, sy = self.image_stretch

        # create a bounding box to house the image
        bb = Bbox.from_bounds(xdescent - sx,
                              ydescent - sy,
                              width + sx,
                              height + sy)

        tbb = TransformedBbox(bb, trans)
        image = BboxImage(tbb)
        image.set_data(self.image_data)
        print("legend : ",legend)
        self.update_prop(image, orig_handle, legend)

        return [image]
    def get_label(self):
        return self.label
    def set_image(self, image_path,label, image_stretch=(0, 0)):
        if not os.path.exists(image_path):
            print("There is no image")
            print(" image_path :",image_path)
            raise KeyboardInterrupt
            sample = get_sample_data("grace_hopper.png", asfileobj=False)
            self.image_data = read_png(sample)
        else:
            self.image_data = plt.imread(image_path)

            # self.image_data = read_png(image_path)

        self.image_stretch = image_stretch
        self.label = label

######

plot_bbox=False
img_list = np.loadtxt("img_path.out",dtype=str)
img_path_base = "/media/syh/ssd2/data/ReID_MUF_all/all"

sampling_num=5
feats=torch.load("features_msmt.pt")
# feats=torch.load("../features_msmt.pt")

feats = np.array(list(map(lambda o: o.data.cpu().numpy(),feats)))

print(feats.shape)
feats=feats[::sampling_num]


lists = [int(i.split('/')[-1][:4]) for i in img_list]
lists=lists[::sampling_num]
id_set = set(lists)
id_max = max(list(id_set))
print("max : ", max(list(id_set)))
# raise KeyboardInterrupt
img_list = img_list[::sampling_num]

cmap_lst = [plt.cm.rainbow, plt.cm.Blues, plt.cm.autumn, plt.cm.RdYlGn]
cmap = plt.cm.get_cmap('rainbow',id_max)
import cv2
from PIL import Image
def getImage(path, zoom=1,label=''):
    path=os.path.join(img_path_base,path[:4],path)
    img =cv2.imread(path)
    # color = label
    border_width = img.shape[0]//20
    top, bottom, left, right = [border_width] * 4
    # h,w=img.size
    color = [x * 255 for x in list(cmap(int(label)))]
    img_with_border = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    img_with_border = cv2.cvtColor(img_with_border, cv2.COLOR_BGR2RGB)
    # img.putalpha(128)
    # img =plt.imread(path)
    # img=frame_image(img, 20)
    # return OffsetImage(img, zoom=zoom/img.shape[0]*0.1)
    return OffsetImage(img_with_border, zoom=zoom/img.shape[0]*0.1)

# def frame_image(img, frame_width):
#     b = frame_width # border size in pixel
#     ny, nx = img.shape[0], img.shape[1] # resolution / number of pixels in x and y
#     if img.ndim == 3: # rgb or rgba array
#         framed_img = np.zeros((b+ny+b, b+nx+b, img.shape[2]))
#     elif img.ndim == 2: # grayscale image
#         framed_img = np.zeros((b+ny+b, b+nx+b))
#     framed_img[b:-b, b:-b] = img
#     return framed_img
tsne = TSNE(n_components=2, random_state=0,verbose=1)
# t-SNE 시각화 함수 정의
# def plot_vecs_n_labels(v, labels, fname):
#     fig = plt.figure(figsize=(10, 10))
#     plt.axis('off')
#     sns.set_style('darkgrid')
#     sns.scatterplot(v[:, 0], v[:, 1], hue=labels, legend='full', palette=sns.color_palette("bright", 10))
#     plt.legend(['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck'])
#     plt.savefig(fname)


# change batch_size
pred_tsne = tsne.fit_transform(feats)
cluster = np.array(pred_tsne)
print(cluster.shape)
labels = [str(i) for i in range(id_max)]
handles = []
fig, ax = plt.subplots()



handler_dict={}

for i, label in zip(range(id_max), labels):
    lists=np.array(lists)
    # print("list :",lists)
    idx = np.where(lists == i)
    custom_handler = ImageHandler()
    if len(img_list[idx])==0:
        custom_handler.set_image("empty.png",label=label,
                                 image_stretch=(0, 20))
    else:
        path = img_list[idx][0]
        custom_handler.set_image(os.path.join(path),label=label,
                                 image_stretch=(0, 20))
    handler_dict[label]=custom_handler
    # handles.append(custom_handler)


collections_list = []

ind=0
for scat, path in zip(cluster,img_list):
    ab = AnnotationBbox(getImage(path,zoom=500,label=lists[ind]), (scat[0], scat[1]), frameon=False)
    ax.add_artist(ab)
    ind+=1

for i, label in zip(range(id_max), labels):
    idx = np.where(lists == i)
    collections_list.append(ax.scatter(cluster[idx, 0], cluster[idx, 1], marker='s', label=label,cmap=cmap,alpha=1))

# print(collections_list[0].)
ax.legend(collections_list,labels,handler_map=handler_dict,labelspacing=2,
           frameon=False)
import matplotlib

for i in range(cmap.N):
    rgba = cmap(i)
    # rgb2hex accepts rgb or rgba
    print(matplotlib.colors.rgb2hex(rgba))
plt.savefig('output.png')
plt.show()
if plot_bbox:
    "/media/syh/ssd2/data/ReID_MUF_all/all", q_paths[q_idx][:4]

# for i in range(len(lists)):
#     pred = feats[i]
#     pred=pred.reshape(-1,1)
#     # 모델의 출력값을 tsne.fit_transform에 입력하기
#     pred_tsne = tsne.fit_transform(pred)
#
#     # t-SNE 시각화 함수 실행
#     plot_vecs_n_labels(pred_tsne, lists[i], 'tsen.png')
#     # break