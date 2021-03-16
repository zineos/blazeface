import math
import glob
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib.pyplot as plt
from kmeans import kmeans, avg_iou, iou
from tqdm import tqdm

# ANNOTATIONS_PATH = "./data/pascalvoc07-annotations"
ANNOTATIONS_PATH = "/data/face_detections/blazefacev3/blaceface/utils/anchors_calculation/data/widerface-annotations"
CLUSTERS = 40
BBOX_NORMALIZE = True
input_dim = 128

def show_cluster(data, cluster, max_points=2000):
	'''
	Display bouding box's size distribution and anchor generated in scatter.
	'''
	if len(data) > max_points:
		idx = np.random.choice(len(data), max_points)
		data = data[idx]
	plt.scatter(data[:,0], data[:,1], s=5, c='lavender')
	plt.scatter(cluster[:,0], cluster[:, 1], c='red', s=100, marker="^")
	plt.xlabel("Width")
	plt.ylabel("Height")
	plt.title("Bounding and anchor distribution")
	plt.savefig("cluster.png")
	plt.show()

def show_width_height(data, cluster, bins=50):
	'''
	Display bouding box distribution with histgram.
	'''
	if data.dtype != np.float32:
		data = data.astype(np.float32)
	width = data[:, 0]
	height = data[:, 1]
	ratio = height / width

	plt.figure(1,figsize=(20, 6))
	plt.subplot(131)
	plt.hist(width, bins=bins, color='green')
	plt.xlabel('width')
	plt.ylabel('number')
	plt.title('Distribution of Width')

	plt.subplot(132)
	plt.hist(height,bins=bins, color='blue')
	plt.xlabel('Height')
	plt.ylabel('Number')
	plt.title('Distribution of Height')

	plt.subplot(133)
	plt.hist(ratio, bins=bins,  color='magenta')
	plt.xlabel('Height / Width')
	plt.ylabel('number')
	plt.title('Distribution of aspect ratio(Height / Width)')
	plt.savefig("shape-distribution.png")
	plt.show()
	

def sort_cluster(cluster):
	'''
	Sort the cluster to with area small to big.
	'''
	if cluster.dtype != np.float32:
		cluster = cluster.astype(np.float32)
	area = cluster[:, 0] * cluster[:, 1]
	cluster = cluster[area.argsort()]
	ratio = cluster[:,1:2] / cluster[:, 0:1]
	return np.concatenate([cluster, ratio], axis=-1)


def evolve(k, wh, gen=1000):


	def metric(k, wh):  # compute metrics
			# r = wh[:, None] / k[None]
			# x = torch.min(r, 1. / r).min(2)[0]  # ratio metric
			# x = wh_iou(wh, torch.tensor(k))  # iou metric
			x = avg_iou(wh, k)
			return x  # x, best_x


	def fitness(k):  # mutation fitness
			res = metric(k, wh)
			return res  # fitness

	
	thr = 0.0001
	npr = np.random
	f, sh, mp, s = fitness(k), k.shape, 0.9, 0.1  # fitness, generations, mutation prob, sigma
	pbar = tqdm(range(gen), desc='Evolving anchors with Genetic Algorithm')  # progress bar
	for _ in pbar:
		v = np.ones(sh)
		while (v == 1).all():  # mutate until a change occurs (prevent duplicates)
			v = ((npr.random(sh) < mp) * npr.random() * npr.randn(*sh) * s + 1).clip(0.00001, 10)
		kg = (k * v).clip(min=0.00001)
		fg = fitness(kg)
		if fg > f:
			f, k = fg, kg
			pbar.desc = 'Evolving anchors with Genetic Algorithm: fitness = %.4f' % f

	return k


def load_dataset(path, normalized=True):
	'''
	load dataset from pasvoc formatl xml files
	'''
	dataset = []
	for xml_file in glob.glob("{}/*xml".format(path)):
		tree = ET.parse(xml_file)

		height = int(tree.findtext("./size/height"))
		width = int(tree.findtext("./size/width"))

		for obj in tree.iter("object"):
			if normalized:
				xmin = int(obj.findtext("bndbox/xmin")) / float(width)
				ymin = int(obj.findtext("bndbox/ymin")) / float(height)
				xmax = int(obj.findtext("bndbox/xmax")) / float(width)
				ymax = int(obj.findtext("bndbox/ymax")) / float(height)
				w = (xmax - xmin) * float(width)
				h = (ymax - ymin) * float(height)

			else:
				xmin = int(obj.findtext("bndbox/xmin")) 
				ymin = int(obj.findtext("bndbox/ymin")) 
				xmax = int(obj.findtext("bndbox/xmax")) 
				ymax = int(obj.findtext("bndbox/ymax"))
				w = xmax - xmin 
				h = ymax - ymin 
			
			if w == 0 or h== 0:
				continue # to avoid divded by zero error.
			if w >= 2 and h >= 2:
				dataset.append([xmax - xmin, ymax - ymin])

	return np.array(dataset)

print("Start to load data annotations on: %s" % ANNOTATIONS_PATH)
data = load_dataset(ANNOTATIONS_PATH, normalized=BBOX_NORMALIZE)

print("Start to do kmeans, please wait for a moment.")

out = kmeans(data, k=CLUSTERS)
out_sorted = sort_cluster(out)
print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))

anchor_box = []
for i in range(len(out_sorted)):
	print("%.3f      %.3f     %.1f" % (out_sorted[i,0]*input_dim, out_sorted[i,1]*input_dim, out_sorted[i,2]))

	anchor_box.append( int(math.sqrt(out_sorted[i,0]*out_sorted[i,1])*input_dim) )

anchor_box = sorted(list(set(anchor_box)))

print("anchor box")
print(anchor_box)


print("Start to do evolve, please wait for a moment.")

out = evolve(out, data)
out_sorted = sort_cluster(out)
print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))

show_cluster(data, out, max_points=2000)

if out.dtype != np.float32:
	out = out.astype(np.float32)

print("Recommanded aspect ratios(width/height)")
print("Width    Height   Height/Width")

anchor_box = []
for i in range(len(out_sorted)):
	print("%.3f      %.3f     %.1f" % (out_sorted[i,0]*input_dim, out_sorted[i,1]*input_dim, out_sorted[i,2]))

	anchor_box.append( int(math.sqrt(out_sorted[i,0]*out_sorted[i,1])*input_dim) )

anchor_box = sorted(list(set(anchor_box)))

print("anchor box")
print(anchor_box)
show_width_height(data, out, bins=50)
