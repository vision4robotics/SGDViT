# SGDViT: Saliency-Guided Dynamic vision Transformer for UAV tracking

### Liangliang Yao, Changhong Fu, Sihang Li, Guangze Zheng and Junjie Ye

## Abstract
Vision-based object tracking has boosted extensive autonomous applications for unmanned aerial vehicles (UAVs). However, the frequent maneuvering flight and viewpoint change are prone to cause nerve-wracking challenges, e.g., aspect ratio change and scale variation. The cross-correlation operation’s weak ability to mine perceptual similarity and easy introduction of background information become more apparent when confronted with these challenges. To address these issues, this work proposes a novel saliency-guided dynamic vision Transformer (SGDViT) for UAV tracking. Specifically, a UAV task-oriented object saliency mining network is designed to refine the perceptual similarity indicated by cross-correlation operation, distinguishing the foreground and background preliminarily. Furthermore, an innovative saliency adaption embedding operation is developed to generate dynamic tokens based on the initial saliency, reducing the computational complexity of the Transformer structure. On this bases, a lightweight saliency filtering Transformer is implemented to refine the saliency information and increase attention to the appearance
information. Comprehensive evaluations on three authoritative UAV tracking benchmarks and real-world tests have proven the effectiveness and robustness of the proposed method. 

![Workflow of our tracker](https://github.com/vision4robotics/SGDViT/blob/main/imgs/2.png)

This figure shows the workflow of our tracker.
## Demo

- 📹 Demo of real-world SGDViT tests.
- Refer to [Test1](https://www.bilibili.com/video/BV19e4y187Km/?vd_source=4bf245fe6a4c3e4931ad481b87f324ae) and [Test2](https://www.bilibili.com/video/BV12d4y1678S/?vd_source=4bf245fe6a4c3e4931ad481b87f324ae) on Youtube for more real-world tests.

## About Code
### 1. Environment setup
This code has been tested on Ubuntu 18.04, Python 3.8.3, Pytorch 0.7.0/1.6.0, CUDA 10.2.
Please install related libraries before running this code: 
```bash
pip install -r requirements.txt
```

### 2. Test
Download pretrained model: [result](https://pan.baidu.com/s/1xr0aLdXmuJc7CbAQRrEwPQ?pwd=x7jh)(code: x7jh) and put it into `tools/snapshot` directory.

Download testing datasets and put them into `test_dataset` directory. If you want to test the tracker on a new dataset, please refer to [pysot-toolkit](https://github.com/StrangerZhang/pysot-toolkit) to set test_dataset.

```bash 
python test.py                                
	--dataset UAV10fps                 #dataset_name
	--snapshot snapshot/result.pth  # tracker_name
```
The testing result will be saved in the `results/dataset_name/tracker_name` directory.

### 3. Train

#### Prepare training datasets

Download the datasets：
* [VID](http://image-net.org/challenges/LSVRC/2017/)
* [YOUTUBEBB](https://pan.baidu.com/s/1ZTdfqvhIRneGFXur-sCjgg) (code: t7j8)
* [COCO](http://cocodataset.org)
* [GOT-10K](http://got-10k.aitestunion.com/downloads)


**Note:** `train_dataset/dataset_name/readme.md` has listed detailed operations about how to generate training datasets.


#### Train a model
To train the model, run `train.py` with the desired configs:

```bash
cd tools
python train.py
```

### 4. Evaluation
We provide the tracking [results](https://pan.baidu.com/s/1x8_GRMRfApIyt4l87dORCg?pwd=l9qy) (code: l9qy) of UAV123@10fps, DTB70, and UAVTrack112. If you want to evaluate the tracker, please put those results into  `results` directory.
```
python eval.py 	                          \
	--tracker_path ./results          \ # result path
	--dataset UAV10                   \ # dataset_name
	--tracker_prefix 'result'   # tracker_name
```

### 5. Contact
If you have any questions, please contact me.

Liangliang Yao

Email: [1951018@tongji.edu.cn](1951018@tongji.edu.cn)



## Acknowledgement
The code is implemented based on [pysot](https://github.com/STVIR/pysot), [HiFT](https://github.com/vision4robotics/HiFT), and [Swin-T] (https://github.com/microsoft/Swin-Transformer). We would like to express our sincere thanks to the contributors.