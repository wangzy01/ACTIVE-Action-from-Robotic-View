# ACTIVE: Action from Robotic View 🤖

[![🌐 Project Page](https://img.shields.io/badge/🌐-Project%20Page-blue?style=for-the-badge)](https://wangzy01.github.io/ACTIVE/index.html) &nbsp; [![📄 arXiv](https://img.shields.io/badge/📄-arXiv-red?style=for-the-badge)](https://arxiv.org/abs/2507.22522) &nbsp; [![🤗 Download Dataset](https://img.shields.io/badge/HuggingFace-Download-grey?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/ACTIVE2750/ACTIVE) &nbsp; [![Baidu Netdisk](https://img.shields.io/badge/Baidu%20Netdisk-Code%3A%20Ks6T-blue?style=for-the-badge)](https://pan.baidu.com/s/1N1agrCKDNMvQNP7C7jMDeg?pwd=Ks6T)

This repository contains the official PyTorch implementation and dataset for our paper: **"Recognizing Actions from Robotic View for Natural Human-Robot Interaction"**, accepted at ICCV 2025.

![pipeline](assets/pipeline.png)

-----



## 📜 About The Project

Natural Human-Robot Interaction (N-HRI) requires robots to recognize human actions at varying distances and states, regardless of whether the robot itself is in motion or stationary. This setup is more flexible and practical than conventional human action recognition tasks. However, existing benchmarks designed for traditional action recognition fail to address the unique complexities in N-HRI due to limited data, modalities, task categories, and diversity of subjects and environments. To address these challenges, we introduce ACTIVE (Action from Robotic View), a large-scale dataset tailored specifically for perception-centric robotic views prevalent in mobile service robots. ACTIVE comprises 30 composite action categories, 80 participants, and 46,868 annotated video instances, covering both RGB and point cloud modalities. Participants performed various human actions in diverse environments at distances ranging from 3m to 50m, while the camera platform was also mobile, simulating real-world scenarios of robot perception with varying camera heights due to uneven ground. This comprehensive and challenging benchmark aims to advance action and attribute recognition research in N-HRI. Furthermore, we propose ACTIVE-PC, a method that accurately perceives human actions at long distances using Multilevel Neighborhood Sampling, Layered Recognizers, Elastic Ellipse Query, and precise decoupling of kinematic interference from human actions. Experimental results demonstrate the effectiveness of ACTIVE-PC.

-----

## 🚀 Getting Started

Follow these steps to set up the environment and run our code.

### Prerequisites

  * Python 3.8+
  * PyTorch 1.12.0+
  * CUDA 11.3+

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/ACTIVE.git
    cd ACTIVE
    ```

2.  **Create a virtual environment and install dependencies:**

    ```bash
    # We recommend using conda or venv
    pip install -r requirements.txt
    ```

3.  **Compile custom CUDA operators:**
    Our model requires custom CUDA extensions for efficient point cloud processing.

      * **PointNet++ Layers:**
        ```bash
        cd modules/
        python setup.py install
        cd ..
        ```
      * **k-Nearest Neighbors (kNN):**
        ```bash
        pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
        ```

For a detailed environment specification, please see the `requirements.yml` file.

-----

## 📊 Dataset

You can download the dataset from either **Hugging Face** or **Baidu Netdisk**:

[![Download Dataset HuggingFace](https://img.shields.io/badge/HuggingFace-Download-lightgrey?style=for-the-badge&logo=huggingface)](https://huggingface.co/datasets/ACTIVE2750/ACTIVE) &nbsp; [![Download Dataset Baidu](https://img.shields.io/badge/Baidu%20Netdisk-Code%3A%20Ks6T-blue?style=for-the-badge)](https://pan.baidu.com/s/1N1agrCKDNMvQNP7C7jMDeg?pwd=Ks6T)


### File Naming Convention

The video files in the dataset follow the format `PxxxSxxxAxxxRxxx`. Each component represents:

| **Component** | **Meaning**    | **Description**                       |
| ------------- | -------------- | ------------------------------------- |
| **P**         | **Person**     | Subject ID (e.g., `P002` is Person 2) |
| **S**         | **Scene**      | Scene ID                              |
| **A**         | **Action**     | Action Label                          |
| **R**         | **Repetition** | Repetition Count                      |

Example: `P002S003A001R001` means Person 2, Scene 3, Action 1, Repetition 1.

### Data Preparation (Meta Files)

The training script (`active-train.py`) requires a meta file path (passed via `--data-meta`). **This file is not included in the raw dataset download**; you should generate it based on the downloaded files.

Meta File Format:

It is a text file where each line contains the video name and the frame count, separated by a space.

Plaintext

```
P002S003A001R001 24
P002S003A001R002 30
...
```


### Evaluation Protocols

For reproducibility, we follow a standard **Cross-Subject Evaluation** protocol. The 80 participants are split as follows:

  * **Training Set:** 53 subjects
  * **Testing Set:** 27 subjects

The participant IDs used for the **training group** are:
`2, 3, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 28, 30, 34, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 52, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 76, 77, 78, 79`

The remaining 27 subjects are used for testing.

-----

## ⚡️ Usage

### Training

`active-train.py` **trains** on the training split and **evaluates** on the test split at the end of **every epoch**. CUDA is required (the script sets `device='cuda'`).

```bash
python active-train.py \
  --data-path /path/to/ACTIVE \
  --data-meta /path/to/split_or_meta_file \
  --output-dir ./outputs/activepc \
```

**Resume training:**

```bash
python active-train.py \
  --data-path /path/to/ACTIVE \
  --data-meta /path/to/split_or_meta_file \
  --output-dir ./outputs/activepc \
  --resume ./outputs/activepc/checkpoint.pth
```


### Testing (during training)

No separate command is required to “enable testing.” The script **always runs evaluation** on the test split each epoch and logs:


### Test-only (how to implement)

`active-train.py` does not include a built-in “test-only” CLI mode. If you want to evaluate a checkpoint without running another training epoch, add a tiny flag and early return:

1. **Add an argument** in `parse_args()`:

```python
parser.add_argument('--test-only', action='store_true',
                    help='Run evaluation on the test split and exit')
```

2. **Short-circuit in `main(args)` after loading the model & checkpoint** and after building `data_loader_test`:

```python
if args.test_only:
    if not args.resume:
        logging.warning('No --resume provided; evaluating random-initialized weights.')
    evaluate(model, criterion, data_loader_test, device=device)
    return
```

Then you can run:

```bash
python active-train.py \
  --data-path /path/to/ACTIVE \
  --data-meta /path/to/split_or_meta_file \
  --model ACTIVEPC \
  --resume /path/to/model_or_checkpoint.pth \
  --test-only
```

This will load the checkpoint and only run the evaluation once on the test split, then exit.


-----

## 🙏 Acknowledgements

This work is built upon the foundational codebase of [PSTNet](https://github.com/hehefan/Point-Spatio-Temporal-Convolution).

-----

## ✍️ Citation

If you use our dataset or model in your research, please consider citing our paper:

```bibtex
@inproceedings{wang2025recognizing,
  title={Recognizing Actions from Robotic View for Natural Human-Robot Interaction},
  author={Wang, Ziyi and Li, Peiming and Liu, Hong and Deng, Zhichao and Wang, Can and Liu, Jun and Yuan, Junsong and Liu, Mengyuan},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={14218--14227},
  year={2025}
}
```
